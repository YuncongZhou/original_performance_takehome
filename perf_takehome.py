"""
Optimized kernel with index deduplication.

Key insight: Tree traversal has DETERMINISTIC index patterns:
- Round 0: All at idx=0 (1 unique)
- Round 10: ALL wrap back to idx=0! (children of level 9 all >= 1023)
- Rounds 1-4 and 11-14: progressively more unique (2, 4, 8, 16)
- Rounds 5-9 and 15: need full gather (32+ unique)

Strategy: Unroll all rounds, optimize early rounds with index deduplication.
Target: <1487 cycles
"""

from collections import defaultdict
import random

from problem import (
    Engine,
    DebugInfo,
    SLOT_LIMITS,
    VLEN,
    N_CORES,
    SCRATCH_SIZE,
    Machine,
    Tree,
    Input,
    HASH_STAGES,
    reference_kernel,
    build_mem_image,
    reference_kernel2,
)


class KernelBuilder:
    def __init__(self):
        self.instrs = []
        self.scratch = {}
        self.scratch_debug = {}
        self.scratch_ptr = 0
        self.const_map = {}

    def debug_info(self):
        return DebugInfo(scratch_map=self.scratch_debug)

    def add(self, engine, slot):
        self.instrs.append({engine: [slot]})

    def alloc_scratch(self, name=None, length=1):
        addr = self.scratch_ptr
        if name is not None:
            self.scratch[name] = addr
            self.scratch_debug[addr] = (name, length)
        self.scratch_ptr += length
        assert self.scratch_ptr <= SCRATCH_SIZE, f"Out of scratch: {self.scratch_ptr}"
        return addr

    def scratch_const(self, val, name=None):
        if val not in self.const_map:
            addr = self.alloc_scratch(name)
            self.add("load", ("const", addr, val))
            self.const_map[val] = addr
        return self.const_map[val]

    def scratch_const_vec(self, val, name=None):
        key = ("vec", val)
        if key not in self.const_map:
            addr = self.alloc_scratch(name, VLEN)
            scalar_addr = self.scratch_const(val)
            self.add("valu", ("vbroadcast", addr, scalar_addr))
            self.const_map[key] = addr
        return self.const_map[key]

    def build_kernel(
        self, forest_height: int, n_nodes: int, batch_size: int, rounds: int
    ):
        """
        Fully unrolled kernel with index deduplication.
        """
        n_vectors = batch_size // VLEN  # 32 vectors

        # ============ ALLOCATION ============
        tmp1 = self.alloc_scratch("tmp1")
        tmp2 = self.alloc_scratch("tmp2")
        tmp3 = self.alloc_scratch("tmp3")

        init_vars = [
            "rounds", "n_nodes", "batch_size", "forest_height",
            "forest_values_p", "inp_indices_p", "inp_values_p",
        ]
        for v in init_vars:
            self.alloc_scratch(v, 1)

        # All idx and val in scratch
        all_idx = [self.alloc_scratch(f"idx_{i}", VLEN) for i in range(n_vectors)]
        all_val = [self.alloc_scratch(f"val_{i}", VLEN) for i in range(n_vectors)]

        # Working buffers for 4 vectors per group (two address buffers for pipelining)
        v_node = [self.alloc_scratch(f"node_{i}", VLEN) for i in range(4)]
        v_tmp1 = [self.alloc_scratch(f"tmp1_{i}", VLEN) for i in range(4)]
        v_tmp2 = [self.alloc_scratch(f"tmp2_{i}", VLEN) for i in range(4)]
        v_addrs = [self.alloc_scratch(f"addrs_{i}", VLEN) for i in range(4)]
        v_addrs2 = [self.alloc_scratch(f"addrs2_{i}", VLEN) for i in range(4)]

        # ============ INITIALIZATION ============
        for i, v in enumerate(init_vars):
            self.add("load", ("const", tmp1, i))
            self.add("load", ("load", self.scratch[v], tmp1))

        v_zero = self.scratch_const_vec(0)
        v_one = self.scratch_const_vec(1)
        v_two = self.scratch_const_vec(2)
        v_n_nodes = self.scratch_const_vec(n_nodes)

        # Hash constants - some stages can use multiply_add for optimization
        # Stage 0: (a + c) + (a << 12) = a * 4097 + c → multiply_add(a, 4097, c)
        # Stage 2: (a + c) + (a << 5) = a * 33 + c → multiply_add(a, 33, c)
        # Stage 4: (a + c) + (a << 3) = a * 9 + c → multiply_add(a, 9, c)
        hash_const_vecs = []
        hash_multipliers = {0: 4097, 2: 33, 4: 9}  # Stages that can use multiply_add
        for hi, (op1, val1, op2, op3, val3) in enumerate(HASH_STAGES):
            if hi in hash_multipliers:
                # For multiply_add stages: store (multiplier_vec, additive_const_vec)
                mult_vec = self.scratch_const_vec(hash_multipliers[hi])
                const_vec = self.scratch_const_vec(val1)
                hash_const_vecs.append(('multiply_add', mult_vec, const_vec))
            else:
                # For regular stages: store (const1_vec, const3_vec)
                hash_const_vecs.append(('regular', self.scratch_const_vec(val1), self.scratch_const_vec(val3)))

        self.add("flow", ("pause",))

        # ============ LOAD ALL DATA ============
        for vi in range(n_vectors):
            self.add("load", ("const", tmp1, vi * VLEN))
            self.instrs.append({
                "alu": [("+", tmp2, self.scratch["inp_indices_p"], tmp1),
                        ("+", tmp3, self.scratch["inp_values_p"], tmp1)]
            })
            self.instrs.append({
                "load": [("vload", all_idx[vi], tmp2),
                         ("vload", all_val[vi], tmp3)]
            })

        # ============ PROCESS ALL 16 ROUNDS (FULLY UNROLLED) ============
        # Tree has 11 levels (0-10) with n_nodes = 2047
        # Round 0: all at level 0 (idx=0)
        # Round 10: at level 10, then wraps to 0 at end
        # Round 11: all at level 0 (idx=0, after round 10 wrap)
        # Rounds 12-15: same pattern as rounds 1-4
        for round_num in range(rounds):
            # Determine effective round for index deduplication
            # Rounds 0-10: effective = round_num
            # Rounds 11-15: effective = round_num - 11 (11->0, 12->1, ..., 15->4)
            if round_num <= 10:
                effective_round = round_num
            else:
                effective_round = round_num - 11  # 11->0, 12->1, ..., 15->4

            is_last_round = (round_num == rounds - 1)

            # Wrap check is only needed for round 10 (going from level 10 to 11, which wraps to 0)
            # All other rounds stay within bounds (levels 0-10 have indices 0-2046)
            needs_wrap_check = (round_num == 10)

            if effective_round == 0:
                # k=1: all indices at 0 (rounds 0 and 11)
                self._emit_round_single_idx(
                    all_idx, all_val, v_node, v_tmp1, v_tmp2,
                    hash_const_vecs, v_one, v_two, v_n_nodes, v_zero,
                    n_vectors, tmp1, idx_value=0, skip_index=is_last_round,
                    skip_wrap_check=not needs_wrap_check
                )
            elif effective_round == 1:
                # k=2: indices in {1, 2} (rounds 1 and 12)
                self._emit_round_k2(
                    all_idx, all_val, v_node, v_tmp1, v_tmp2,
                    hash_const_vecs, v_one, v_two, v_n_nodes, v_zero,
                    n_vectors, tmp1, tmp2, idx_base=1, skip_index=is_last_round,
                    skip_wrap_check=not needs_wrap_check
                )
            else:
                # Full gather for all other rounds
                self._emit_round_full_gather(
                    all_idx, all_val, v_node, v_tmp1, v_tmp2, v_addrs, v_addrs2,
                    hash_const_vecs, v_one, v_two, v_n_nodes, v_zero,
                    n_vectors, tmp1, skip_index=is_last_round,
                    skip_wrap_check=not needs_wrap_check
                )

        # ============ STORE ALL VALUES ============
        for vi in range(n_vectors):
            self.add("load", ("const", tmp1, vi * VLEN))
            self.instrs.append({"alu": [("+", tmp2, self.scratch["inp_values_p"], tmp1)]})
            self.instrs.append({"store": [("vstore", tmp2, all_val[vi])]})

        self.instrs.append({"flow": [("pause",)]})

    def _emit_round_single_idx(self, all_idx, all_val, v_node, v_tmp1, v_tmp2,
                                hash_const_vecs, v_one, v_two, v_n_nodes, v_zero,
                                n_vectors, tmp1, idx_value, skip_index=False, skip_wrap_check=False):
        """Emit round with single unique index (rounds 0 and 11)."""
        # Load single node value
        self.add("load", ("load", tmp1, self.scratch["forest_values_p"]))
        self.add("valu", ("vbroadcast", v_node[0], tmp1))

        # XOR all vectors with the single node value (3 at a time)
        for vi in range(0, n_vectors, 3):
            vecs = min(3, n_vectors - vi)
            if vecs == 3:
                self.instrs.append({
                    "valu": [("^", all_val[vi], all_val[vi], v_node[0]),
                             ("^", all_val[vi + 1], all_val[vi + 1], v_node[0]),
                             ("^", all_val[vi + 2], all_val[vi + 2], v_node[0])]
                })
            else:
                self.instrs.append({
                    "valu": [("^", all_val[vi + j], all_val[vi + j], v_node[0]) for j in range(vecs)]
                })

        # Hash all vectors
        self._emit_hash_all(all_val, v_tmp1, v_tmp2, hash_const_vecs, n_vectors)

        # Index update all vectors (skip for last round)
        if not skip_index:
            self._emit_index_all(all_idx, all_val, v_tmp1, v_one, v_two, v_n_nodes, v_zero, n_vectors,
                                 skip_wrap_check=skip_wrap_check)

    def _emit_round_k2(self, all_idx, all_val, v_node, v_tmp1, v_tmp2,
                       hash_const_vecs, v_one, v_two, v_n_nodes, v_zero,
                       n_vectors, tmp1, tmp2, idx_base, skip_index=False, skip_wrap_check=False):
        """Emit round with k=2 unique indices using VALU-based conditional select.
        Optimized: process 4 vectors at a time using 4 VALU slots."""
        # Load 2 node values: forest[idx_base] and forest[idx_base+1]
        self.add("load", ("const", tmp1, idx_base))
        self.add("load", ("const", tmp2, idx_base + 1))
        self.instrs.append({"alu": [("+", tmp1, self.scratch["forest_values_p"], tmp1),
                                    ("+", tmp2, self.scratch["forest_values_p"], tmp2)]})
        self.instrs.append({"load": [("load", tmp1, tmp1), ("load", tmp2, tmp2)]})
        self.instrs.append({"valu": [("vbroadcast", v_node[0], tmp1),
                                      ("vbroadcast", v_node[1], tmp2)]})

        # Create vector constant for idx_base
        idx_base_vec = self.scratch_const_vec(idx_base)

        # Precompute diff = node[0] ^ node[1] (used for all vectors)
        self.instrs.append({"valu": [("^", v_tmp2[3], v_node[0], v_node[1])]})

        # Process 6 vectors at a time to use all 6 VALU slots
        # Use v_tmp1[0:4] + v_tmp2[0:2] for 6 vectors (v_tmp2[3] has diff)
        for vi in range(0, n_vectors, 6):
            vecs = min(6, n_vectors - vi)
            # Map vectors to temp buffers
            tmps = [v_tmp1[0], v_tmp1[1], v_tmp1[2], v_tmp1[3], v_tmp2[0], v_tmp2[1]]

            if vecs == 6:
                # offset = idx - idx_base (0 or 1) for 6 vectors
                self.instrs.append({
                    "valu": [("-", tmps[j], all_idx[vi+j], idx_base_vec) for j in range(6)]
                })
                # mask = 0 - offset (0 or all-1s)
                self.instrs.append({
                    "valu": [("-", tmps[j], v_zero, tmps[j]) for j in range(6)]
                })
                # masked_diff = diff & mask
                self.instrs.append({
                    "valu": [("&", tmps[j], v_tmp2[3], tmps[j]) for j in range(6)]
                })
                # selected = node[0] ^ masked_diff
                self.instrs.append({
                    "valu": [("^", tmps[j], v_node[0], tmps[j]) for j in range(6)]
                })
                # XOR with val
                self.instrs.append({
                    "valu": [("^", all_val[vi+j], all_val[vi+j], tmps[j]) for j in range(6)]
                })
            else:
                # Handle remaining vectors (2 for 32 vectors)
                self.instrs.append({
                    "valu": [("-", tmps[j], all_idx[vi+j], idx_base_vec) for j in range(vecs)]
                })
                self.instrs.append({
                    "valu": [("-", tmps[j], v_zero, tmps[j]) for j in range(vecs)]
                })
                self.instrs.append({
                    "valu": [("&", tmps[j], v_tmp2[3], tmps[j]) for j in range(vecs)]
                })
                self.instrs.append({
                    "valu": [("^", tmps[j], v_node[0], tmps[j]) for j in range(vecs)]
                })
                self.instrs.append({
                    "valu": [("^", all_val[vi+j], all_val[vi+j], tmps[j]) for j in range(vecs)]
                })

        # Hash all vectors
        self._emit_hash_all(all_val, v_tmp1, v_tmp2, hash_const_vecs, n_vectors)

        # Index update all vectors (skip for last round)
        if not skip_index:
            self._emit_index_all(all_idx, all_val, v_tmp1, v_one, v_two, v_n_nodes, v_zero, n_vectors,
                                 skip_wrap_check=skip_wrap_check)

    def _emit_round_k2_preloaded(self, all_idx, all_val, v_node, v_tmp1, v_tmp2,
                                  hash_const_vecs, v_one, v_two, v_n_nodes, v_zero,
                                  n_vectors, tmp1, tmp2, idx_base, skip_index=False, skip_wrap_check=False,
                                  preloaded_node1=None, preloaded_node2=None):
        """Emit round with k=2 unique indices using preloaded node values.
        Uses VALU-based conditional select, skipping the load phase."""
        # Broadcast preloaded values (already in tmp1/tmp2 from preload)
        self.instrs.append({"valu": [("vbroadcast", v_node[0], preloaded_node1),
                                      ("vbroadcast", v_node[1], preloaded_node2)]})

        # Create vector constant for idx_base
        idx_base_vec = self.scratch_const_vec(idx_base)

        # Precompute diff = node[0] ^ node[1] (used for all vectors)
        self.instrs.append({"valu": [("^", v_tmp2[3], v_node[0], v_node[1])]})

        # Process 6 vectors at a time to use all 6 VALU slots
        tmps = [v_tmp1[0], v_tmp1[1], v_tmp1[2], v_tmp1[3], v_tmp2[0], v_tmp2[1]]
        for vi in range(0, n_vectors, 6):
            vecs = min(6, n_vectors - vi)

            if vecs == 6:
                self.instrs.append({
                    "valu": [("-", tmps[j], all_idx[vi+j], idx_base_vec) for j in range(6)]
                })
                self.instrs.append({
                    "valu": [("-", tmps[j], v_zero, tmps[j]) for j in range(6)]
                })
                self.instrs.append({
                    "valu": [("&", tmps[j], v_tmp2[3], tmps[j]) for j in range(6)]
                })
                self.instrs.append({
                    "valu": [("^", tmps[j], v_node[0], tmps[j]) for j in range(6)]
                })
                self.instrs.append({
                    "valu": [("^", all_val[vi+j], all_val[vi+j], tmps[j]) for j in range(6)]
                })
            else:
                self.instrs.append({
                    "valu": [("-", tmps[j], all_idx[vi+j], idx_base_vec) for j in range(vecs)]
                })
                self.instrs.append({
                    "valu": [("-", tmps[j], v_zero, tmps[j]) for j in range(vecs)]
                })
                self.instrs.append({
                    "valu": [("&", tmps[j], v_tmp2[3], tmps[j]) for j in range(vecs)]
                })
                self.instrs.append({
                    "valu": [("^", tmps[j], v_node[0], tmps[j]) for j in range(vecs)]
                })
                self.instrs.append({
                    "valu": [("^", all_val[vi+j], all_val[vi+j], tmps[j]) for j in range(vecs)]
                })

        # Hash all vectors
        self._emit_hash_all(all_val, v_tmp1, v_tmp2, hash_const_vecs, n_vectors)

        # Index update all vectors (skip for last round)
        if not skip_index:
            self._emit_index_all(all_idx, all_val, v_tmp1, v_one, v_two, v_n_nodes, v_zero, n_vectors,
                                 skip_wrap_check=skip_wrap_check)

    def _emit_round_k4_optimized(self, all_idx, all_val, v_node, v_tmp1, v_tmp2,
                                  hash_const_vecs, v_one, v_two, v_n_nodes, v_zero,
                                  n_vectors, tmp1, tmp2, idx_base, skip_index=False):
        """Emit round with k=4 unique indices, processing 4 vectors at a time."""
        # Load 4 node values
        for i in range(4):
            self.add("load", ("const", tmp1, idx_base + i))
            self.instrs.append({"alu": [("+", tmp1, self.scratch["forest_values_p"], tmp1)]})
            self.add("load", ("load", tmp1, tmp1))
            self.add("valu", ("vbroadcast", v_node[i], tmp1))

        idx_base_vec = self.scratch_const_vec(idx_base)
        v_two_const = self.scratch_const_vec(2)

        # Precompute diffs (reused for all vectors)
        # diff02 = node[0] ^ node[2], diff13 = node[1] ^ node[3]
        diff02 = v_tmp2[2]
        diff13 = v_tmp2[3]
        self.instrs.append({
            "valu": [("^", diff02, v_node[0], v_node[2]),
                     ("^", diff13, v_node[1], v_node[3])]
        })

        # Process 4 vectors at a time
        for vi in range(0, n_vectors, 4):
            # offset = idx - idx_base (0-3)
            self.instrs.append({
                "valu": [("-", v_tmp1[0], all_idx[vi], idx_base_vec),
                         ("-", v_tmp1[1], all_idx[vi+1], idx_base_vec),
                         ("-", v_tmp1[2], all_idx[vi+2], idx_base_vec),
                         ("-", v_tmp1[3], all_idx[vi+3], idx_base_vec)]
            })

            # Level 1: bit1 = (offset & 2) >> 1, mask1 = 0 - bit1
            self.instrs.append({
                "valu": [("&", v_tmp2[0], v_tmp1[0], v_two_const),
                         ("&", v_tmp2[1], v_tmp1[1], v_two_const)]
            })
            self.instrs.append({
                "valu": [(">>", v_tmp2[0], v_tmp2[0], v_one),
                         (">>", v_tmp2[1], v_tmp2[1], v_one),
                         ("&", v_tmp1[2], v_tmp1[2], v_two_const),
                         ("&", v_tmp1[3], v_tmp1[3], v_two_const)]
            })
            self.instrs.append({
                "valu": [("-", v_tmp2[0], v_zero, v_tmp2[0]),
                         ("-", v_tmp2[1], v_zero, v_tmp2[1]),
                         (">>", v_tmp1[2], v_tmp1[2], v_one),
                         (">>", v_tmp1[3], v_tmp1[3], v_one)]
            })
            self.instrs.append({
                "valu": [("-", v_tmp1[2], v_zero, v_tmp1[2]),
                         ("-", v_tmp1[3], v_zero, v_tmp1[3])]
            })
            # Now v_tmp2[0,1] and v_tmp1[2,3] have mask1 for 4 vectors

            # Select between (n0,n1) and (n2,n3) for all 4 vectors
            # We need to save offset for level 2, so use different temps
            # Store mask1 and compute sel_low, sel_high
            # sel_low = n0 ^ (diff02 & mask1), sel_high = n1 ^ (diff13 & mask1)
            self.instrs.append({
                "valu": [("&", v_tmp2[0], diff02, v_tmp2[0]),
                         ("&", v_tmp2[1], diff02, v_tmp2[1]),
                         ("&", v_tmp1[2], diff02, v_tmp1[2]),
                         ("&", v_tmp1[3], diff02, v_tmp1[3])]
            })
            self.instrs.append({
                "valu": [("^", v_tmp2[0], v_node[0], v_tmp2[0]),
                         ("^", v_tmp2[1], v_node[0], v_tmp2[1]),
                         ("^", v_tmp1[2], v_node[0], v_tmp1[2]),
                         ("^", v_tmp1[3], v_node[0], v_tmp1[3])]
            })
            # Now v_tmp2[0,1] and v_tmp1[2,3] have sel_low for 4 vectors

            # Recompute mask1 for sel_high (reuse offset calculation)
            self.instrs.append({
                "valu": [("-", v_node[0], all_idx[vi], idx_base_vec),
                         ("-", v_node[1], all_idx[vi+1], idx_base_vec),
                         ("-", v_node[2], all_idx[vi+2], idx_base_vec),
                         ("-", v_node[3], all_idx[vi+3], idx_base_vec)]
            })
            # Save offset in v_node for level 2
            # Compute mask1 again
            self.instrs.append({
                "valu": [("&", v_tmp1[0], v_node[0], v_two_const),
                         ("&", v_tmp1[1], v_node[1], v_two_const)]
            })
            self.instrs.append({
                "valu": [(">>", v_tmp1[0], v_tmp1[0], v_one),
                         (">>", v_tmp1[1], v_tmp1[1], v_one)]
            })
            self.instrs.append({
                "valu": [("-", v_tmp1[0], v_zero, v_tmp1[0]),
                         ("-", v_tmp1[1], v_zero, v_tmp1[1])]
            })
            # Compute sel_high for first 2 vectors
            self.instrs.append({
                "valu": [("&", v_tmp1[0], diff13, v_tmp1[0]),
                         ("&", v_tmp1[1], diff13, v_tmp1[1])]
            })
            self.instrs.append({
                "valu": [("^", v_tmp1[0], v_node[1], v_tmp1[0]),
                         ("^", v_tmp1[1], v_node[1], v_tmp1[1])]
            })
            # v_tmp1[0,1] have sel_high for vectors 0,1

            # Level 2: select between sel_low and sel_high based on bit 0
            # mask0 = 0 - (offset & 1)
            self.instrs.append({
                "valu": [("&", v_node[0], v_node[0], v_one),
                         ("&", v_node[1], v_node[1], v_one)]
            })
            self.instrs.append({
                "valu": [("-", v_node[0], v_zero, v_node[0]),
                         ("-", v_node[1], v_zero, v_node[1])]
            })
            # diff_lowhigh = sel_low ^ sel_high
            self.instrs.append({
                "valu": [("^", v_tmp1[0], v_tmp2[0], v_tmp1[0]),
                         ("^", v_tmp1[1], v_tmp2[1], v_tmp1[1])]
            })
            # masked = diff_lowhigh & mask0
            self.instrs.append({
                "valu": [("&", v_tmp1[0], v_tmp1[0], v_node[0]),
                         ("&", v_tmp1[1], v_tmp1[1], v_node[1])]
            })
            # final = sel_low ^ masked
            self.instrs.append({
                "valu": [("^", v_tmp1[0], v_tmp2[0], v_tmp1[0]),
                         ("^", v_tmp1[1], v_tmp2[1], v_tmp1[1])]
            })
            # XOR with val for vectors 0,1
            self.instrs.append({
                "valu": [("^", all_val[vi], all_val[vi], v_tmp1[0]),
                         ("^", all_val[vi+1], all_val[vi+1], v_tmp1[1])]
            })

            # Now do vectors 2,3 - need to compute sel_high for them
            # Offset is in v_node[2,3], sel_low is in v_tmp1[2,3]
            # Compute mask1 for vectors 2,3
            self.instrs.append({
                "valu": [("&", v_tmp2[0], v_node[2], v_two_const),
                         ("&", v_tmp2[1], v_node[3], v_two_const)]
            })
            self.instrs.append({
                "valu": [(">>", v_tmp2[0], v_tmp2[0], v_one),
                         (">>", v_tmp2[1], v_tmp2[1], v_one)]
            })
            self.instrs.append({
                "valu": [("-", v_tmp2[0], v_zero, v_tmp2[0]),
                         ("-", v_tmp2[1], v_zero, v_tmp2[1])]
            })
            # Compute sel_high for vectors 2,3
            self.instrs.append({
                "valu": [("&", v_tmp2[0], diff13, v_tmp2[0]),
                         ("&", v_tmp2[1], diff13, v_tmp2[1])]
            })
            self.instrs.append({
                "valu": [("^", v_tmp2[0], v_node[1], v_tmp2[0]),
                         ("^", v_tmp2[1], v_node[1], v_tmp2[1])]
            })
            # v_tmp2[0,1] have sel_high for vectors 2,3

            # Level 2 for vectors 2,3
            self.instrs.append({
                "valu": [("&", v_node[2], v_node[2], v_one),
                         ("&", v_node[3], v_node[3], v_one)]
            })
            self.instrs.append({
                "valu": [("-", v_node[2], v_zero, v_node[2]),
                         ("-", v_node[3], v_zero, v_node[3])]
            })
            self.instrs.append({
                "valu": [("^", v_tmp2[0], v_tmp1[2], v_tmp2[0]),
                         ("^", v_tmp2[1], v_tmp1[3], v_tmp2[1])]
            })
            self.instrs.append({
                "valu": [("&", v_tmp2[0], v_tmp2[0], v_node[2]),
                         ("&", v_tmp2[1], v_tmp2[1], v_node[3])]
            })
            self.instrs.append({
                "valu": [("^", v_tmp2[0], v_tmp1[2], v_tmp2[0]),
                         ("^", v_tmp2[1], v_tmp1[3], v_tmp2[1])]
            })
            # XOR with val for vectors 2,3
            self.instrs.append({
                "valu": [("^", all_val[vi+2], all_val[vi+2], v_tmp2[0]),
                         ("^", all_val[vi+3], all_val[vi+3], v_tmp2[1])]
            })

        # Hash all vectors
        self._emit_hash_all(all_val, v_tmp1, v_tmp2, hash_const_vecs, n_vectors)

        # Index update
        if not skip_index:
            self._emit_index_all(all_idx, all_val, v_tmp1, v_one, v_two, v_n_nodes, v_zero, n_vectors)

    def _emit_round_k4_correct(self, all_idx, all_val, v_node, v_tmp1, v_tmp2,
                                hash_const_vecs, v_one, v_two, v_n_nodes, v_zero,
                                n_vectors, tmp1, tmp2, idx_base):
        """Emit round with k=4 unique indices, processing 1 vector at a time.
        Uses careful buffer management to avoid corrupting node values."""
        # Load 4 node values
        for i in range(4):
            self.add("load", ("const", tmp1, idx_base + i))
            self.instrs.append({"alu": [("+", tmp1, self.scratch["forest_values_p"], tmp1)]})
            self.add("load", ("load", tmp1, tmp1))
            self.add("valu", ("vbroadcast", v_node[i], tmp1))

        idx_base_vec = self.scratch_const_vec(idx_base)
        v_two_const = self.scratch_const_vec(2)

        # Process 1 vector at a time (simple but correct)
        for vi in range(n_vectors):
            # offset = idx - idx_base (0-3)
            self.instrs.append({"valu": [("-", v_tmp1[0], all_idx[vi], idx_base_vec)]})

            # Level 1: select based on bit 1 (offset & 2)
            # mask1 = 0 - ((offset & 2) >> 1)
            self.instrs.append({"valu": [("&", v_tmp1[1], v_tmp1[0], v_two_const)]})
            self.instrs.append({"valu": [(">>", v_tmp1[1], v_tmp1[1], v_one)]})
            self.instrs.append({"valu": [("-", v_tmp1[1], v_zero, v_tmp1[1])]})

            # diff_02 = n0 ^ n2, diff_13 = n1 ^ n3 (use tmp2)
            self.instrs.append({
                "valu": [("^", v_tmp2[0], v_node[0], v_node[2]),
                         ("^", v_tmp2[1], v_node[1], v_node[3])]
            })
            # masked diffs
            self.instrs.append({
                "valu": [("&", v_tmp2[0], v_tmp2[0], v_tmp1[1]),
                         ("&", v_tmp2[1], v_tmp2[1], v_tmp1[1])]
            })
            # sel_low = n0 ^ masked, sel_high = n1 ^ masked (store in tmp2)
            self.instrs.append({
                "valu": [("^", v_tmp2[0], v_node[0], v_tmp2[0]),
                         ("^", v_tmp2[1], v_node[1], v_tmp2[1])]
            })

            # Level 2: select based on bit 0 (offset & 1)
            # mask0 = 0 - (offset & 1)
            self.instrs.append({"valu": [("&", v_tmp1[1], v_tmp1[0], v_one)]})
            self.instrs.append({"valu": [("-", v_tmp1[1], v_zero, v_tmp1[1])]})

            # diff = sel_low ^ sel_high
            self.instrs.append({"valu": [("^", v_tmp2[2], v_tmp2[0], v_tmp2[1])]})
            # masked diff
            self.instrs.append({"valu": [("&", v_tmp2[2], v_tmp2[2], v_tmp1[1])]})
            # final = sel_low ^ masked_diff
            self.instrs.append({"valu": [("^", v_tmp1[2], v_tmp2[0], v_tmp2[2])]})

            # XOR with val
            self.instrs.append({"valu": [("^", all_val[vi], all_val[vi], v_tmp1[2])]})

        # Hash all vectors
        self._emit_hash_all(all_val, v_tmp1, v_tmp2, hash_const_vecs, n_vectors)

        # Index update all vectors
        self._emit_index_all(all_idx, all_val, v_tmp1, v_one, v_two, v_n_nodes, v_zero, n_vectors)

    def _emit_round_k4(self, all_idx, all_val, v_node, v_tmp1, v_tmp2,
                       hash_const_vecs, v_one, v_two, v_n_nodes, v_zero,
                       n_vectors, tmp1, tmp2, tmp3, idx_base):
        """Emit round with k=4 unique indices using VALU-based conditional select."""
        # Load 4 node values: forest[idx_base] through forest[idx_base+3]
        for i in range(4):
            self.add("load", ("const", tmp1, idx_base + i))
            self.instrs.append({"alu": [("+", tmp1, self.scratch["forest_values_p"], tmp1)]})
            self.add("load", ("load", tmp1, tmp1))
            self.add("valu", ("vbroadcast", v_node[i], tmp1))

        # Create vector constants
        idx_base_vec = self.scratch_const_vec(idx_base)
        v_two_const = self.scratch_const_vec(2)

        for vi in range(n_vectors):
            # offset = idx - idx_base (0-3)
            self.instrs.append({"valu": [("-", v_tmp1[0], all_idx[vi], idx_base_vec)]})

            # Binary tree select using bit 1 (high bit) first, then bit 0
            # Level 1: if offset >= 2, use nodes 2,3; else use nodes 0,1
            self.instrs.append({"valu": [("&", v_tmp1[1], v_tmp1[0], v_two_const)]})  # offset & 2
            # For mask: need (offset & 2) != 0 → divide by 2 to get 0 or 1
            self.instrs.append({"valu": [(">>", v_tmp1[1], v_tmp1[1], v_one)]})  # 0 or 1
            self.instrs.append({"valu": [("-", v_tmp1[1], v_zero, v_tmp1[1])]})  # 0 or -1

            # diff_02 = node[0] ^ node[2], diff_13 = node[1] ^ node[3]
            self.instrs.append({
                "valu": [("^", v_tmp2[0], v_node[0], v_node[2]),
                         ("^", v_tmp2[1], v_node[1], v_node[3])]
            })
            # masked diffs
            self.instrs.append({
                "valu": [("&", v_tmp2[0], v_tmp2[0], v_tmp1[1]),
                         ("&", v_tmp2[1], v_tmp2[1], v_tmp1[1])]
            })
            # sel_low = node[0] ^ masked_02 (gives node[0] or node[2])
            # sel_high = node[1] ^ masked_13 (gives node[1] or node[3])
            self.instrs.append({
                "valu": [("^", v_tmp2[0], v_node[0], v_tmp2[0]),
                         ("^", v_tmp2[1], v_node[1], v_tmp2[1])]
            })

            # Level 2: if offset & 1, use sel_high; else use sel_low
            self.instrs.append({"valu": [("&", v_tmp1[1], v_tmp1[0], v_one)]})  # offset & 1
            self.instrs.append({"valu": [("-", v_tmp1[1], v_zero, v_tmp1[1])]})  # 0 or -1

            self.instrs.append({"valu": [("^", v_tmp2[2], v_tmp2[0], v_tmp2[1])]})  # diff
            self.instrs.append({"valu": [("&", v_tmp2[2], v_tmp2[2], v_tmp1[1])]})  # masked
            self.instrs.append({"valu": [("^", v_tmp1[2], v_tmp2[0], v_tmp2[2])]})  # final selected

            # XOR with val
            self.instrs.append({"valu": [("^", all_val[vi], all_val[vi], v_tmp1[2])]})

        # Hash all vectors
        self._emit_hash_all(all_val, v_tmp1, v_tmp2, hash_const_vecs, n_vectors)

        # Index update all vectors
        self._emit_index_all(all_idx, all_val, v_tmp1, v_one, v_two, v_n_nodes, v_zero, n_vectors)

    def _emit_round_k_unique(self, all_idx, all_val, v_node, v_tmp1, v_tmp2, v_addrs,
                             hash_const_vecs, v_one, v_two, v_n_nodes, v_zero,
                             n_vectors, tmp1, tmp2, tmp3, k, idx_base):
        """Emit round with k unique indices using VALU-based conditional select."""

        if k == 2:
            # Load 2 node values and broadcast
            self.add("load", ("const", tmp1, idx_base))
            self.add("load", ("const", tmp2, idx_base + 1))
            self.instrs.append({"alu": [("+", tmp1, self.scratch["forest_values_p"], tmp1),
                                        ("+", tmp2, self.scratch["forest_values_p"], tmp2)]})
            self.instrs.append({"load": [("load", tmp1, tmp1), ("load", tmp2, tmp2)]})
            self.instrs.append({"valu": [("vbroadcast", v_node[0], tmp1),
                                          ("vbroadcast", v_node[1], tmp2)]})

            # Create vector constant for idx_base
            idx_base_vec = self.scratch_const_vec(idx_base)

            # Precompute diff = node[0] ^ node[1] (used for all vectors)
            self.instrs.append({"valu": [("^", v_tmp2[3], v_node[0], v_node[1])]})

            # For each vector: VALU-based select
            # result = node[0] ^ ((node[0] ^ node[1]) & mask)
            # where mask = 0 - (idx - idx_base) = 0 if idx==base, -1 if idx==base+1
            for vi in range(0, n_vectors, 2):
                # offset = idx - idx_base (0 or 1)
                self.instrs.append({
                    "valu": [("-", v_tmp1[0], all_idx[vi], idx_base_vec),
                             ("-", v_tmp1[1], all_idx[vi+1], idx_base_vec)]
                })
                # mask = 0 - offset (0 or all-1s)
                self.instrs.append({
                    "valu": [("-", v_tmp1[0], v_zero, v_tmp1[0]),
                             ("-", v_tmp1[1], v_zero, v_tmp1[1])]
                })
                # masked_diff = diff & mask
                self.instrs.append({
                    "valu": [("&", v_tmp1[0], v_tmp2[3], v_tmp1[0]),
                             ("&", v_tmp1[1], v_tmp2[3], v_tmp1[1])]
                })
                # selected = node[0] ^ masked_diff, then XOR with val
                self.instrs.append({
                    "valu": [("^", v_tmp1[0], v_node[0], v_tmp1[0]),
                             ("^", v_tmp1[1], v_node[0], v_tmp1[1])]
                })
                self.instrs.append({
                    "valu": [("^", all_val[vi], all_val[vi], v_tmp1[0]),
                             ("^", all_val[vi+1], all_val[vi+1], v_tmp1[1])]
                })

        elif k == 4:
            # Load 4 node values
            for i in range(4):
                self.add("load", ("const", tmp1, idx_base + i))
                self.instrs.append({"alu": [("+", tmp1, self.scratch["forest_values_p"], tmp1)]})
                self.add("load", ("load", tmp1, tmp1))
                self.add("valu", ("vbroadcast", v_node[i], tmp1))

            idx_base_vec = self.scratch_const_vec(idx_base)
            v_two_const = self.scratch_const_vec(2)

            for vi in range(n_vectors):
                # offset = idx - idx_base (0-3)
                self.instrs.append({"valu": [("-", v_tmp1[0], all_idx[vi], idx_base_vec)]})

                # Level 1: select based on bit 1 (offset >= 2)
                # mask1 = 0 - ((offset & 2) >> 1) = 0 if offset<2, -1 if offset>=2
                self.instrs.append({"valu": [("&", v_tmp1[1], v_tmp1[0], v_two_const)]})  # offset & 2
                self.instrs.append({"valu": [(">>", v_tmp1[1], v_tmp1[1], v_one)]})  # (offset & 2) >> 1 -> 0 or 1
                self.instrs.append({"valu": [("-", v_tmp1[1], v_zero, v_tmp1[1])]})  # mask: 0 or -1

                # Select between (n0,n1) and (n2,n3) pairs
                self.instrs.append({
                    "valu": [("^", v_tmp2[0], v_node[0], v_node[2]),  # diff02
                             ("^", v_tmp2[1], v_node[1], v_node[3])]  # diff13
                })
                self.instrs.append({
                    "valu": [("&", v_tmp2[0], v_tmp2[0], v_tmp1[1]),  # masked diff02
                             ("&", v_tmp2[1], v_tmp2[1], v_tmp1[1])]  # masked diff13
                })
                self.instrs.append({
                    "valu": [("^", v_tmp2[0], v_node[0], v_tmp2[0]),  # sel_low = n0 or n2
                             ("^", v_tmp2[1], v_node[1], v_tmp2[1])]  # sel_high = n1 or n3
                })

                # Level 2: select based on bit 0 (offset & 1)
                self.instrs.append({"valu": [("&", v_tmp1[1], v_tmp1[0], v_one)]})  # offset & 1
                self.instrs.append({"valu": [("-", v_tmp1[1], v_zero, v_tmp1[1])]})  # mask: 0 or -1

                self.instrs.append({"valu": [("^", v_tmp2[2], v_tmp2[0], v_tmp2[1])]})  # diff
                self.instrs.append({"valu": [("&", v_tmp2[2], v_tmp2[2], v_tmp1[1])]})  # masked
                self.instrs.append({"valu": [("^", v_tmp1[2], v_tmp2[0], v_tmp2[2])]})  # final selected

                # XOR with val
                self.instrs.append({"valu": [("^", all_val[vi], all_val[vi], v_tmp1[2])]})

        else:
            # k >= 8: use full gather (simpler, similar perf at this point)
            self._emit_round_full_gather(
                all_idx, all_val, v_node, v_tmp1, v_tmp2, v_addrs,
                hash_const_vecs, v_one, v_two, v_n_nodes, v_zero,
                n_vectors, tmp1
            )
            return

        # Hash and index update
        self._emit_hash_all(all_val, v_tmp1, v_tmp2, hash_const_vecs, n_vectors)
        self._emit_index_all(all_idx, all_val, v_tmp1, v_one, v_two, v_n_nodes, v_zero, n_vectors)

    def _emit_hash_all(self, all_val, v_tmp1, v_tmp2, hash_const_vecs, n_vectors,
                        extra_alu_ops=None, extra_load_ops=None):
        """Hash all vectors using multiply_add optimization for stages 0, 2, 4.

        Stages 0, 2, 4 use multiply_add: dest = a * mult + const (1 VALU op)
        Stages 1, 3, 5 use regular: op1, op3, then op2 (3 VALU ops, can pipeline)

        This reduces hash from 18 ops to 12 ops per vector (33% reduction).
        """
        # Process 6 vectors at a time for maximum VALU slot utilization
        for vi in range(0, n_vectors, 6):
            vecs = min(6, n_vectors - vi)

            for hi, stage_info in enumerate(hash_const_vecs):
                op1, val1, op2, op3, val3 = HASH_STAGES[hi]

                if stage_info[0] == 'multiply_add':
                    # Stages 0, 2, 4: use multiply_add (1 op per vector)
                    _, mult_vec, const_vec = stage_info
                    if vecs >= 6:
                        # Process all 6 vectors in one instruction
                        self.instrs.append({
                            "valu": [("multiply_add", all_val[vi+j], all_val[vi+j], mult_vec, const_vec)
                                     for j in range(6)]
                        })
                    elif vecs >= 3:
                        # Process 3 vectors
                        self.instrs.append({
                            "valu": [("multiply_add", all_val[vi+j], all_val[vi+j], mult_vec, const_vec)
                                     for j in range(vecs)]
                        })
                    else:
                        # Process 1-2 vectors
                        self.instrs.append({
                            "valu": [("multiply_add", all_val[vi+j], all_val[vi+j], mult_vec, const_vec)
                                     for j in range(vecs)]
                        })
                else:
                    # Stages 1, 3, 5: regular 3-op pattern with pipelining
                    _, const1, const3 = stage_info

                    if vecs >= 6:
                        # Full 6-vector batch with pipelining
                        # Instr 1: compute op1+op3 for v0-2 (6 slots)
                        self.instrs.append({
                            "valu": [(op1, v_tmp1[0], all_val[vi], const1),
                                     (op3, v_tmp2[0], all_val[vi], const3),
                                     (op1, v_tmp1[1], all_val[vi+1], const1),
                                     (op3, v_tmp2[1], all_val[vi+1], const3),
                                     (op1, v_tmp1[2], all_val[vi+2], const1),
                                     (op3, v_tmp2[2], all_val[vi+2], const3)]
                        })

                        # Instr 2: combine v0-2 + start compute v3-5
                        self.instrs.append({
                            "valu": [(op2, all_val[vi], v_tmp1[0], v_tmp2[0]),
                                     (op2, all_val[vi+1], v_tmp1[1], v_tmp2[1]),
                                     (op2, all_val[vi+2], v_tmp1[2], v_tmp2[2]),
                                     (op1, v_tmp1[0], all_val[vi+3], const1),
                                     (op3, v_tmp2[0], all_val[vi+3], const3),
                                     (op1, v_tmp1[1], all_val[vi+4], const1)]
                        })

                        # Instr 3: finish compute v4-5 + combine v3
                        self.instrs.append({
                            "valu": [(op3, v_tmp2[1], all_val[vi+4], const3),
                                     (op1, v_tmp1[2], all_val[vi+5], const1),
                                     (op3, v_tmp2[2], all_val[vi+5], const3),
                                     (op2, all_val[vi+3], v_tmp1[0], v_tmp2[0])]
                        })

                        # Instr 4: combine v4-5
                        self.instrs.append({
                            "valu": [(op2, all_val[vi+4], v_tmp1[1], v_tmp2[1]),
                                     (op2, all_val[vi+5], v_tmp1[2], v_tmp2[2])]
                        })

                    elif vecs >= 3:
                        # 3-5 vectors
                        for v_off in range(0, vecs, 3):
                            v_count = min(3, vecs - v_off)
                            base = vi + v_off

                            if v_count == 3:
                                self.instrs.append({
                                    "valu": [(op1, v_tmp1[0], all_val[base], const1),
                                             (op3, v_tmp2[0], all_val[base], const3),
                                             (op1, v_tmp1[1], all_val[base+1], const1),
                                             (op3, v_tmp2[1], all_val[base+1], const3),
                                             (op1, v_tmp1[2], all_val[base+2], const1),
                                             (op3, v_tmp2[2], all_val[base+2], const3)]
                                })
                                self.instrs.append({
                                    "valu": [(op2, all_val[base], v_tmp1[0], v_tmp2[0]),
                                             (op2, all_val[base+1], v_tmp1[1], v_tmp2[1]),
                                             (op2, all_val[base+2], v_tmp1[2], v_tmp2[2])]
                                })
                            else:  # v_count == 2
                                self.instrs.append({
                                    "valu": [(op1, v_tmp1[0], all_val[base], const1),
                                             (op3, v_tmp2[0], all_val[base], const3),
                                             (op1, v_tmp1[1], all_val[base+1], const1),
                                             (op3, v_tmp2[1], all_val[base+1], const3)]
                                })
                                self.instrs.append({
                                    "valu": [(op2, all_val[base], v_tmp1[0], v_tmp2[0]),
                                             (op2, all_val[base+1], v_tmp1[1], v_tmp2[1])]
                                })
                    else:
                        # 1-2 vectors
                        if vecs == 2:
                            self.instrs.append({
                                "valu": [(op1, v_tmp1[0], all_val[vi], const1),
                                         (op3, v_tmp2[0], all_val[vi], const3),
                                         (op1, v_tmp1[1], all_val[vi+1], const1),
                                         (op3, v_tmp2[1], all_val[vi+1], const3)]
                            })
                            self.instrs.append({
                                "valu": [(op2, all_val[vi], v_tmp1[0], v_tmp2[0]),
                                         (op2, all_val[vi+1], v_tmp1[1], v_tmp2[1])]
                            })
                        else:
                            self.instrs.append({
                                "valu": [(op1, v_tmp1[0], all_val[vi], const1),
                                         (op3, v_tmp2[0], all_val[vi], const3)]
                            })
                            self.instrs.append({
                                "valu": [(op2, all_val[vi], v_tmp1[0], v_tmp2[0])]
                            })

    def _emit_index_all(self, all_idx, all_val, v_tmp1, v_one, v_two, v_n_nodes, v_zero, n_vectors,
                         skip_wrap_check=False):
        """Index update all vectors, 3 at a time for better VALU utilization.

        skip_wrap_check: If True, skip the comparison and masking (saves 3 instructions per group).
                        Use this for rounds where idx is guaranteed to stay < n_nodes.
        """
        for vi in range(0, n_vectors, 3):
            vecs = min(3, n_vectors - vi)
            # new_idx = old_idx * 2 + (val & 1) + 1
            # if new_idx >= n_nodes: new_idx = 0
            if vecs == 3:
                self.instrs.append({
                    "valu": [("&", v_tmp1[0], all_val[vi], v_one),
                             ("*", all_idx[vi], all_idx[vi], v_two),
                             ("&", v_tmp1[1], all_val[vi+1], v_one),
                             ("*", all_idx[vi+1], all_idx[vi+1], v_two),
                             ("&", v_tmp1[2], all_val[vi+2], v_one),
                             ("*", all_idx[vi+2], all_idx[vi+2], v_two)]
                })
                self.instrs.append({
                    "valu": [("+", v_tmp1[0], v_tmp1[0], v_one),
                             ("+", v_tmp1[1], v_tmp1[1], v_one),
                             ("+", v_tmp1[2], v_tmp1[2], v_one)]
                })
                self.instrs.append({
                    "valu": [("+", all_idx[vi], all_idx[vi], v_tmp1[0]),
                             ("+", all_idx[vi+1], all_idx[vi+1], v_tmp1[1]),
                             ("+", all_idx[vi+2], all_idx[vi+2], v_tmp1[2])]
                })
                if not skip_wrap_check:
                    self.instrs.append({
                        "valu": [("<", v_tmp1[0], all_idx[vi], v_n_nodes),
                                 ("<", v_tmp1[1], all_idx[vi+1], v_n_nodes),
                                 ("<", v_tmp1[2], all_idx[vi+2], v_n_nodes)]
                    })
                    self.instrs.append({
                        "valu": [("-", v_tmp1[0], v_zero, v_tmp1[0]),
                                 ("-", v_tmp1[1], v_zero, v_tmp1[1]),
                                 ("-", v_tmp1[2], v_zero, v_tmp1[2])]
                    })
                    self.instrs.append({
                        "valu": [("&", all_idx[vi], all_idx[vi], v_tmp1[0]),
                                 ("&", all_idx[vi+1], all_idx[vi+1], v_tmp1[1]),
                                 ("&", all_idx[vi+2], all_idx[vi+2], v_tmp1[2])]
                    })
            elif vecs == 2:
                self.instrs.append({
                    "valu": [("&", v_tmp1[0], all_val[vi], v_one),
                             ("*", all_idx[vi], all_idx[vi], v_two),
                             ("&", v_tmp1[1], all_val[vi+1], v_one),
                             ("*", all_idx[vi+1], all_idx[vi+1], v_two)]
                })
                self.instrs.append({
                    "valu": [("+", v_tmp1[0], v_tmp1[0], v_one),
                             ("+", v_tmp1[1], v_tmp1[1], v_one)]
                })
                self.instrs.append({
                    "valu": [("+", all_idx[vi], all_idx[vi], v_tmp1[0]),
                             ("+", all_idx[vi+1], all_idx[vi+1], v_tmp1[1])]
                })
                if not skip_wrap_check:
                    self.instrs.append({
                        "valu": [("<", v_tmp1[0], all_idx[vi], v_n_nodes),
                                 ("<", v_tmp1[1], all_idx[vi+1], v_n_nodes)]
                    })
                    self.instrs.append({
                        "valu": [("-", v_tmp1[0], v_zero, v_tmp1[0]),
                                 ("-", v_tmp1[1], v_zero, v_tmp1[1])]
                    })
                    self.instrs.append({
                        "valu": [("&", all_idx[vi], all_idx[vi], v_tmp1[0]),
                                 ("&", all_idx[vi+1], all_idx[vi+1], v_tmp1[1])]
                    })
            else:
                self.instrs.append({
                    "valu": [("&", v_tmp1[0], all_val[vi], v_one),
                             ("*", all_idx[vi], all_idx[vi], v_two)]
                })
                self.instrs.append({
                    "valu": [("+", v_tmp1[0], v_tmp1[0], v_one)]
                })
                self.instrs.append({
                    "valu": [("+", all_idx[vi], all_idx[vi], v_tmp1[0])]
                })
                if not skip_wrap_check:
                    self.instrs.append({
                        "valu": [("<", v_tmp1[0], all_idx[vi], v_n_nodes)]
                    })
                    self.instrs.append({
                        "valu": [("-", v_tmp1[0], v_zero, v_tmp1[0])]
                    })
                    self.instrs.append({
                        "valu": [("&", all_idx[vi], all_idx[vi], v_tmp1[0])]
                    })

    def _emit_round_full_gather(self, all_idx, all_val, v_node, v_tmp1, v_tmp2, v_addrs, v_addrs2,
                                 hash_const_vecs, v_one, v_two, v_n_nodes, v_zero,
                                 n_vectors, tmp1, skip_index=False, skip_wrap_check=False):
        """Full gather round with pipelining - overlaps Index B with next group's address computation."""
        n_groups = n_vectors // 4  # 8 groups
        fp = self.scratch["forest_values_p"]

        # Swap between address buffers: even groups use v_addrs, odd groups use v_addrs2
        addr_bufs = [v_addrs, v_addrs2]

        # Track pending Index B from previous group
        pending_index_b = False
        prev_g = 0

        for group_idx in range(n_groups):
            g = group_idx * 4
            cur_addrs = addr_bufs[group_idx % 2]
            next_addrs = addr_bufs[(group_idx + 1) % 2]
            has_next_group = (group_idx < n_groups - 1)
            ng = g + 4  # next group start

            # Compute addresses for current group (only for group 0)
            if group_idx == 0:
                # First group: compute addresses normally
                self.instrs.append({
                    "alu": [("+", cur_addrs[0] + vi, fp, all_idx[g] + vi) for vi in range(8)] +
                           [("+", cur_addrs[1] + vi, fp, all_idx[g + 1] + vi) for vi in range(4)]
                })
                self.instrs.append({
                    "alu": [("+", cur_addrs[1] + 4 + vi, fp, all_idx[g + 1] + 4 + vi) for vi in range(4)] +
                           [("+", cur_addrs[2] + vi, fp, all_idx[g + 2] + vi) for vi in range(8)]
                })
                self.instrs.append({
                    "alu": [("+", cur_addrs[3] + vi, fp, all_idx[g + 3] + vi) for vi in range(8)]
                })

            # Gather A (vec 0,1) - overlap with prev Index B if pending
            if pending_index_b and not skip_index:
                # Overlap Gather A with prev group's Index B
                pg = prev_g  # previous group start
                # 8 loads, 6 Index B instructions - overlap first 6 loads with Index B
                self.instrs.append({
                    "load": [("load", v_node[0] + 0, cur_addrs[0] + 0),
                             ("load", v_node[1] + 0, cur_addrs[1] + 0)],
                    "valu": [("&", v_tmp1[2], all_val[pg + 2], v_one),
                             ("*", all_idx[pg + 2], all_idx[pg + 2], v_two),
                             ("&", v_tmp1[3], all_val[pg + 3], v_one),
                             ("*", all_idx[pg + 3], all_idx[pg + 3], v_two)]
                })
                self.instrs.append({
                    "load": [("load", v_node[0] + 1, cur_addrs[0] + 1),
                             ("load", v_node[1] + 1, cur_addrs[1] + 1)],
                    "valu": [("+", v_tmp1[2], v_tmp1[2], v_one),
                             ("+", v_tmp1[3], v_tmp1[3], v_one)]
                })
                self.instrs.append({
                    "load": [("load", v_node[0] + 2, cur_addrs[0] + 2),
                             ("load", v_node[1] + 2, cur_addrs[1] + 2)],
                    "valu": [("+", all_idx[pg + 2], all_idx[pg + 2], v_tmp1[2]),
                             ("+", all_idx[pg + 3], all_idx[pg + 3], v_tmp1[3])]
                })
                if skip_wrap_check:
                    # Skip comparison/masking, just do loads
                    for vi in range(3, 8):
                        self.instrs.append({
                            "load": [("load", v_node[0] + vi, cur_addrs[0] + vi),
                                     ("load", v_node[1] + vi, cur_addrs[1] + vi)]
                        })
                else:
                    self.instrs.append({
                        "load": [("load", v_node[0] + 3, cur_addrs[0] + 3),
                                 ("load", v_node[1] + 3, cur_addrs[1] + 3)],
                        "valu": [("<", v_tmp1[2], all_idx[pg + 2], v_n_nodes),
                                 ("<", v_tmp1[3], all_idx[pg + 3], v_n_nodes)]
                    })
                    self.instrs.append({
                        "load": [("load", v_node[0] + 4, cur_addrs[0] + 4),
                                 ("load", v_node[1] + 4, cur_addrs[1] + 4)],
                        "valu": [("-", v_tmp1[2], v_zero, v_tmp1[2]),
                                 ("-", v_tmp1[3], v_zero, v_tmp1[3])]
                    })
                    self.instrs.append({
                        "load": [("load", v_node[0] + 5, cur_addrs[0] + 5),
                                 ("load", v_node[1] + 5, cur_addrs[1] + 5)],
                        "valu": [("&", all_idx[pg + 2], all_idx[pg + 2], v_tmp1[2]),
                                 ("&", all_idx[pg + 3], all_idx[pg + 3], v_tmp1[3])]
                    })
                    # Remaining loads (no more Index B ops)
                    self.instrs.append({
                        "load": [("load", v_node[0] + 6, cur_addrs[0] + 6),
                                 ("load", v_node[1] + 6, cur_addrs[1] + 6)]
                    })
                    self.instrs.append({
                        "load": [("load", v_node[0] + 7, cur_addrs[0] + 7),
                                 ("load", v_node[1] + 7, cur_addrs[1] + 7)]
                    })
                pending_index_b = False
            else:
                # Normal Gather A (no overlap)
                for vi in range(VLEN):
                    self.instrs.append({
                        "load": [("load", v_node[0] + vi, cur_addrs[0] + vi),
                                 ("load", v_node[1] + vi, cur_addrs[1] + vi)]
                    })

            # XOR A
            # XOR A + start Gather B (overlap to save 1 instruction)
            self.instrs.append({
                "valu": [("^", all_val[g], all_val[g], v_node[0]),
                         ("^", all_val[g + 1], all_val[g + 1], v_node[1])],
                "load": [("load", v_node[2] + 0, cur_addrs[2] + 0),
                         ("load", v_node[3] + 0, cur_addrs[3] + 0)]
            })

            # Hash A + Gather B overlapped (start from load_idx=1)
            load_idx = 1
            for hi, stage_info in enumerate(hash_const_vecs):
                op1, val1, op2, op3, val3 = HASH_STAGES[hi]

                if stage_info[0] == 'multiply_add':
                    # multiply_add stage: 1 instruction with 2 multiply_add ops
                    _, mult_vec, const_vec = stage_info
                    vi = load_idx
                    if vi < VLEN:
                        self.instrs.append({
                            "valu": [("multiply_add", all_val[g], all_val[g], mult_vec, const_vec),
                                     ("multiply_add", all_val[g + 1], all_val[g + 1], mult_vec, const_vec)],
                            "load": [("load", v_node[2] + vi, cur_addrs[2] + vi),
                                     ("load", v_node[3] + vi, cur_addrs[3] + vi)]
                        })
                        load_idx += 1
                    else:
                        self.instrs.append({
                            "valu": [("multiply_add", all_val[g], all_val[g], mult_vec, const_vec),
                                     ("multiply_add", all_val[g + 1], all_val[g + 1], mult_vec, const_vec)]
                        })
                else:
                    # Regular stage: 2 instructions (compute, then combine)
                    _, const1_vec, const3_vec = stage_info

                    vi = load_idx
                    if vi < VLEN:
                        self.instrs.append({
                            "valu": [(op1, v_tmp1[0], all_val[g], const1_vec),
                                     (op3, v_tmp2[0], all_val[g], const3_vec),
                                     (op1, v_tmp1[1], all_val[g + 1], const1_vec),
                                     (op3, v_tmp2[1], all_val[g + 1], const3_vec)],
                            "load": [("load", v_node[2] + vi, cur_addrs[2] + vi),
                                     ("load", v_node[3] + vi, cur_addrs[3] + vi)]
                        })
                        load_idx += 1
                    else:
                        self.instrs.append({
                            "valu": [(op1, v_tmp1[0], all_val[g], const1_vec),
                                     (op3, v_tmp2[0], all_val[g], const3_vec),
                                     (op1, v_tmp1[1], all_val[g + 1], const1_vec),
                                     (op3, v_tmp2[1], all_val[g + 1], const3_vec)]
                        })

                    vi = load_idx
                    if vi < VLEN:
                        self.instrs.append({
                            "valu": [(op2, all_val[g], v_tmp1[0], v_tmp2[0]),
                                     (op2, all_val[g + 1], v_tmp1[1], v_tmp2[1])],
                            "load": [("load", v_node[2] + vi, cur_addrs[2] + vi),
                                     ("load", v_node[3] + vi, cur_addrs[3] + vi)]
                        })
                        load_idx += 1
                    else:
                        self.instrs.append({
                            "valu": [(op2, all_val[g], v_tmp1[0], v_tmp2[0]),
                                     (op2, all_val[g + 1], v_tmp1[1], v_tmp2[1])]
                        })

            while load_idx < VLEN:
                self.instrs.append({
                    "load": [("load", v_node[2] + load_idx, cur_addrs[2] + load_idx),
                             ("load", v_node[3] + load_idx, cur_addrs[3] + load_idx)]
                })
                load_idx += 1

            # XOR B
            self.instrs.append({
                "valu": [("^", all_val[g + 2], all_val[g + 2], v_node[2]),
                         ("^", all_val[g + 3], all_val[g + 3], v_node[3])]
            })

            # Hash B + Index A (skip Index A when skip_index is True)
            # Note: stages 0, 2, 4 are multiply_add (1 instr), stages 1, 3, 5 are regular (2 instr)
            # Index A needs to be interleaved across all stages
            for hi, stage_info in enumerate(hash_const_vecs):
                op1, val1, op2, op3, val3 = HASH_STAGES[hi]
                is_multiply_add = (stage_info[0] == 'multiply_add')

                if is_multiply_add:
                    _, mult_vec, const_vec = stage_info
                else:
                    _, const1_vec, const3_vec = stage_info

                if skip_index:
                    # Skip index operations on last round, but still compute addresses for next group
                    if is_multiply_add:
                        if hi == 4 and has_next_group:
                            self.instrs.append({
                                "valu": [("multiply_add", all_val[g + 2], all_val[g + 2], mult_vec, const_vec),
                                         ("multiply_add", all_val[g + 3], all_val[g + 3], mult_vec, const_vec)],
                                "alu": [("+", next_addrs[0] + vi, fp, all_idx[ng] + vi) for vi in range(8)] +
                                       [("+", next_addrs[1] + vi, fp, all_idx[ng + 1] + vi) for vi in range(4)]
                            })
                        else:
                            self.instrs.append({
                                "valu": [("multiply_add", all_val[g + 2], all_val[g + 2], mult_vec, const_vec),
                                         ("multiply_add", all_val[g + 3], all_val[g + 3], mult_vec, const_vec)]
                            })
                    else:
                        if hi == 5 and has_next_group:
                            self.instrs.append({
                                "valu": [(op1, v_tmp1[2], all_val[g + 2], const1_vec),
                                         (op3, v_tmp2[2], all_val[g + 2], const3_vec),
                                         (op1, v_tmp1[3], all_val[g + 3], const1_vec),
                                         (op3, v_tmp2[3], all_val[g + 3], const3_vec)],
                                "alu": [("+", next_addrs[1] + 4 + vi, fp, all_idx[ng + 1] + 4 + vi) for vi in range(4)] +
                                       [("+", next_addrs[2] + vi, fp, all_idx[ng + 2] + vi) for vi in range(8)]
                            })
                            self.instrs.append({
                                "valu": [(op2, all_val[g + 2], v_tmp1[2], v_tmp2[2]),
                                         (op2, all_val[g + 3], v_tmp1[3], v_tmp2[3])],
                                "alu": [("+", next_addrs[3] + vi, fp, all_idx[ng + 3] + vi) for vi in range(8)]
                            })
                        else:
                            self.instrs.append({
                                "valu": [(op1, v_tmp1[2], all_val[g + 2], const1_vec),
                                         (op3, v_tmp2[2], all_val[g + 2], const3_vec),
                                         (op1, v_tmp1[3], all_val[g + 3], const1_vec),
                                         (op3, v_tmp2[3], all_val[g + 3], const3_vec)]
                            })
                            self.instrs.append({
                                "valu": [(op2, all_val[g + 2], v_tmp1[2], v_tmp2[2]),
                                         (op2, all_val[g + 3], v_tmp1[3], v_tmp2[3])]
                            })
                elif hi == 0:
                    # Stage 0 is multiply_add - interleave with Index A start
                    self.instrs.append({
                        "valu": [("multiply_add", all_val[g + 2], all_val[g + 2], mult_vec, const_vec),
                                 ("multiply_add", all_val[g + 3], all_val[g + 3], mult_vec, const_vec),
                                 ("&", v_node[0], all_val[g], v_one),
                                 ("*", all_idx[g], all_idx[g], v_two),
                                 ("&", v_node[1], all_val[g + 1], v_one),
                                 ("*", all_idx[g + 1], all_idx[g + 1], v_two)]
                    })
                elif hi == 1:
                    # Stage 1 is regular - interleave with Index A continue
                    self.instrs.append({
                        "valu": [(op1, v_tmp1[2], all_val[g + 2], const1_vec),
                                 (op3, v_tmp2[2], all_val[g + 2], const3_vec),
                                 (op1, v_tmp1[3], all_val[g + 3], const1_vec),
                                 (op3, v_tmp2[3], all_val[g + 3], const3_vec),
                                 ("+", v_node[0], v_node[0], v_one),
                                 ("+", v_node[1], v_node[1], v_one)]
                    })
                    self.instrs.append({
                        "valu": [(op2, all_val[g + 2], v_tmp1[2], v_tmp2[2]),
                                 (op2, all_val[g + 3], v_tmp1[3], v_tmp2[3]),
                                 ("+", all_idx[g], all_idx[g], v_node[0]),
                                 ("+", all_idx[g + 1], all_idx[g + 1], v_node[1])]
                    })
                elif hi == 2:
                    # Stage 2 is multiply_add - interleave with Index A compare (if wrap check)
                    if skip_wrap_check:
                        self.instrs.append({
                            "valu": [("multiply_add", all_val[g + 2], all_val[g + 2], mult_vec, const_vec),
                                     ("multiply_add", all_val[g + 3], all_val[g + 3], mult_vec, const_vec)]
                        })
                    else:
                        self.instrs.append({
                            "valu": [("multiply_add", all_val[g + 2], all_val[g + 2], mult_vec, const_vec),
                                     ("multiply_add", all_val[g + 3], all_val[g + 3], mult_vec, const_vec),
                                     ("<", v_node[0], all_idx[g], v_n_nodes),
                                     ("<", v_node[1], all_idx[g + 1], v_n_nodes)]
                        })
                elif hi == 3:
                    # Stage 3 is regular - interleave with Index A mask (if wrap check)
                    if skip_wrap_check:
                        self.instrs.append({
                            "valu": [(op1, v_tmp1[2], all_val[g + 2], const1_vec),
                                     (op3, v_tmp2[2], all_val[g + 2], const3_vec),
                                     (op1, v_tmp1[3], all_val[g + 3], const1_vec),
                                     (op3, v_tmp2[3], all_val[g + 3], const3_vec)]
                        })
                        self.instrs.append({
                            "valu": [(op2, all_val[g + 2], v_tmp1[2], v_tmp2[2]),
                                     (op2, all_val[g + 3], v_tmp1[3], v_tmp2[3])]
                        })
                    else:
                        self.instrs.append({
                            "valu": [(op1, v_tmp1[2], all_val[g + 2], const1_vec),
                                     (op3, v_tmp2[2], all_val[g + 2], const3_vec),
                                     (op1, v_tmp1[3], all_val[g + 3], const1_vec),
                                     (op3, v_tmp2[3], all_val[g + 3], const3_vec),
                                     ("-", v_node[0], v_zero, v_node[0]),
                                     ("-", v_node[1], v_zero, v_node[1])]
                        })
                        self.instrs.append({
                            "valu": [(op2, all_val[g + 2], v_tmp1[2], v_tmp2[2]),
                                     (op2, all_val[g + 3], v_tmp1[3], v_tmp2[3]),
                                     ("&", all_idx[g], all_idx[g], v_node[0]),
                                     ("&", all_idx[g + 1], all_idx[g + 1], v_node[1])]
                        })
                elif hi == 4:
                    # Stage 4 is multiply_add - add address computation for next group
                    if has_next_group:
                        self.instrs.append({
                            "valu": [("multiply_add", all_val[g + 2], all_val[g + 2], mult_vec, const_vec),
                                     ("multiply_add", all_val[g + 3], all_val[g + 3], mult_vec, const_vec)],
                            "alu": [("+", next_addrs[0] + vi, fp, all_idx[ng] + vi) for vi in range(8)] +
                                   [("+", next_addrs[1] + vi, fp, all_idx[ng + 1] + vi) for vi in range(4)]
                        })
                    else:
                        self.instrs.append({
                            "valu": [("multiply_add", all_val[g + 2], all_val[g + 2], mult_vec, const_vec),
                                     ("multiply_add", all_val[g + 3], all_val[g + 3], mult_vec, const_vec)]
                        })
                elif hi == 5:
                    # Stage 5 is regular - add address computation for next group
                    if has_next_group:
                        self.instrs.append({
                            "valu": [(op1, v_tmp1[2], all_val[g + 2], const1_vec),
                                     (op3, v_tmp2[2], all_val[g + 2], const3_vec),
                                     (op1, v_tmp1[3], all_val[g + 3], const1_vec),
                                     (op3, v_tmp2[3], all_val[g + 3], const3_vec)],
                            "alu": [("+", next_addrs[1] + 4 + vi, fp, all_idx[ng + 1] + 4 + vi) for vi in range(4)] +
                                   [("+", next_addrs[2] + vi, fp, all_idx[ng + 2] + vi) for vi in range(8)]
                        })
                        self.instrs.append({
                            "valu": [(op2, all_val[g + 2], v_tmp1[2], v_tmp2[2]),
                                     (op2, all_val[g + 3], v_tmp1[3], v_tmp2[3])],
                            "alu": [("+", next_addrs[3] + vi, fp, all_idx[ng + 3] + vi) for vi in range(8)]
                        })
                    else:
                        self.instrs.append({
                            "valu": [(op1, v_tmp1[2], all_val[g + 2], const1_vec),
                                     (op3, v_tmp2[2], all_val[g + 2], const3_vec),
                                     (op1, v_tmp1[3], all_val[g + 3], const1_vec),
                                     (op3, v_tmp2[3], all_val[g + 3], const3_vec)]
                        })
                        self.instrs.append({
                            "valu": [(op2, all_val[g + 2], v_tmp1[2], v_tmp2[2]),
                                     (op2, all_val[g + 3], v_tmp1[3], v_tmp2[3])]
                        })

            # Index B - deferred to overlap with next group's Gather A (skip on last round)
            if skip_index:
                # Skip Index B on last round
                pass
            elif has_next_group:
                # Defer Index B to overlap with next group's Gather A
                pending_index_b = True
                prev_g = g
            else:
                # Last group: execute Index B now (no next group to overlap with)
                self.instrs.append({
                    "valu": [("&", v_tmp1[2], all_val[g + 2], v_one),
                             ("*", all_idx[g + 2], all_idx[g + 2], v_two),
                             ("&", v_tmp1[3], all_val[g + 3], v_one),
                             ("*", all_idx[g + 3], all_idx[g + 3], v_two)]
                })
                self.instrs.append({
                    "valu": [("+", v_tmp1[2], v_tmp1[2], v_one),
                             ("+", v_tmp1[3], v_tmp1[3], v_one)]
                })
                self.instrs.append({
                    "valu": [("+", all_idx[g + 2], all_idx[g + 2], v_tmp1[2]),
                             ("+", all_idx[g + 3], all_idx[g + 3], v_tmp1[3])]
                })
                if not skip_wrap_check:
                    self.instrs.append({
                        "valu": [("<", v_tmp1[2], all_idx[g + 2], v_n_nodes),
                                 ("<", v_tmp1[3], all_idx[g + 3], v_n_nodes)]
                    })
                    self.instrs.append({
                        "valu": [("-", v_tmp1[2], v_zero, v_tmp1[2]),
                                 ("-", v_tmp1[3], v_zero, v_tmp1[3])]
                    })
                    self.instrs.append({
                        "valu": [("&", all_idx[g + 2], all_idx[g + 2], v_tmp1[2]),
                                 ("&", all_idx[g + 3], all_idx[g + 3], v_tmp1[3])]
                    })


BASELINE = 147734

def do_kernel_test(
    forest_height: int, rounds: int, batch_size: int,
    seed: int = 123, trace: bool = False, prints: bool = False,
):
    print(f"{forest_height=}, {rounds=}, {batch_size=}")
    random.seed(seed)
    forest = Tree.generate(forest_height)
    inp = Input.generate(forest, batch_size, rounds)
    mem = build_mem_image(forest, inp)

    kb = KernelBuilder()
    kb.build_kernel(forest.height, len(forest.values), len(inp.indices), rounds)

    machine = Machine(mem, kb.instrs, kb.debug_info(), n_cores=N_CORES, trace=trace)
    machine.enable_pause = False
    machine.enable_debug = False
    machine.run()
    print(f"PC: {machine.cores[0].pc}, State: {machine.cores[0].state}")

    for ref_mem in reference_kernel2(mem):
        pass

    inp_values_p = ref_mem[6]
    if prints:
        print("Machine:", machine.mem[inp_values_p : inp_values_p + len(inp.values)][:20])
        print("Ref:", ref_mem[inp_values_p : inp_values_p + len(inp.values)][:20])

    match = (machine.mem[inp_values_p : inp_values_p + len(inp.values)]
             == ref_mem[inp_values_p : inp_values_p + len(inp.values)])

    print("CYCLES: ", machine.cycle)
    print("Speedup over baseline: ", BASELINE / machine.cycle)
    print("CORRECT:", match)
    return machine.cycle


if __name__ == "__main__":
    kb = KernelBuilder()
    kb.build_kernel(10, 1023, 256, 16)
    print(f"Total instructions: {len(kb.instrs)}")
    print(f"Scratch used: {kb.scratch_ptr}")
    do_kernel_test(10, 16, 256, prints=True)
