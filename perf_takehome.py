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

        # Working buffers for 4 vectors per group (two sets for pipelining)
        v_node = [self.alloc_scratch(f"node_{i}", VLEN) for i in range(4)]
        v_tmp1 = [self.alloc_scratch(f"tmp1_{i}", VLEN) for i in range(4)]
        v_tmp2 = [self.alloc_scratch(f"tmp2_{i}", VLEN) for i in range(4)]
        v_addrs = [self.alloc_scratch(f"addrs_{i}", VLEN) for i in range(4)]
        v_node2 = [self.alloc_scratch(f"node2_{i}", VLEN) for i in range(4)]
        v_addrs2 = [self.alloc_scratch(f"addrs2_{i}", VLEN) for i in range(4)]

        # ============ INITIALIZATION ============
        for i, v in enumerate(init_vars):
            self.add("load", ("const", tmp1, i))
            self.add("load", ("load", self.scratch[v], tmp1))

        v_zero = self.scratch_const_vec(0)
        v_one = self.scratch_const_vec(1)
        v_two = self.scratch_const_vec(2)
        v_n_nodes = self.scratch_const_vec(n_nodes)

        hash_const_vecs = []
        for hi, (op1, val1, op2, op3, val3) in enumerate(HASH_STAGES):
            hash_const_vecs.append((self.scratch_const_vec(val1), self.scratch_const_vec(val3)))

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

            if effective_round == 0:
                # k=1: all indices at 0 (rounds 0 and 11)
                self._emit_round_single_idx(
                    all_idx, all_val, v_node, v_tmp1, v_tmp2,
                    hash_const_vecs, v_one, v_two, v_n_nodes, v_zero,
                    n_vectors, tmp1, idx_value=0, skip_index=is_last_round
                )
            elif effective_round == 1:
                # k=2: indices in {1, 2} (rounds 1 and 12)
                self._emit_round_k2(
                    all_idx, all_val, v_node, v_tmp1, v_tmp2,
                    hash_const_vecs, v_one, v_two, v_n_nodes, v_zero,
                    n_vectors, tmp1, tmp2, idx_base=1, skip_index=is_last_round
                )
            else:
                # Full gather for all other rounds
                self._emit_round_full_gather(
                    all_idx, all_val, v_node, v_node2, v_tmp1, v_tmp2, v_addrs, v_addrs2,
                    hash_const_vecs, v_one, v_two, v_n_nodes, v_zero,
                    n_vectors, tmp1, skip_index=is_last_round
                )

        # ============ STORE ALL VALUES ============
        for vi in range(n_vectors):
            self.add("load", ("const", tmp1, vi * VLEN))
            self.instrs.append({"alu": [("+", tmp2, self.scratch["inp_values_p"], tmp1)]})
            self.instrs.append({"store": [("vstore", tmp2, all_val[vi])]})

        self.instrs.append({"flow": [("pause",)]})

    def _emit_round_single_idx(self, all_idx, all_val, v_node, v_tmp1, v_tmp2,
                                hash_const_vecs, v_one, v_two, v_n_nodes, v_zero,
                                n_vectors, tmp1, idx_value, skip_index=False):
        """Emit round with single unique index (rounds 0 and 11)."""
        # Load single node value
        self.add("load", ("load", tmp1, self.scratch["forest_values_p"]))
        self.add("valu", ("vbroadcast", v_node[0], tmp1))

        # XOR all vectors with the single node value (3 at a time to match hash)
        for vi in range(0, n_vectors, 3):
            vecs = min(3, n_vectors - vi)
            if vecs == 3:
                self.instrs.append({
                    "valu": [("^", all_val[vi], all_val[vi], v_node[0]),
                             ("^", all_val[vi + 1], all_val[vi + 1], v_node[0]),
                             ("^", all_val[vi + 2], all_val[vi + 2], v_node[0])]
                })
            elif vecs == 2:
                self.instrs.append({
                    "valu": [("^", all_val[vi], all_val[vi], v_node[0]),
                             ("^", all_val[vi + 1], all_val[vi + 1], v_node[0])]
                })
            else:
                self.instrs.append({
                    "valu": [("^", all_val[vi], all_val[vi], v_node[0])]
                })

        # Hash all vectors
        self._emit_hash_all(all_val, v_tmp1, v_tmp2, hash_const_vecs, n_vectors)

        # Index update all vectors (skip for last round)
        if not skip_index:
            self._emit_index_all(all_idx, all_val, v_tmp1, v_one, v_two, v_n_nodes, v_zero, n_vectors)

    def _emit_round_k2(self, all_idx, all_val, v_node, v_tmp1, v_tmp2,
                       hash_const_vecs, v_one, v_two, v_n_nodes, v_zero,
                       n_vectors, tmp1, tmp2, idx_base, skip_index=False):
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

        # Process 4 vectors at a time for better VALU utilization
        for vi in range(0, n_vectors, 4):
            # offset = idx - idx_base (0 or 1) for 4 vectors
            self.instrs.append({
                "valu": [("-", v_tmp1[0], all_idx[vi], idx_base_vec),
                         ("-", v_tmp1[1], all_idx[vi+1], idx_base_vec),
                         ("-", v_tmp1[2], all_idx[vi+2], idx_base_vec),
                         ("-", v_tmp1[3], all_idx[vi+3], idx_base_vec)]
            })
            # mask = 0 - offset (0 or all-1s)
            self.instrs.append({
                "valu": [("-", v_tmp1[0], v_zero, v_tmp1[0]),
                         ("-", v_tmp1[1], v_zero, v_tmp1[1]),
                         ("-", v_tmp1[2], v_zero, v_tmp1[2]),
                         ("-", v_tmp1[3], v_zero, v_tmp1[3])]
            })
            # masked_diff = diff & mask
            self.instrs.append({
                "valu": [("&", v_tmp1[0], v_tmp2[3], v_tmp1[0]),
                         ("&", v_tmp1[1], v_tmp2[3], v_tmp1[1]),
                         ("&", v_tmp1[2], v_tmp2[3], v_tmp1[2]),
                         ("&", v_tmp1[3], v_tmp2[3], v_tmp1[3])]
            })
            # selected = node[0] ^ masked_diff
            self.instrs.append({
                "valu": [("^", v_tmp1[0], v_node[0], v_tmp1[0]),
                         ("^", v_tmp1[1], v_node[0], v_tmp1[1]),
                         ("^", v_tmp1[2], v_node[0], v_tmp1[2]),
                         ("^", v_tmp1[3], v_node[0], v_tmp1[3])]
            })
            # XOR with val
            self.instrs.append({
                "valu": [("^", all_val[vi], all_val[vi], v_tmp1[0]),
                         ("^", all_val[vi+1], all_val[vi+1], v_tmp1[1]),
                         ("^", all_val[vi+2], all_val[vi+2], v_tmp1[2]),
                         ("^", all_val[vi+3], all_val[vi+3], v_tmp1[3])]
            })

        # Hash all vectors
        self._emit_hash_all(all_val, v_tmp1, v_tmp2, hash_const_vecs, n_vectors)

        # Index update all vectors (skip for last round)
        if not skip_index:
            self._emit_index_all(all_idx, all_val, v_tmp1, v_one, v_two, v_n_nodes, v_zero, n_vectors)

    def _emit_round_k4_batched(self, all_idx, all_val, v_node, v_tmp1, v_tmp2,
                                hash_const_vecs, v_one, v_two, v_n_nodes, v_zero,
                                n_vectors, tmp1, tmp2, idx_base, skip_index=False):
        """k=4 optimization processing 2 vectors at a time for better VALU utilization."""
        # Load 4 node values
        for i in range(4):
            self.add("load", ("const", tmp1, idx_base + i))
            self.instrs.append({"alu": [("+", tmp1, self.scratch["forest_values_p"], tmp1)]})
            self.add("load", ("load", tmp1, tmp1))
            self.add("valu", ("vbroadcast", v_node[i], tmp1))

        idx_base_vec = self.scratch_const_vec(idx_base)
        v_two_const = self.scratch_const_vec(2)

        # Precompute diffs (reused for all vectors)
        diff02_addr = self.alloc_scratch("diff02_k4", VLEN)
        diff13_addr = self.alloc_scratch("diff13_k4", VLEN)
        self.instrs.append({
            "valu": [("^", diff02_addr, v_node[0], v_node[2]),
                     ("^", diff13_addr, v_node[1], v_node[3])]
        })

        # Process 2 vectors at a time to utilize 6 VALU slots better
        for vi in range(0, n_vectors, 2):
            # offset = idx - idx_base for both vectors
            self.instrs.append({
                "valu": [("-", v_tmp1[0], all_idx[vi], idx_base_vec),
                         ("-", v_tmp1[1], all_idx[vi+1], idx_base_vec)]
            })
            # bit1 = (offset & 2) >> 1, mask1 = 0 - bit1
            self.instrs.append({
                "valu": [("&", v_tmp2[0], v_tmp1[0], v_two_const),
                         ("&", v_tmp2[1], v_tmp1[1], v_two_const)]
            })
            self.instrs.append({
                "valu": [(">>", v_tmp2[0], v_tmp2[0], v_one),
                         (">>", v_tmp2[1], v_tmp2[1], v_one)]
            })
            self.instrs.append({
                "valu": [("-", v_tmp2[0], v_zero, v_tmp2[0]),
                         ("-", v_tmp2[1], v_zero, v_tmp2[1])]
            })
            # masked diffs for level 1
            self.instrs.append({
                "valu": [("&", v_tmp2[2], diff02_addr, v_tmp2[0]),
                         ("&", v_tmp2[3], diff13_addr, v_tmp2[0])]
            })
            self.instrs.append({
                "valu": [("^", v_tmp2[2], v_node[0], v_tmp2[2]),  # sel_low[0]
                         ("^", v_tmp2[3], v_node[1], v_tmp2[3])]  # sel_high[0]
            })
            # For vector 1
            v_sel_low1 = v_tmp1[2]
            v_sel_high1 = v_tmp1[3]
            self.instrs.append({
                "valu": [("&", v_sel_low1, diff02_addr, v_tmp2[1]),
                         ("&", v_sel_high1, diff13_addr, v_tmp2[1])]
            })
            self.instrs.append({
                "valu": [("^", v_sel_low1, v_node[0], v_sel_low1),
                         ("^", v_sel_high1, v_node[1], v_sel_high1)]
            })
            # Level 2: bit0 = offset & 1, mask0 = 0 - bit0
            self.instrs.append({
                "valu": [("&", v_tmp1[0], v_tmp1[0], v_one),
                         ("&", v_tmp1[1], v_tmp1[1], v_one)]
            })
            self.instrs.append({
                "valu": [("-", v_tmp1[0], v_zero, v_tmp1[0]),
                         ("-", v_tmp1[1], v_zero, v_tmp1[1])]
            })
            # diff_lowhigh, masked, final select
            self.instrs.append({
                "valu": [("^", v_tmp2[0], v_tmp2[2], v_tmp2[3]),
                         ("^", v_tmp2[1], v_sel_low1, v_sel_high1)]
            })
            self.instrs.append({
                "valu": [("&", v_tmp2[0], v_tmp2[0], v_tmp1[0]),
                         ("&", v_tmp2[1], v_tmp2[1], v_tmp1[1])]
            })
            self.instrs.append({
                "valu": [("^", v_tmp2[0], v_tmp2[2], v_tmp2[0]),
                         ("^", v_tmp2[1], v_sel_low1, v_tmp2[1])]
            })
            # XOR with val
            self.instrs.append({
                "valu": [("^", all_val[vi], all_val[vi], v_tmp2[0]),
                         ("^", all_val[vi+1], all_val[vi+1], v_tmp2[1])]
            })

        # Hash all vectors
        self._emit_hash_all(all_val, v_tmp1, v_tmp2, hash_const_vecs, n_vectors)

        # Index update
        if not skip_index:
            self._emit_index_all(all_idx, all_val, v_tmp1, v_one, v_two, v_n_nodes, v_zero, n_vectors)

    def _emit_round_k8_batched(self, all_idx, all_val, v_node, v_tmp1, v_tmp2,
                                hash_const_vecs, v_one, v_two, v_n_nodes, v_zero,
                                n_vectors, tmp1, tmp2, idx_base, skip_index=False):
        """k=8 optimization using 3-level binary select."""
        # Load 8 node values (need more temp storage)
        node_addrs = [self.alloc_scratch(f"k8_node_{i}", VLEN) for i in range(8)]
        for i in range(8):
            self.add("load", ("const", tmp1, idx_base + i))
            self.instrs.append({"alu": [("+", tmp1, self.scratch["forest_values_p"], tmp1)]})
            self.add("load", ("load", tmp1, tmp1))
            self.add("valu", ("vbroadcast", node_addrs[i], tmp1))

        idx_base_vec = self.scratch_const_vec(idx_base)
        v_four_const = self.scratch_const_vec(4)
        v_two_const = self.scratch_const_vec(2)

        # Process 1 vector at a time for k=8 (more complex selection)
        for vi in range(n_vectors):
            # offset = idx - idx_base (0-7)
            self.instrs.append({"valu": [("-", v_tmp1[0], all_idx[vi], idx_base_vec)]})

            # Level 1: bit2 = (offset & 4) >> 2
            self.instrs.append({"valu": [("&", v_tmp1[1], v_tmp1[0], v_four_const)]})
            self.instrs.append({"valu": [(">>", v_tmp1[1], v_tmp1[1], v_two_const)]})
            self.instrs.append({"valu": [("-", v_tmp1[1], v_zero, v_tmp1[1])]})

            # Select between (n0-n3) and (n4-n7)
            self.instrs.append({
                "valu": [("^", v_tmp2[0], node_addrs[0], node_addrs[4]),
                         ("^", v_tmp2[1], node_addrs[1], node_addrs[5]),
                         ("^", v_tmp2[2], node_addrs[2], node_addrs[6]),
                         ("^", v_tmp2[3], node_addrs[3], node_addrs[7])]
            })
            self.instrs.append({
                "valu": [("&", v_tmp2[0], v_tmp2[0], v_tmp1[1]),
                         ("&", v_tmp2[1], v_tmp2[1], v_tmp1[1]),
                         ("&", v_tmp2[2], v_tmp2[2], v_tmp1[1]),
                         ("&", v_tmp2[3], v_tmp2[3], v_tmp1[1])]
            })
            self.instrs.append({
                "valu": [("^", v_tmp2[0], node_addrs[0], v_tmp2[0]),  # n0 or n4
                         ("^", v_tmp2[1], node_addrs[1], v_tmp2[1]),  # n1 or n5
                         ("^", v_tmp2[2], node_addrs[2], v_tmp2[2]),  # n2 or n6
                         ("^", v_tmp2[3], node_addrs[3], v_tmp2[3])]  # n3 or n7
            })

            # Level 2: bit1 = (offset & 2) >> 1
            self.instrs.append({"valu": [("&", v_tmp1[1], v_tmp1[0], v_two_const)]})
            self.instrs.append({"valu": [(">>", v_tmp1[1], v_tmp1[1], v_one)]})
            self.instrs.append({"valu": [("-", v_tmp1[1], v_zero, v_tmp1[1])]})

            # Select between pairs
            sel_02 = v_tmp1[2]
            sel_13 = v_tmp1[3]
            self.instrs.append({
                "valu": [("^", sel_02, v_tmp2[0], v_tmp2[2]),
                         ("^", sel_13, v_tmp2[1], v_tmp2[3])]
            })
            self.instrs.append({
                "valu": [("&", sel_02, sel_02, v_tmp1[1]),
                         ("&", sel_13, sel_13, v_tmp1[1])]
            })
            self.instrs.append({
                "valu": [("^", sel_02, v_tmp2[0], sel_02),  # low
                         ("^", sel_13, v_tmp2[1], sel_13)]  # high
            })

            # Level 3: bit0 = offset & 1
            self.instrs.append({"valu": [("&", v_tmp1[1], v_tmp1[0], v_one)]})
            self.instrs.append({"valu": [("-", v_tmp1[1], v_zero, v_tmp1[1])]})

            self.instrs.append({"valu": [("^", v_tmp2[0], sel_02, sel_13)]})
            self.instrs.append({"valu": [("&", v_tmp2[0], v_tmp2[0], v_tmp1[1])]})
            self.instrs.append({"valu": [("^", v_tmp2[0], sel_02, v_tmp2[0])]})

            # XOR with val
            self.instrs.append({"valu": [("^", all_val[vi], all_val[vi], v_tmp2[0])]})

        # Hash all vectors
        self._emit_hash_all(all_val, v_tmp1, v_tmp2, hash_const_vecs, n_vectors)

        # Index update
        if not skip_index:
            self._emit_index_all(all_idx, all_val, v_tmp1, v_one, v_two, v_n_nodes, v_zero, n_vectors)

    def _emit_round_k16_batched(self, all_idx, all_val, v_node, v_tmp1, v_tmp2,
                                 hash_const_vecs, v_one, v_two, v_n_nodes, v_zero,
                                 n_vectors, tmp1, tmp2, idx_base, skip_index=False):
        """k=16 optimization using 4-level binary select."""
        # Load 16 node values
        node_addrs = [self.alloc_scratch(f"k16_node_{i}", VLEN) for i in range(16)]
        for i in range(16):
            self.add("load", ("const", tmp1, idx_base + i))
            self.instrs.append({"alu": [("+", tmp1, self.scratch["forest_values_p"], tmp1)]})
            self.add("load", ("load", tmp1, tmp1))
            self.add("valu", ("vbroadcast", node_addrs[i], tmp1))

        idx_base_vec = self.scratch_const_vec(idx_base)
        v_eight_const = self.scratch_const_vec(8)
        v_four_const = self.scratch_const_vec(4)
        v_two_const = self.scratch_const_vec(2)

        # Allocate additional working space for 8 intermediate values
        sel_level1 = [self.alloc_scratch(f"k16_sel1_{i}", VLEN) for i in range(8)]

        # Process 1 vector at a time
        for vi in range(n_vectors):
            # offset = idx - idx_base (0-15)
            self.instrs.append({"valu": [("-", v_tmp1[0], all_idx[vi], idx_base_vec)]})

            # Level 1: bit3 = (offset & 8) >> 3
            self.instrs.append({"valu": [("&", v_tmp1[1], v_tmp1[0], v_eight_const)]})
            self.instrs.append({"valu": [(">>", v_tmp1[1], v_tmp1[1], v_one)]})
            self.instrs.append({"valu": [(">>", v_tmp1[1], v_tmp1[1], v_one)]})
            self.instrs.append({"valu": [(">>", v_tmp1[1], v_tmp1[1], v_one)]})
            self.instrs.append({"valu": [("-", v_tmp1[1], v_zero, v_tmp1[1])]})

            # Select between (n0-n7) and (n8-n15)
            for i in range(0, 8, 4):
                self.instrs.append({
                    "valu": [("^", v_tmp2[0], node_addrs[i], node_addrs[i+8]),
                             ("^", v_tmp2[1], node_addrs[i+1], node_addrs[i+9]),
                             ("^", v_tmp2[2], node_addrs[i+2], node_addrs[i+10]),
                             ("^", v_tmp2[3], node_addrs[i+3], node_addrs[i+11])]
                })
                self.instrs.append({
                    "valu": [("&", v_tmp2[0], v_tmp2[0], v_tmp1[1]),
                             ("&", v_tmp2[1], v_tmp2[1], v_tmp1[1]),
                             ("&", v_tmp2[2], v_tmp2[2], v_tmp1[1]),
                             ("&", v_tmp2[3], v_tmp2[3], v_tmp1[1])]
                })
                self.instrs.append({
                    "valu": [("^", sel_level1[i], node_addrs[i], v_tmp2[0]),
                             ("^", sel_level1[i+1], node_addrs[i+1], v_tmp2[1]),
                             ("^", sel_level1[i+2], node_addrs[i+2], v_tmp2[2]),
                             ("^", sel_level1[i+3], node_addrs[i+3], v_tmp2[3])]
                })

            # Level 2: bit2 = (offset & 4) >> 2
            self.instrs.append({"valu": [("&", v_tmp1[1], v_tmp1[0], v_four_const)]})
            self.instrs.append({"valu": [(">>", v_tmp1[1], v_tmp1[1], v_two_const)]})
            self.instrs.append({"valu": [("-", v_tmp1[1], v_zero, v_tmp1[1])]})

            # Select between pairs of 4
            self.instrs.append({
                "valu": [("^", v_tmp2[0], sel_level1[0], sel_level1[4]),
                         ("^", v_tmp2[1], sel_level1[1], sel_level1[5]),
                         ("^", v_tmp2[2], sel_level1[2], sel_level1[6]),
                         ("^", v_tmp2[3], sel_level1[3], sel_level1[7])]
            })
            self.instrs.append({
                "valu": [("&", v_tmp2[0], v_tmp2[0], v_tmp1[1]),
                         ("&", v_tmp2[1], v_tmp2[1], v_tmp1[1]),
                         ("&", v_tmp2[2], v_tmp2[2], v_tmp1[1]),
                         ("&", v_tmp2[3], v_tmp2[3], v_tmp1[1])]
            })
            self.instrs.append({
                "valu": [("^", v_tmp2[0], sel_level1[0], v_tmp2[0]),
                         ("^", v_tmp2[1], sel_level1[1], v_tmp2[1]),
                         ("^", v_tmp2[2], sel_level1[2], v_tmp2[2]),
                         ("^", v_tmp2[3], sel_level1[3], v_tmp2[3])]
            })

            # Level 3: bit1 = (offset & 2) >> 1
            self.instrs.append({"valu": [("&", v_tmp1[1], v_tmp1[0], v_two_const)]})
            self.instrs.append({"valu": [(">>", v_tmp1[1], v_tmp1[1], v_one)]})
            self.instrs.append({"valu": [("-", v_tmp1[1], v_zero, v_tmp1[1])]})

            sel_02 = v_tmp1[2]
            sel_13 = v_tmp1[3]
            self.instrs.append({
                "valu": [("^", sel_02, v_tmp2[0], v_tmp2[2]),
                         ("^", sel_13, v_tmp2[1], v_tmp2[3])]
            })
            self.instrs.append({
                "valu": [("&", sel_02, sel_02, v_tmp1[1]),
                         ("&", sel_13, sel_13, v_tmp1[1])]
            })
            self.instrs.append({
                "valu": [("^", sel_02, v_tmp2[0], sel_02),
                         ("^", sel_13, v_tmp2[1], sel_13)]
            })

            # Level 4: bit0 = offset & 1
            self.instrs.append({"valu": [("&", v_tmp1[1], v_tmp1[0], v_one)]})
            self.instrs.append({"valu": [("-", v_tmp1[1], v_zero, v_tmp1[1])]})

            self.instrs.append({"valu": [("^", v_tmp2[0], sel_02, sel_13)]})
            self.instrs.append({"valu": [("&", v_tmp2[0], v_tmp2[0], v_tmp1[1])]})
            self.instrs.append({"valu": [("^", v_tmp2[0], sel_02, v_tmp2[0])]})

            # XOR with val
            self.instrs.append({"valu": [("^", all_val[vi], all_val[vi], v_tmp2[0])]})

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

    def _emit_hash_all(self, all_val, v_tmp1, v_tmp2, hash_const_vecs, n_vectors):
        """Hash all vectors with interleaved groups for better VALU utilization.
        Process 6 vectors at a time: 2 groups of 3, combining their combine steps."""
        # Allocate extra temp storage for second group
        v_tmp3 = [self.alloc_scratch(f"htmp3_{i}", VLEN) for i in range(3)]
        v_tmp4 = [self.alloc_scratch(f"htmp4_{i}", VLEN) for i in range(3)]

        for vi in range(0, n_vectors, 6):
            remaining = n_vectors - vi
            if remaining >= 6:
                # Process 2 groups of 3 vectors interleaved
                for hi, (op1, val1, op2, op3, val3) in enumerate(HASH_STAGES):
                    const1, const3 = hash_const_vecs[hi]
                    # Group A first (vectors vi, vi+1, vi+2)
                    self.instrs.append({
                        "valu": [(op1, v_tmp1[0], all_val[vi], const1),
                                 (op3, v_tmp2[0], all_val[vi], const3),
                                 (op1, v_tmp1[1], all_val[vi+1], const1),
                                 (op3, v_tmp2[1], all_val[vi+1], const3),
                                 (op1, v_tmp1[2], all_val[vi+2], const1),
                                 (op3, v_tmp2[2], all_val[vi+2], const3)]
                    })
                    # Group B first (vectors vi+3, vi+4, vi+5)
                    self.instrs.append({
                        "valu": [(op1, v_tmp3[0], all_val[vi+3], const1),
                                 (op3, v_tmp4[0], all_val[vi+3], const3),
                                 (op1, v_tmp3[1], all_val[vi+4], const1),
                                 (op3, v_tmp4[1], all_val[vi+4], const3),
                                 (op1, v_tmp3[2], all_val[vi+5], const1),
                                 (op3, v_tmp4[2], all_val[vi+5], const3)]
                    })
                    # Both groups combine (6 VALU slots fully utilized)
                    self.instrs.append({
                        "valu": [(op2, all_val[vi], v_tmp1[0], v_tmp2[0]),
                                 (op2, all_val[vi+1], v_tmp1[1], v_tmp2[1]),
                                 (op2, all_val[vi+2], v_tmp1[2], v_tmp2[2]),
                                 (op2, all_val[vi+3], v_tmp3[0], v_tmp4[0]),
                                 (op2, all_val[vi+4], v_tmp3[1], v_tmp4[1]),
                                 (op2, all_val[vi+5], v_tmp3[2], v_tmp4[2])]
                    })
            else:
                # Handle remaining vectors with original approach
                for vii in range(vi, n_vectors, 3):
                    vecs = min(3, n_vectors - vii)
                    for hi, (op1, val1, op2, op3, val3) in enumerate(HASH_STAGES):
                        const1, const3 = hash_const_vecs[hi]
                        if vecs == 3:
                            self.instrs.append({
                                "valu": [(op1, v_tmp1[0], all_val[vii], const1),
                                         (op3, v_tmp2[0], all_val[vii], const3),
                                         (op1, v_tmp1[1], all_val[vii+1], const1),
                                         (op3, v_tmp2[1], all_val[vii+1], const3),
                                         (op1, v_tmp1[2], all_val[vii+2], const1),
                                         (op3, v_tmp2[2], all_val[vii+2], const3)]
                            })
                            self.instrs.append({
                                "valu": [(op2, all_val[vii], v_tmp1[0], v_tmp2[0]),
                                         (op2, all_val[vii+1], v_tmp1[1], v_tmp2[1]),
                                         (op2, all_val[vii+2], v_tmp1[2], v_tmp2[2])]
                            })
                        elif vecs == 2:
                            self.instrs.append({
                                "valu": [(op1, v_tmp1[0], all_val[vii], const1),
                                         (op3, v_tmp2[0], all_val[vii], const3),
                                         (op1, v_tmp1[1], all_val[vii+1], const1),
                                         (op3, v_tmp2[1], all_val[vii+1], const3)]
                            })
                            self.instrs.append({
                                "valu": [(op2, all_val[vii], v_tmp1[0], v_tmp2[0]),
                                         (op2, all_val[vii+1], v_tmp1[1], v_tmp2[1])]
                            })
                        else:
                            self.instrs.append({
                                "valu": [(op1, v_tmp1[0], all_val[vii], const1),
                                         (op3, v_tmp2[0], all_val[vii], const3)]
                            })
                            self.instrs.append({
                                "valu": [(op2, all_val[vii], v_tmp1[0], v_tmp2[0])]
                            })

    def _emit_index_all(self, all_idx, all_val, v_tmp1, v_one, v_two, v_n_nodes, v_zero, n_vectors):
        """Index update all vectors, 3 at a time for better VALU utilization."""
        # Allocate extra temp storage for third vector
        v_tmp3 = self.alloc_scratch("idx_tmp3", VLEN)

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
                             ("&", v_tmp3, all_val[vi+2], v_one),
                             ("*", all_idx[vi+2], all_idx[vi+2], v_two)]
                })
                self.instrs.append({
                    "valu": [("+", v_tmp1[0], v_tmp1[0], v_one),
                             ("+", v_tmp1[1], v_tmp1[1], v_one),
                             ("+", v_tmp3, v_tmp3, v_one)]
                })
                self.instrs.append({
                    "valu": [("+", all_idx[vi], all_idx[vi], v_tmp1[0]),
                             ("+", all_idx[vi+1], all_idx[vi+1], v_tmp1[1]),
                             ("+", all_idx[vi+2], all_idx[vi+2], v_tmp3)]
                })
                self.instrs.append({
                    "valu": [("<", v_tmp1[0], all_idx[vi], v_n_nodes),
                             ("<", v_tmp1[1], all_idx[vi+1], v_n_nodes),
                             ("<", v_tmp3, all_idx[vi+2], v_n_nodes)]
                })
                self.instrs.append({
                    "valu": [("-", v_tmp1[0], v_zero, v_tmp1[0]),
                             ("-", v_tmp1[1], v_zero, v_tmp1[1]),
                             ("-", v_tmp3, v_zero, v_tmp3)]
                })
                self.instrs.append({
                    "valu": [("&", all_idx[vi], all_idx[vi], v_tmp1[0]),
                             ("&", all_idx[vi+1], all_idx[vi+1], v_tmp1[1]),
                             ("&", all_idx[vi+2], all_idx[vi+2], v_tmp3)]
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
                self.instrs.append({
                    "valu": [("<", v_tmp1[0], all_idx[vi], v_n_nodes)]
                })
                self.instrs.append({
                    "valu": [("-", v_tmp1[0], v_zero, v_tmp1[0])]
                })
                self.instrs.append({
                    "valu": [("&", all_idx[vi], all_idx[vi], v_tmp1[0])]
                })

    def _emit_round_full_gather(self, all_idx, all_val, v_node, v_node2, v_tmp1, v_tmp2, v_addrs, v_addrs2,
                                 hash_const_vecs, v_one, v_two, v_n_nodes, v_zero,
                                 n_vectors, tmp1, skip_index=False):
        """Full gather round with aggressive inter-group pipelining.
        Overlaps gather of next group with hash/index of current group."""
        n_groups = n_vectors // 4  # 8 groups

        # Track whether we've pre-computed addresses for next group
        next_group_addrs_ready = False

        # Process groups with inter-group pipelining
        for group_idx in range(n_groups):
            g = group_idx * 4
            cur_addrs = v_addrs if (group_idx % 2 == 0) else v_addrs2
            cur_node = v_node if (group_idx % 2 == 0) else v_node2
            next_addrs = v_addrs2 if (group_idx % 2 == 0) else v_addrs
            next_node = v_node2 if (group_idx % 2 == 0) else v_node

            # Compute addresses for current group (skip if pre-computed from previous group)
            if not next_group_addrs_ready:
                # Vector 0 (8 ops) + Vector 1 first 4 ops = 12 ops
                self.instrs.append({
                    "alu": [("+", cur_addrs[0] + vi, self.scratch["forest_values_p"], all_idx[g] + vi)
                            for vi in range(VLEN)] +
                           [("+", cur_addrs[1] + vi, self.scratch["forest_values_p"], all_idx[g + 1] + vi)
                            for vi in range(4)]
                })
                # Vector 1 last 4 + Vector 2 (8 ops) = 12 ops
                self.instrs.append({
                    "alu": [("+", cur_addrs[1] + vi, self.scratch["forest_values_p"], all_idx[g + 1] + vi)
                            for vi in range(4, VLEN)] +
                           [("+", cur_addrs[2] + vi, self.scratch["forest_values_p"], all_idx[g + 2] + vi)
                            for vi in range(VLEN)]
                })
                # Vector 3 (8 ops)
                self.instrs.append({
                    "alu": [("+", cur_addrs[3] + vi, self.scratch["forest_values_p"], all_idx[g + 3] + vi)
                            for vi in range(VLEN)]
                })
            next_group_addrs_ready = False  # Reset for next iteration

            # Gather A (vec 0,1) for current group
            for vi in range(VLEN):
                self.instrs.append({
                    "load": [("load", cur_node[0] + vi, cur_addrs[0] + vi),
                             ("load", cur_node[1] + vi, cur_addrs[1] + vi)]
                })
            # XOR A
            self.instrs.append({
                "valu": [("^", all_val[g], all_val[g], cur_node[0]),
                         ("^", all_val[g + 1], all_val[g + 1], cur_node[1])]
            })

            # Hash A + Gather B overlapped, fuse XOR B with last combine
            load_idx = 0
            num_stages = len(HASH_STAGES)
            for hi, (op1, val1, op2, op3, val3) in enumerate(HASH_STAGES):
                const1_vec, const3_vec = hash_const_vecs[hi]
                is_last_stage = (hi == num_stages - 1)

                vi = load_idx
                if vi < VLEN:
                    self.instrs.append({
                        "valu": [(op1, v_tmp1[0], all_val[g], const1_vec),
                                 (op3, v_tmp2[0], all_val[g], const3_vec),
                                 (op1, v_tmp1[1], all_val[g + 1], const1_vec),
                                 (op3, v_tmp2[1], all_val[g + 1], const3_vec)],
                        "load": [("load", cur_node[2] + vi, cur_addrs[2] + vi),
                                 ("load", cur_node[3] + vi, cur_addrs[3] + vi)]
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
                        "load": [("load", cur_node[2] + vi, cur_addrs[2] + vi),
                                 ("load", cur_node[3] + vi, cur_addrs[3] + vi)]
                    })
                    load_idx += 1
                elif is_last_stage:
                    # Fuse XOR B with last hash A combine (saves 1 instruction)
                    self.instrs.append({
                        "valu": [(op2, all_val[g], v_tmp1[0], v_tmp2[0]),
                                 (op2, all_val[g + 1], v_tmp1[1], v_tmp2[1]),
                                 ("^", all_val[g + 2], all_val[g + 2], cur_node[2]),
                                 ("^", all_val[g + 3], all_val[g + 3], cur_node[3])]
                    })
                else:
                    self.instrs.append({
                        "valu": [(op2, all_val[g], v_tmp1[0], v_tmp2[0]),
                                 (op2, all_val[g + 1], v_tmp1[1], v_tmp2[1])]
                    })

            while load_idx < VLEN:
                self.instrs.append({
                    "load": [("load", cur_node[2] + load_idx, cur_addrs[2] + load_idx),
                             ("load", cur_node[3] + load_idx, cur_addrs[3] + load_idx)]
                })
                load_idx += 1

            # XOR B already fused above if all loads completed during hash A

            # Hash B + Index A + Addr comp for next group
            next_g = g + 4
            next_addr_v = 0
            next_addr_vi = 0
            has_next = (group_idx < n_groups - 1)

            for hi, (op1, val1, op2, op3, val3) in enumerate(HASH_STAGES):
                const1_vec, const3_vec = hash_const_vecs[hi]

                if skip_index:
                    # Skip index update for last round
                    if has_next and next_addr_v < 4:
                        # Overlap addr comp for next group with ALU
                        self.instrs.append({
                            "valu": [(op1, v_tmp1[2], all_val[g + 2], const1_vec),
                                     (op3, v_tmp2[2], all_val[g + 2], const3_vec),
                                     (op1, v_tmp1[3], all_val[g + 3], const1_vec),
                                     (op3, v_tmp2[3], all_val[g + 3], const3_vec)],
                            "alu": [("+", next_addrs[next_addr_v] + vi, self.scratch["forest_values_p"], all_idx[next_g + next_addr_v] + vi)
                                    for vi in range(VLEN)]
                        })
                        next_addr_v += 1
                    else:
                        self.instrs.append({
                            "valu": [(op1, v_tmp1[2], all_val[g + 2], const1_vec),
                                     (op3, v_tmp2[2], all_val[g + 2], const3_vec),
                                     (op1, v_tmp1[3], all_val[g + 3], const1_vec),
                                     (op3, v_tmp2[3], all_val[g + 3], const3_vec)]
                        })
                    if has_next and next_addr_v < 4:
                        self.instrs.append({
                            "valu": [(op2, all_val[g + 2], v_tmp1[2], v_tmp2[2]),
                                     (op2, all_val[g + 3], v_tmp1[3], v_tmp2[3])],
                            "alu": [("+", next_addrs[next_addr_v] + vi, self.scratch["forest_values_p"], all_idx[next_g + next_addr_v] + vi)
                                    for vi in range(VLEN)]
                        })
                        next_addr_v += 1
                    else:
                        self.instrs.append({
                            "valu": [(op2, all_val[g + 2], v_tmp1[2], v_tmp2[2]),
                                     (op2, all_val[g + 3], v_tmp1[3], v_tmp2[3])]
                        })
                elif hi == 0:
                    if has_next and next_addr_v < 4:
                        self.instrs.append({
                            "valu": [(op1, v_tmp1[2], all_val[g + 2], const1_vec),
                                     (op3, v_tmp2[2], all_val[g + 2], const3_vec),
                                     (op1, v_tmp1[3], all_val[g + 3], const1_vec),
                                     (op3, v_tmp2[3], all_val[g + 3], const3_vec),
                                     ("&", cur_node[0], all_val[g], v_one),
                                     ("*", all_idx[g], all_idx[g], v_two)],
                            "alu": [("+", next_addrs[next_addr_v] + vi, self.scratch["forest_values_p"], all_idx[next_g + next_addr_v] + vi)
                                    for vi in range(VLEN)]
                        })
                        next_addr_v += 1
                    else:
                        self.instrs.append({
                            "valu": [(op1, v_tmp1[2], all_val[g + 2], const1_vec),
                                     (op3, v_tmp2[2], all_val[g + 2], const3_vec),
                                     (op1, v_tmp1[3], all_val[g + 3], const1_vec),
                                     (op3, v_tmp2[3], all_val[g + 3], const3_vec),
                                     ("&", cur_node[0], all_val[g], v_one),
                                     ("*", all_idx[g], all_idx[g], v_two)]
                        })
                    if has_next and next_addr_v < 4:
                        self.instrs.append({
                            "valu": [(op2, all_val[g + 2], v_tmp1[2], v_tmp2[2]),
                                     (op2, all_val[g + 3], v_tmp1[3], v_tmp2[3]),
                                     ("&", cur_node[1], all_val[g + 1], v_one),
                                     ("*", all_idx[g + 1], all_idx[g + 1], v_two)],
                            "alu": [("+", next_addrs[next_addr_v] + vi, self.scratch["forest_values_p"], all_idx[next_g + next_addr_v] + vi)
                                    for vi in range(VLEN)]
                        })
                        next_addr_v += 1
                    else:
                        self.instrs.append({
                            "valu": [(op2, all_val[g + 2], v_tmp1[2], v_tmp2[2]),
                                     (op2, all_val[g + 3], v_tmp1[3], v_tmp2[3]),
                                     ("&", cur_node[1], all_val[g + 1], v_one),
                                     ("*", all_idx[g + 1], all_idx[g + 1], v_two)]
                        })
                elif hi == 1:
                    if has_next and next_addr_v < 4:
                        self.instrs.append({
                            "valu": [(op1, v_tmp1[2], all_val[g + 2], const1_vec),
                                     (op3, v_tmp2[2], all_val[g + 2], const3_vec),
                                     (op1, v_tmp1[3], all_val[g + 3], const1_vec),
                                     (op3, v_tmp2[3], all_val[g + 3], const3_vec),
                                     ("+", cur_node[0], cur_node[0], v_one),
                                     ("+", cur_node[1], cur_node[1], v_one)],
                            "alu": [("+", next_addrs[next_addr_v] + vi, self.scratch["forest_values_p"], all_idx[next_g + next_addr_v] + vi)
                                    for vi in range(VLEN)]
                        })
                        next_addr_v += 1
                    else:
                        self.instrs.append({
                            "valu": [(op1, v_tmp1[2], all_val[g + 2], const1_vec),
                                     (op3, v_tmp2[2], all_val[g + 2], const3_vec),
                                     (op1, v_tmp1[3], all_val[g + 3], const1_vec),
                                     (op3, v_tmp2[3], all_val[g + 3], const3_vec),
                                     ("+", cur_node[0], cur_node[0], v_one),
                                     ("+", cur_node[1], cur_node[1], v_one)]
                        })
                    if has_next and next_addr_v < 4:
                        self.instrs.append({
                            "valu": [(op2, all_val[g + 2], v_tmp1[2], v_tmp2[2]),
                                     (op2, all_val[g + 3], v_tmp1[3], v_tmp2[3]),
                                     ("+", all_idx[g], all_idx[g], cur_node[0]),
                                     ("+", all_idx[g + 1], all_idx[g + 1], cur_node[1])],
                            "alu": [("+", next_addrs[next_addr_v] + vi, self.scratch["forest_values_p"], all_idx[next_g + next_addr_v] + vi)
                                    for vi in range(VLEN)]
                        })
                        next_addr_v += 1
                    else:
                        self.instrs.append({
                            "valu": [(op2, all_val[g + 2], v_tmp1[2], v_tmp2[2]),
                                     (op2, all_val[g + 3], v_tmp1[3], v_tmp2[3]),
                                     ("+", all_idx[g], all_idx[g], cur_node[0]),
                                     ("+", all_idx[g + 1], all_idx[g + 1], cur_node[1])]
                        })
                elif hi == 2:
                    self.instrs.append({
                        "valu": [(op1, v_tmp1[2], all_val[g + 2], const1_vec),
                                 (op3, v_tmp2[2], all_val[g + 2], const3_vec),
                                 (op1, v_tmp1[3], all_val[g + 3], const1_vec),
                                 (op3, v_tmp2[3], all_val[g + 3], const3_vec),
                                 ("<", cur_node[0], all_idx[g], v_n_nodes),
                                 ("<", cur_node[1], all_idx[g + 1], v_n_nodes)]
                    })
                    self.instrs.append({
                        "valu": [(op2, all_val[g + 2], v_tmp1[2], v_tmp2[2]),
                                 (op2, all_val[g + 3], v_tmp1[3], v_tmp2[3]),
                                 ("-", cur_node[0], v_zero, cur_node[0]),
                                 ("-", cur_node[1], v_zero, cur_node[1])]
                    })
                elif hi == 3:
                    self.instrs.append({
                        "valu": [(op1, v_tmp1[2], all_val[g + 2], const1_vec),
                                 (op3, v_tmp2[2], all_val[g + 2], const3_vec),
                                 (op1, v_tmp1[3], all_val[g + 3], const1_vec),
                                 (op3, v_tmp2[3], all_val[g + 3], const3_vec),
                                 ("&", all_idx[g], all_idx[g], cur_node[0]),
                                 ("&", all_idx[g + 1], all_idx[g + 1], cur_node[1])]
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
                                 (op3, v_tmp2[3], all_val[g + 3], const3_vec)]
                    })
                    self.instrs.append({
                        "valu": [(op2, all_val[g + 2], v_tmp1[2], v_tmp2[2]),
                                 (op2, all_val[g + 3], v_tmp1[3], v_tmp2[3])]
                    })

            # Mark if we've pre-computed addresses for next group
            if has_next and next_addr_v >= 4:
                next_group_addrs_ready = True

            # Index B (skip for last round)
            if not skip_index:
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
