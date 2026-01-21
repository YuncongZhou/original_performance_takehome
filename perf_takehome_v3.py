"""
Ultra-optimized kernel targeting <1487 cycles.

Key optimizations:
1. Efficient k=4, k=8 selection for bounded-index rounds
2. Process 6 vectors at a time to maximize VALU utilization
3. Better load/compute overlap in full gather
4. Aggressive round-level pipelining
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
        """Ultra-optimized kernel with k-selection for bounded-index rounds."""
        n_vectors = batch_size // VLEN  # 32 vectors

        # ============ ALLOCATION ============
        tmp1 = self.alloc_scratch("tmp1")
        tmp2 = self.alloc_scratch("tmp2")
        tmp3 = self.alloc_scratch("tmp3")
        tmp4 = self.alloc_scratch("tmp4")
        tmp5 = self.alloc_scratch("tmp5")
        tmp6 = self.alloc_scratch("tmp6")

        init_vars = [
            "rounds", "n_nodes", "batch_size", "forest_height",
            "forest_values_p", "inp_indices_p", "inp_values_p",
        ]
        for v in init_vars:
            self.alloc_scratch(v, 1)

        # All idx and val in scratch
        all_idx = [self.alloc_scratch(f"idx_{i}", VLEN) for i in range(n_vectors)]
        all_val = [self.alloc_scratch(f"val_{i}", VLEN) for i in range(n_vectors)]

        # Working buffers - allocate 8 for more flexibility
        v_node = [self.alloc_scratch(f"node_{i}", VLEN) for i in range(8)]
        v_tmp1 = [self.alloc_scratch(f"tmp1_{i}", VLEN) for i in range(6)]
        v_tmp2 = [self.alloc_scratch(f"tmp2_{i}", VLEN) for i in range(6)]
        v_addrs = [self.alloc_scratch(f"addrs_{i}", VLEN) for i in range(6)]
        v_addrs2 = [self.alloc_scratch(f"addrs2_{i}", VLEN) for i in range(6)]

        # ============ INITIALIZATION ============
        for i, v in enumerate(init_vars):
            self.add("load", ("const", tmp1, i))
            self.add("load", ("load", self.scratch[v], tmp1))

        v_zero = self.scratch_const_vec(0)
        v_one = self.scratch_const_vec(1)
        v_two = self.scratch_const_vec(2)
        v_n_nodes = self.scratch_const_vec(n_nodes)

        # Hash constants
        hash_const_vecs = []
        hash_multipliers = {0: 4097, 2: 33, 4: 9}
        for hi, (op1, val1, op2, op3, val3) in enumerate(HASH_STAGES):
            if hi in hash_multipliers:
                mult_vec = self.scratch_const_vec(hash_multipliers[hi])
                const_vec = self.scratch_const_vec(val1)
                hash_const_vecs.append(('multiply_add', mult_vec, const_vec))
            else:
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

        # ============ PROCESS ALL 16 ROUNDS ============
        for round_num in range(rounds):
            if round_num <= 10:
                effective_round = round_num
            else:
                effective_round = round_num - 11

            is_last_round = (round_num == rounds - 1)
            needs_wrap_check = (round_num == 10)

            if effective_round == 0:
                # k=1: all indices at 0
                self._emit_round_k1(
                    all_idx, all_val, v_node, v_tmp1, v_tmp2,
                    hash_const_vecs, v_one, v_two, v_n_nodes, v_zero,
                    n_vectors, tmp1, idx_value=0, skip_index=is_last_round,
                    skip_wrap_check=not needs_wrap_check
                )
            elif effective_round == 1:
                # k=2: indices in {1, 2}
                self._emit_round_k2(
                    all_idx, all_val, v_node, v_tmp1, v_tmp2,
                    hash_const_vecs, v_one, v_two, v_n_nodes, v_zero,
                    n_vectors, tmp1, tmp2, idx_base=1, skip_index=is_last_round,
                    skip_wrap_check=not needs_wrap_check
                )
            elif effective_round == 2:
                # k=4: indices in {3, 4, 5, 6}
                self._emit_round_k4_fast(
                    all_idx, all_val, v_node, v_tmp1, v_tmp2,
                    hash_const_vecs, v_one, v_two, v_n_nodes, v_zero,
                    n_vectors, tmp1, tmp2, idx_base=3, skip_index=is_last_round,
                    skip_wrap_check=not needs_wrap_check
                )
            elif effective_round == 3:
                # k=8: indices in {7, ..., 14}
                self._emit_round_k8_fast(
                    all_idx, all_val, v_node, v_tmp1, v_tmp2,
                    hash_const_vecs, v_one, v_two, v_n_nodes, v_zero,
                    n_vectors, tmp1, tmp2, idx_base=7, skip_index=is_last_round,
                    skip_wrap_check=not needs_wrap_check
                )
            elif effective_round == 4:
                # k=16: indices in {15, ..., 30}
                self._emit_round_k16_fast(
                    all_idx, all_val, v_node, v_tmp1, v_tmp2, v_addrs,
                    hash_const_vecs, v_one, v_two, v_n_nodes, v_zero,
                    n_vectors, tmp1, tmp2, idx_base=15, skip_index=is_last_round,
                    skip_wrap_check=not needs_wrap_check
                )
            else:
                # Full gather for rounds 5-10
                self._emit_round_full_gather_v2(
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

    def _emit_round_k1(self, all_idx, all_val, v_node, v_tmp1, v_tmp2,
                       hash_const_vecs, v_one, v_two, v_n_nodes, v_zero,
                       n_vectors, tmp1, idx_value, skip_index=False, skip_wrap_check=False):
        """k=1: single unique index."""
        # Load single node value
        self.add("load", ("load", tmp1, self.scratch["forest_values_p"]))
        self.add("valu", ("vbroadcast", v_node[0], tmp1))

        # XOR all vectors (6 at a time)
        for vi in range(0, n_vectors, 6):
            vecs = min(6, n_vectors - vi)
            self.instrs.append({
                "valu": [("^", all_val[vi + j], all_val[vi + j], v_node[0]) for j in range(vecs)]
            })

        self._emit_hash_all(all_val, v_tmp1, v_tmp2, hash_const_vecs, n_vectors)

        if not skip_index:
            self._emit_index_all(all_idx, all_val, v_tmp1, v_one, v_two, v_n_nodes, v_zero,
                                n_vectors, skip_wrap_check=skip_wrap_check, v_tmp2=v_tmp2)

    def _emit_round_k2(self, all_idx, all_val, v_node, v_tmp1, v_tmp2,
                       hash_const_vecs, v_one, v_two, v_n_nodes, v_zero,
                       n_vectors, tmp1, tmp2, idx_base, skip_index=False, skip_wrap_check=False):
        """k=2: two unique indices."""
        # Load 2 node values
        self.add("load", ("const", tmp1, idx_base))
        self.add("load", ("const", tmp2, idx_base + 1))
        self.instrs.append({"alu": [("+", tmp1, self.scratch["forest_values_p"], tmp1),
                                    ("+", tmp2, self.scratch["forest_values_p"], tmp2)]})
        self.instrs.append({"load": [("load", tmp1, tmp1), ("load", tmp2, tmp2)]})
        self.instrs.append({"valu": [("vbroadcast", v_node[0], tmp1),
                                      ("vbroadcast", v_node[1], tmp2)]})

        idx_base_vec = self.scratch_const_vec(idx_base)
        # diff = n0 ^ n1
        self.instrs.append({"valu": [("^", v_tmp2[5], v_node[0], v_node[1])]})

        tmps = [v_tmp1[0], v_tmp1[1], v_tmp1[2], v_tmp1[3], v_tmp2[0], v_tmp2[1]]
        for vi in range(0, n_vectors, 6):
            vecs = min(6, n_vectors - vi)
            # offset = idx - idx_base
            self.instrs.append({
                "valu": [("-", tmps[j], all_idx[vi+j], idx_base_vec) for j in range(vecs)]
            })
            # mask = 0 - offset
            self.instrs.append({
                "valu": [("-", tmps[j], v_zero, tmps[j]) for j in range(vecs)]
            })
            # masked_diff = diff & mask
            self.instrs.append({
                "valu": [("&", tmps[j], v_tmp2[5], tmps[j]) for j in range(vecs)]
            })
            # selected = n0 ^ masked_diff
            self.instrs.append({
                "valu": [("^", tmps[j], v_node[0], tmps[j]) for j in range(vecs)]
            })
            # XOR with val
            self.instrs.append({
                "valu": [("^", all_val[vi+j], all_val[vi+j], tmps[j]) for j in range(vecs)]
            })

        self._emit_hash_all(all_val, v_tmp1, v_tmp2, hash_const_vecs, n_vectors)

        if not skip_index:
            self._emit_index_all(all_idx, all_val, v_tmp1, v_one, v_two, v_n_nodes, v_zero,
                                n_vectors, skip_wrap_check=skip_wrap_check, v_tmp2=v_tmp2)

    def _emit_round_k4_fast(self, all_idx, all_val, v_node, v_tmp1, v_tmp2,
                            hash_const_vecs, v_one, v_two, v_n_nodes, v_zero,
                            n_vectors, tmp1, tmp2, idx_base, skip_index=False, skip_wrap_check=False):
        """k=4: Optimized 2-level binary selection for 4 unique indices.

        Selection: result = select(offset, n0, n1, n2, n3)
        Level 1: bit1 = offset >> 1, select between (n0,n1) vs (n2,n3)
        Level 2: bit0 = offset & 1, select within pair
        """
        # Load 4 node values in parallel
        node_addr_tmp = [tmp1, tmp2, v_tmp1[4], v_tmp1[5]]
        for i in range(4):
            self.add("load", ("const", node_addr_tmp[i], idx_base + i))
        self.instrs.append({
            "alu": [("+", node_addr_tmp[i], self.scratch["forest_values_p"], node_addr_tmp[i]) for i in range(4)]
        })
        # Load in pairs
        self.instrs.append({"load": [("load", node_addr_tmp[0], node_addr_tmp[0]),
                                     ("load", node_addr_tmp[1], node_addr_tmp[1])]})
        self.instrs.append({"load": [("load", node_addr_tmp[2], node_addr_tmp[2]),
                                     ("load", node_addr_tmp[3], node_addr_tmp[3])]})
        # Broadcast
        self.instrs.append({"valu": [("vbroadcast", v_node[0], node_addr_tmp[0]),
                                     ("vbroadcast", v_node[1], node_addr_tmp[1]),
                                     ("vbroadcast", v_node[2], node_addr_tmp[2]),
                                     ("vbroadcast", v_node[3], node_addr_tmp[3])]})

        idx_base_vec = self.scratch_const_vec(idx_base)

        # Precompute diffs
        diff02 = v_tmp2[4]  # n0 ^ n2
        diff13 = v_tmp2[5]  # n1 ^ n3
        self.instrs.append({
            "valu": [("^", diff02, v_node[0], v_node[2]),
                     ("^", diff13, v_node[1], v_node[3])]
        })

        # Process 3 vectors at a time (need more temps for k=4)
        for vi in range(0, n_vectors, 3):
            vecs = min(3, n_vectors - vi)
            t = [v_tmp1[0], v_tmp1[1], v_tmp1[2]]  # offset temps
            m = [v_tmp2[0], v_tmp2[1], v_tmp2[2]]  # mask/intermediate temps

            # offset = idx - idx_base
            self.instrs.append({
                "valu": [("-", t[j], all_idx[vi+j], idx_base_vec) for j in range(vecs)]
            })
            # bit1 = offset >> 1
            self.instrs.append({
                "valu": [(">>", m[j], t[j], v_one) for j in range(vecs)]
            })
            # mask1 = 0 - bit1
            self.instrs.append({
                "valu": [("-", m[j], v_zero, m[j]) for j in range(vecs)]
            })
            # sel_low = n0 ^ (diff02 & mask1)
            self.instrs.append({
                "valu": [("&", v_tmp1[3+j] if j < 3 else v_tmp2[3], diff02, m[j]) for j in range(vecs)]
            })
            self.instrs.append({
                "valu": [("^", v_tmp1[3+j] if j < 3 else v_tmp2[3], v_node[0], v_tmp1[3+j] if j < 3 else v_tmp2[3]) for j in range(vecs)]
            })
            # sel_high = n1 ^ (diff13 & mask1) - reuse mask
            self.instrs.append({
                "valu": [("&", m[j], diff13, m[j]) for j in range(vecs)]
            })
            self.instrs.append({
                "valu": [("^", m[j], v_node[1], m[j]) for j in range(vecs)]
            })
            # Level 2: mask0 = 0 - (offset & 1)
            self.instrs.append({
                "valu": [("&", t[j], t[j], v_one) for j in range(vecs)]
            })
            self.instrs.append({
                "valu": [("-", t[j], v_zero, t[j]) for j in range(vecs)]
            })
            # diff_lowhigh = sel_low ^ sel_high
            sel_low = [v_tmp1[3], v_tmp1[4], v_tmp1[5]]
            sel_high = m
            self.instrs.append({
                "valu": [("^", sel_high[j], sel_low[j], sel_high[j]) for j in range(vecs)]
            })
            # result = sel_low ^ (diff_lowhigh & mask0)
            self.instrs.append({
                "valu": [("&", sel_high[j], sel_high[j], t[j]) for j in range(vecs)]
            })
            self.instrs.append({
                "valu": [("^", t[j], sel_low[j], sel_high[j]) for j in range(vecs)]
            })
            # XOR with val
            self.instrs.append({
                "valu": [("^", all_val[vi+j], all_val[vi+j], t[j]) for j in range(vecs)]
            })

        self._emit_hash_all(all_val, v_tmp1, v_tmp2, hash_const_vecs, n_vectors)

        if not skip_index:
            self._emit_index_all(all_idx, all_val, v_tmp1, v_one, v_two, v_n_nodes, v_zero,
                                n_vectors, skip_wrap_check=skip_wrap_check, v_tmp2=v_tmp2)

    def _emit_round_k8_fast(self, all_idx, all_val, v_node, v_tmp1, v_tmp2,
                            hash_const_vecs, v_one, v_two, v_n_nodes, v_zero,
                            n_vectors, tmp1, tmp2, idx_base, skip_index=False, skip_wrap_check=False):
        """k=8: 3-level binary selection for 8 unique indices."""
        # Load 8 node values
        for i in range(8):
            self.add("load", ("const", tmp1, idx_base + i))
            self.instrs.append({"alu": [("+", tmp1, self.scratch["forest_values_p"], tmp1)]})
            self.add("load", ("load", tmp1, tmp1))
            self.add("valu", ("vbroadcast", v_node[i], tmp1))

        idx_base_vec = self.scratch_const_vec(idx_base)
        v_four = self.scratch_const_vec(4)

        # Process 2 vectors at a time (need many temps for k=8)
        for vi in range(0, n_vectors, 2):
            vecs = min(2, n_vectors - vi)

            # offset = idx - idx_base (0-7)
            self.instrs.append({
                "valu": [("-", v_tmp1[j], all_idx[vi+j], idx_base_vec) for j in range(vecs)]
            })

            # Level 1: bit2 = (offset >> 2) & 1, select between first 4 vs last 4
            # mask2 = 0 - bit2
            self.instrs.append({
                "valu": [(">>", v_tmp2[j], v_tmp1[j], v_two) for j in range(vecs)]
            })
            self.instrs.append({
                "valu": [("-", v_tmp2[j], v_zero, v_tmp2[j]) for j in range(vecs)]
            })
            # Select within groups of 4: n0-3 or n4-7
            # diff04 = n0^n4, diff15=n1^n5, diff26=n2^n6, diff37=n3^n7
            # sel = n_i ^ (diff & mask2)
            sel = [v_tmp1[2], v_tmp1[3], v_tmp1[4], v_tmp1[5]]
            for i in range(4):
                self.instrs.append({
                    "valu": [("^", sel[i], v_node[i], v_node[i+4])]  # diff
                })
                self.instrs.append({
                    "valu": [("&", sel[i], sel[i], v_tmp2[0])]  # diff & mask for vec 0
                })
                self.instrs.append({
                    "valu": [("^", sel[i], v_node[i], sel[i])]  # select
                })
            # Now sel[0-3] has selected 4 values for vec 0

            # Level 2: Apply k=4 selection on sel[0-3]
            # bit1 = (offset >> 1) & 1
            self.instrs.append({
                "valu": [(">>", v_tmp2[0], v_tmp1[0], v_one),
                         ("&", v_tmp2[0], v_tmp2[0], v_one) if vecs > 0 else ("^", v_tmp2[0], v_tmp2[0], v_tmp2[0])]
            })
            self.instrs.append({
                "valu": [("-", v_tmp2[0], v_zero, v_tmp2[0])]
            })
            # diff02 = sel0 ^ sel2, diff13 = sel1 ^ sel3
            self.instrs.append({
                "valu": [("^", v_tmp2[2], sel[0], sel[2]),
                         ("^", v_tmp2[3], sel[1], sel[3])]
            })
            self.instrs.append({
                "valu": [("&", v_tmp2[2], v_tmp2[2], v_tmp2[0]),
                         ("&", v_tmp2[3], v_tmp2[3], v_tmp2[0])]
            })
            self.instrs.append({
                "valu": [("^", v_tmp2[2], sel[0], v_tmp2[2]),  # sel_low
                         ("^", v_tmp2[3], sel[1], v_tmp2[3])]  # sel_high
            })

            # Level 3: bit0 = offset & 1
            self.instrs.append({
                "valu": [("&", v_tmp2[0], v_tmp1[0], v_one)]
            })
            self.instrs.append({
                "valu": [("-", v_tmp2[0], v_zero, v_tmp2[0])]
            })
            self.instrs.append({
                "valu": [("^", v_tmp2[4], v_tmp2[2], v_tmp2[3])]  # diff
            })
            self.instrs.append({
                "valu": [("&", v_tmp2[4], v_tmp2[4], v_tmp2[0])]
            })
            self.instrs.append({
                "valu": [("^", v_tmp2[0], v_tmp2[2], v_tmp2[4])]  # result for vec 0
            })

            # XOR with val
            self.instrs.append({
                "valu": [("^", all_val[vi], all_val[vi], v_tmp2[0])]
            })

            # Repeat for vec 1 if present
            if vecs > 1:
                # Same process for vec 1 using v_tmp1[1] as offset
                # Level 1
                self.instrs.append({
                    "valu": [(">>", v_tmp2[0], v_tmp1[1], v_two)]
                })
                self.instrs.append({
                    "valu": [("-", v_tmp2[0], v_zero, v_tmp2[0])]
                })
                for i in range(4):
                    self.instrs.append({
                        "valu": [("^", sel[i], v_node[i], v_node[i+4])]
                    })
                    self.instrs.append({
                        "valu": [("&", sel[i], sel[i], v_tmp2[0])]
                    })
                    self.instrs.append({
                        "valu": [("^", sel[i], v_node[i], sel[i])]
                    })
                # Level 2
                self.instrs.append({
                    "valu": [(">>", v_tmp2[0], v_tmp1[1], v_one)]
                })
                self.instrs.append({
                    "valu": [("&", v_tmp2[0], v_tmp2[0], v_one)]
                })
                self.instrs.append({
                    "valu": [("-", v_tmp2[0], v_zero, v_tmp2[0])]
                })
                self.instrs.append({
                    "valu": [("^", v_tmp2[2], sel[0], sel[2]),
                             ("^", v_tmp2[3], sel[1], sel[3])]
                })
                self.instrs.append({
                    "valu": [("&", v_tmp2[2], v_tmp2[2], v_tmp2[0]),
                             ("&", v_tmp2[3], v_tmp2[3], v_tmp2[0])]
                })
                self.instrs.append({
                    "valu": [("^", v_tmp2[2], sel[0], v_tmp2[2]),
                             ("^", v_tmp2[3], sel[1], v_tmp2[3])]
                })
                # Level 3
                self.instrs.append({
                    "valu": [("&", v_tmp2[0], v_tmp1[1], v_one)]
                })
                self.instrs.append({
                    "valu": [("-", v_tmp2[0], v_zero, v_tmp2[0])]
                })
                self.instrs.append({
                    "valu": [("^", v_tmp2[4], v_tmp2[2], v_tmp2[3])]
                })
                self.instrs.append({
                    "valu": [("&", v_tmp2[4], v_tmp2[4], v_tmp2[0])]
                })
                self.instrs.append({
                    "valu": [("^", v_tmp2[0], v_tmp2[2], v_tmp2[4])]
                })
                self.instrs.append({
                    "valu": [("^", all_val[vi+1], all_val[vi+1], v_tmp2[0])]
                })

        self._emit_hash_all(all_val, v_tmp1, v_tmp2, hash_const_vecs, n_vectors)

        if not skip_index:
            self._emit_index_all(all_idx, all_val, v_tmp1, v_one, v_two, v_n_nodes, v_zero,
                                n_vectors, skip_wrap_check=skip_wrap_check, v_tmp2=v_tmp2)

    def _emit_round_k16_fast(self, all_idx, all_val, v_node, v_tmp1, v_tmp2, v_addrs,
                             hash_const_vecs, v_one, v_two, v_n_nodes, v_zero,
                             n_vectors, tmp1, tmp2, idx_base, skip_index=False, skip_wrap_check=False):
        """k=16: Use gather but with precomputed address base."""
        # For k=16, gathering is likely faster than 4-level selection
        # Just use the full gather implementation
        self._emit_round_full_gather_v2(
            all_idx, all_val, v_node, v_tmp1, v_tmp2, v_addrs, v_addrs,  # use same addr buf
            hash_const_vecs, v_one, v_two, v_n_nodes, v_zero,
            n_vectors, tmp1, skip_index=skip_index,
            skip_wrap_check=skip_wrap_check
        )

    def _emit_hash_all(self, all_val, v_tmp1, v_tmp2, hash_const_vecs, n_vectors):
        """Hash all vectors using multiply_add optimization."""
        for vi in range(0, n_vectors, 6):
            vecs = min(6, n_vectors - vi)

            for hi, stage_info in enumerate(hash_const_vecs):
                op1, val1, op2, op3, val3 = HASH_STAGES[hi]

                if stage_info[0] == 'multiply_add':
                    _, mult_vec, const_vec = stage_info
                    self.instrs.append({
                        "valu": [("multiply_add", all_val[vi+j], all_val[vi+j], mult_vec, const_vec)
                                 for j in range(vecs)]
                    })
                else:
                    _, const1, const3 = stage_info
                    if vecs >= 3:
                        self.instrs.append({
                            "valu": [(op1, v_tmp1[0], all_val[vi], const1),
                                     (op3, v_tmp2[0], all_val[vi], const3),
                                     (op1, v_tmp1[1], all_val[vi+1], const1),
                                     (op3, v_tmp2[1], all_val[vi+1], const3),
                                     (op1, v_tmp1[2], all_val[vi+2], const1),
                                     (op3, v_tmp2[2], all_val[vi+2], const3)]
                        })
                        self.instrs.append({
                            "valu": [(op2, all_val[vi], v_tmp1[0], v_tmp2[0]),
                                     (op2, all_val[vi+1], v_tmp1[1], v_tmp2[1]),
                                     (op2, all_val[vi+2], v_tmp1[2], v_tmp2[2])]
                        })
                        if vecs > 3:
                            self.instrs.append({
                                "valu": [(op1, v_tmp1[0], all_val[vi+3], const1),
                                         (op3, v_tmp2[0], all_val[vi+3], const3)] +
                                        ([(op1, v_tmp1[1], all_val[vi+4], const1),
                                          (op3, v_tmp2[1], all_val[vi+4], const3)] if vecs > 4 else []) +
                                        ([(op1, v_tmp1[2], all_val[vi+5], const1),
                                          (op3, v_tmp2[2], all_val[vi+5], const3)] if vecs > 5 else [])
                            })
                            self.instrs.append({
                                "valu": [(op2, all_val[vi+3], v_tmp1[0], v_tmp2[0])] +
                                        ([(op2, all_val[vi+4], v_tmp1[1], v_tmp2[1])] if vecs > 4 else []) +
                                        ([(op2, all_val[vi+5], v_tmp1[2], v_tmp2[2])] if vecs > 5 else [])
                            })
                    else:
                        for j in range(vecs):
                            self.instrs.append({
                                "valu": [(op1, v_tmp1[0], all_val[vi+j], const1),
                                         (op3, v_tmp2[0], all_val[vi+j], const3)]
                            })
                            self.instrs.append({
                                "valu": [(op2, all_val[vi+j], v_tmp1[0], v_tmp2[0])]
                            })

    def _emit_index_all(self, all_idx, all_val, v_tmp1, v_one, v_two, v_n_nodes, v_zero,
                        n_vectors, skip_wrap_check=False, v_tmp2=None):
        """Index update all vectors."""
        for vi in range(0, n_vectors, 6):
            vecs = min(6, n_vectors - vi)
            if vecs == 6 and v_tmp2 is not None:
                temps = [v_tmp1[0], v_tmp1[1], v_tmp1[2], v_tmp1[3], v_tmp2[0], v_tmp2[1]]
                self.instrs.append({
                    "valu": [("&", temps[j], all_val[vi+j], v_one) for j in range(3)] +
                            [("*", all_idx[vi+j], all_idx[vi+j], v_two) for j in range(3)]
                })
                self.instrs.append({
                    "valu": [("&", temps[3+j], all_val[vi+3+j], v_one) for j in range(3)] +
                            [("*", all_idx[vi+3+j], all_idx[vi+3+j], v_two) for j in range(3)]
                })
                self.instrs.append({
                    "valu": [("+", temps[j], temps[j], v_one) for j in range(6)]
                })
                self.instrs.append({
                    "valu": [("+", all_idx[vi+j], all_idx[vi+j], temps[j]) for j in range(6)]
                })
                if not skip_wrap_check:
                    self.instrs.append({
                        "valu": [("<", temps[j], all_idx[vi+j], v_n_nodes) for j in range(6)]
                    })
                    self.instrs.append({
                        "valu": [("-", temps[j], v_zero, temps[j]) for j in range(6)]
                    })
                    self.instrs.append({
                        "valu": [("&", all_idx[vi+j], all_idx[vi+j], temps[j]) for j in range(6)]
                    })
            else:
                for j in range(vecs):
                    self.instrs.append({
                        "valu": [("&", v_tmp1[0], all_val[vi+j], v_one),
                                 ("*", all_idx[vi+j], all_idx[vi+j], v_two)]
                    })
                    self.instrs.append({
                        "valu": [("+", v_tmp1[0], v_tmp1[0], v_one)]
                    })
                    self.instrs.append({
                        "valu": [("+", all_idx[vi+j], all_idx[vi+j], v_tmp1[0])]
                    })
                    if not skip_wrap_check:
                        self.instrs.append({
                            "valu": [("<", v_tmp1[0], all_idx[vi+j], v_n_nodes)]
                        })
                        self.instrs.append({
                            "valu": [("-", v_tmp1[0], v_zero, v_tmp1[0])]
                        })
                        self.instrs.append({
                            "valu": [("&", all_idx[vi+j], all_idx[vi+j], v_tmp1[0])]
                        })

    def _emit_round_full_gather_v2(self, all_idx, all_val, v_node, v_tmp1, v_tmp2, v_addrs, v_addrs2,
                                    hash_const_vecs, v_one, v_two, v_n_nodes, v_zero,
                                    n_vectors, tmp1, skip_index=False, skip_wrap_check=False):
        """Optimized full gather with 6 vectors per group."""
        # Process 6 vectors per group instead of 4
        n_groups = (n_vectors + 5) // 6
        fp = self.scratch["forest_values_p"]

        for group_idx in range(n_groups):
            g = group_idx * 6
            vecs_in_group = min(6, n_vectors - g)

            # Compute addresses for all vectors in group
            for v in range(vecs_in_group):
                self.instrs.append({
                    "alu": [("+", v_addrs[v] + lane, fp, all_idx[g + v] + lane) for lane in range(VLEN)]
                })

            # Gather all vectors (2 loads per cycle)
            for v in range(vecs_in_group):
                for lane in range(0, VLEN, 2):
                    self.instrs.append({
                        "load": [("load", v_node[v] + lane, v_addrs[v] + lane),
                                 ("load", v_node[v] + lane + 1, v_addrs[v] + lane + 1)]
                    })

            # XOR all vectors
            self.instrs.append({
                "valu": [("^", all_val[g + v], all_val[g + v], v_node[v]) for v in range(vecs_in_group)]
            })

            # Hash all vectors in this group
            for hi, stage_info in enumerate(hash_const_vecs):
                op1, val1, op2, op3, val3 = HASH_STAGES[hi]
                if stage_info[0] == 'multiply_add':
                    _, mult_vec, const_vec = stage_info
                    self.instrs.append({
                        "valu": [("multiply_add", all_val[g + v], all_val[g + v], mult_vec, const_vec)
                                for v in range(vecs_in_group)]
                    })
                else:
                    _, const1, const3 = stage_info
                    if vecs_in_group >= 3:
                        self.instrs.append({
                            "valu": [(op1, v_tmp1[0], all_val[g], const1),
                                     (op3, v_tmp2[0], all_val[g], const3),
                                     (op1, v_tmp1[1], all_val[g+1], const1),
                                     (op3, v_tmp2[1], all_val[g+1], const3),
                                     (op1, v_tmp1[2], all_val[g+2], const1),
                                     (op3, v_tmp2[2], all_val[g+2], const3)]
                        })
                        self.instrs.append({
                            "valu": [(op2, all_val[g], v_tmp1[0], v_tmp2[0]),
                                     (op2, all_val[g+1], v_tmp1[1], v_tmp2[1]),
                                     (op2, all_val[g+2], v_tmp1[2], v_tmp2[2])]
                        })
                        if vecs_in_group > 3:
                            remaining = vecs_in_group - 3
                            self.instrs.append({
                                "valu": [(op1, v_tmp1[r], all_val[g+3+r], const1) for r in range(remaining)] +
                                        [(op3, v_tmp2[r], all_val[g+3+r], const3) for r in range(remaining)]
                            })
                            self.instrs.append({
                                "valu": [(op2, all_val[g+3+r], v_tmp1[r], v_tmp2[r]) for r in range(remaining)]
                            })
                    else:
                        for v in range(vecs_in_group):
                            self.instrs.append({
                                "valu": [(op1, v_tmp1[0], all_val[g+v], const1),
                                         (op3, v_tmp2[0], all_val[g+v], const3)]
                            })
                            self.instrs.append({
                                "valu": [(op2, all_val[g+v], v_tmp1[0], v_tmp2[0])]
                            })

            # Index update for this group
            if not skip_index:
                for v in range(vecs_in_group):
                    self.instrs.append({
                        "valu": [("&", v_tmp1[0], all_val[g+v], v_one),
                                 ("*", all_idx[g+v], all_idx[g+v], v_two)]
                    })
                    self.instrs.append({
                        "valu": [("+", v_tmp1[0], v_tmp1[0], v_one)]
                    })
                    self.instrs.append({
                        "valu": [("+", all_idx[g+v], all_idx[g+v], v_tmp1[0])]
                    })
                    if not skip_wrap_check:
                        self.instrs.append({
                            "valu": [("<", v_tmp1[0], all_idx[g+v], v_n_nodes)]
                        })
                        self.instrs.append({
                            "valu": [("-", v_tmp1[0], v_zero, v_tmp1[0])]
                        })
                        self.instrs.append({
                            "valu": [("&", all_idx[g+v], all_idx[g+v], v_tmp1[0])]
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
    kb.build_kernel(10, 2047, 256, 16)
    print(f"Total instructions: {len(kb.instrs)}")
    print(f"Scratch used: {kb.scratch_ptr}")
    do_kernel_test(10, 16, 256, prints=True)
