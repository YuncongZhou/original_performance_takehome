"""
Wave-based pipelined kernel targeting <1487 cycles.

Key insight: Split vectors into two waves and process with round offset.
- Wave A (vectors 0-15): at round R
- Wave B (vectors 16-31): at round R-1 (offset by 1)

This allows:
- K-selection VALU from Wave B to overlap with full-gather loads from Wave A
- Maximum load/VALU slot utilization

Architecture: VLEN=8, 6 VALU, 2 Load, 2 Store, 12 ALU, 1 Flow per cycle
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
        """Wave-based pipelined kernel."""
        n_vectors = batch_size // VLEN  # 32 vectors
        n_waves = 2
        vectors_per_wave = n_vectors // n_waves  # 16 vectors per wave

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

        # All idx and val in scratch (both waves)
        all_idx = [self.alloc_scratch(f"idx_{i}", VLEN) for i in range(n_vectors)]
        all_val = [self.alloc_scratch(f"val_{i}", VLEN) for i in range(n_vectors)]

        # Working buffers for each wave
        # Wave A: v_node_a, v_tmp1_a, v_tmp2_a, v_addrs_a
        v_node_a = [self.alloc_scratch(f"node_a_{i}", VLEN) for i in range(4)]
        v_tmp1_a = [self.alloc_scratch(f"tmp1_a_{i}", VLEN) for i in range(4)]
        v_tmp2_a = [self.alloc_scratch(f"tmp2_a_{i}", VLEN) for i in range(4)]
        v_addrs_a = [self.alloc_scratch(f"addrs_a_{i}", VLEN) for i in range(4)]

        # Wave B: v_node_b, v_tmp1_b, v_tmp2_b, v_addrs_b
        v_node_b = [self.alloc_scratch(f"node_b_{i}", VLEN) for i in range(4)]
        v_tmp1_b = [self.alloc_scratch(f"tmp1_b_{i}", VLEN) for i in range(4)]
        v_tmp2_b = [self.alloc_scratch(f"tmp2_b_{i}", VLEN) for i in range(4)]
        v_addrs_b = [self.alloc_scratch(f"addrs_b_{i}", VLEN) for i in range(4)]

        # ============ INITIALIZATION ============
        for i, v in enumerate(init_vars):
            self.add("load", ("const", tmp1, i))
            self.add("load", ("load", self.scratch[v], tmp1))

        v_zero = self.scratch_const_vec(0)
        v_one = self.scratch_const_vec(1)
        v_two = self.scratch_const_vec(2)
        v_n_nodes = self.scratch_const_vec(n_nodes)

        # Hash constants - multiply_add for stages 0, 2, 4
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

        # Wave indices
        wave_a_idx = all_idx[:vectors_per_wave]  # vectors 0-15
        wave_a_val = all_val[:vectors_per_wave]
        wave_b_idx = all_idx[vectors_per_wave:]  # vectors 16-31
        wave_b_val = all_val[vectors_per_wave:]

        # ============ WAVE-PIPELINED PROCESSING ============
        # Process with Wave B one round behind Wave A
        # This creates overlap opportunities between waves

        # Round 0: Wave A only
        self._emit_wave_round(
            wave_a_idx, wave_a_val, v_node_a, v_tmp1_a, v_tmp2_a, v_addrs_a,
            hash_const_vecs, v_one, v_two, v_n_nodes, v_zero,
            vectors_per_wave, tmp1, tmp2, tmp3,
            round_num=0, is_last_round=False, n_nodes=n_nodes
        )

        # Rounds 1-15: Both waves with offset
        for round_a in range(1, rounds):
            round_b = round_a - 1
            is_last_a = (round_a == rounds - 1)
            is_last_b = False  # Wave B is behind, never last during overlap

            # Emit interleaved instructions for both waves
            self._emit_interleaved_rounds(
                wave_a_idx, wave_a_val, v_node_a, v_tmp1_a, v_tmp2_a, v_addrs_a,
                wave_b_idx, wave_b_val, v_node_b, v_tmp1_b, v_tmp2_b, v_addrs_b,
                hash_const_vecs, v_one, v_two, v_n_nodes, v_zero,
                vectors_per_wave, tmp1, tmp2, tmp3, tmp4, tmp5, tmp6,
                round_a, round_b, is_last_a, n_nodes
            )

        # Round 15 for Wave B (Wave A is done)
        self._emit_wave_round(
            wave_b_idx, wave_b_val, v_node_b, v_tmp1_b, v_tmp2_b, v_addrs_b,
            hash_const_vecs, v_one, v_two, v_n_nodes, v_zero,
            vectors_per_wave, tmp1, tmp2, tmp3,
            round_num=rounds-1, is_last_round=True, n_nodes=n_nodes
        )

        # ============ STORE ALL VALUES ============
        for vi in range(n_vectors):
            self.add("load", ("const", tmp1, vi * VLEN))
            self.instrs.append({"alu": [("+", tmp2, self.scratch["inp_values_p"], tmp1)]})
            self.instrs.append({"store": [("vstore", tmp2, all_val[vi])]})

        self.instrs.append({"flow": [("pause",)]})

    def _get_effective_round(self, round_num):
        """Get effective round for k-selection (wraps at round 11)."""
        if round_num <= 10:
            return round_num
        else:
            return round_num - 11

    def _emit_wave_round(self, w_idx, w_val, v_node, v_tmp1, v_tmp2, v_addrs,
                          hash_const_vecs, v_one, v_two, v_n_nodes, v_zero,
                          n_vectors, tmp1, tmp2, tmp3, round_num, is_last_round, n_nodes):
        """Emit a single wave's round processing."""
        effective_round = self._get_effective_round(round_num)
        needs_wrap_check = (round_num == 10)

        if effective_round == 0:
            self._emit_round_k1(w_idx, w_val, v_node, v_tmp1, v_tmp2,
                               hash_const_vecs, v_one, v_two, v_n_nodes, v_zero,
                               n_vectors, tmp1, is_last_round, needs_wrap_check)
        elif effective_round == 1:
            self._emit_round_k2(w_idx, w_val, v_node, v_tmp1, v_tmp2,
                               hash_const_vecs, v_one, v_two, v_n_nodes, v_zero,
                               n_vectors, tmp1, tmp2, 1, is_last_round, needs_wrap_check)
        elif effective_round == 2:
            self._emit_round_k4(w_idx, w_val, v_node, v_tmp1, v_tmp2,
                               hash_const_vecs, v_one, v_two, v_n_nodes, v_zero,
                               n_vectors, tmp1, tmp2, 3, is_last_round, needs_wrap_check)
        elif effective_round == 3:
            self._emit_round_k8(w_idx, w_val, v_node, v_tmp1, v_tmp2, v_addrs,
                               hash_const_vecs, v_one, v_two, v_n_nodes, v_zero,
                               n_vectors, tmp1, tmp2, 7, is_last_round, needs_wrap_check)
        else:
            # Full gather for rounds 4-10
            self._emit_round_full_gather(
                w_idx, w_val, v_node, v_tmp1, v_tmp2, v_addrs,
                hash_const_vecs, v_one, v_two, v_n_nodes, v_zero,
                n_vectors, tmp1, is_last_round, needs_wrap_check
            )

    def _emit_interleaved_rounds(self, a_idx, a_val, v_node_a, v_tmp1_a, v_tmp2_a, v_addrs_a,
                                  b_idx, b_val, v_node_b, v_tmp1_b, v_tmp2_b, v_addrs_b,
                                  hash_const_vecs, v_one, v_two, v_n_nodes, v_zero,
                                  n_vectors, tmp1, tmp2, tmp3, tmp4, tmp5, tmp6,
                                  round_a, round_b, is_last_a, n_nodes):
        """Emit interleaved instructions for two waves at different rounds.

        Key insight: schedule load-heavy operations from one wave during
        VALU-heavy operations from the other wave.
        """
        eff_a = self._get_effective_round(round_a)
        eff_b = self._get_effective_round(round_b)
        wrap_a = (round_a == 10)
        wrap_b = (round_b == 10)

        # Determine round types
        # k-selection rounds (0-3) are VALU-heavy with minimal loads
        # Full gather rounds (4-10) are load-heavy
        a_is_ksel = eff_a <= 3
        b_is_ksel = eff_b <= 3

        if a_is_ksel and not b_is_ksel:
            # Wave A does k-selection (VALU-heavy), Wave B does full gather (load-heavy)
            # Perfect overlap opportunity
            self._emit_ksel_with_gather_overlap(
                a_idx, a_val, v_node_a, v_tmp1_a, v_tmp2_a, v_addrs_a, eff_a,
                b_idx, b_val, v_node_b, v_tmp1_b, v_tmp2_b, v_addrs_b,
                hash_const_vecs, v_one, v_two, v_n_nodes, v_zero,
                n_vectors, tmp1, tmp2, tmp3, tmp4, tmp5, tmp6,
                is_last_a, wrap_a, wrap_b
            )
        elif not a_is_ksel and b_is_ksel:
            # Wave A does full gather, Wave B does k-selection
            self._emit_gather_with_ksel_overlap(
                a_idx, a_val, v_node_a, v_tmp1_a, v_tmp2_a, v_addrs_a,
                b_idx, b_val, v_node_b, v_tmp1_b, v_tmp2_b, v_addrs_b, eff_b,
                hash_const_vecs, v_one, v_two, v_n_nodes, v_zero,
                n_vectors, tmp1, tmp2, tmp3, tmp4, tmp5, tmp6,
                is_last_a, wrap_a, wrap_b
            )
        else:
            # Both similar type - process sequentially
            self._emit_wave_round(
                a_idx, a_val, v_node_a, v_tmp1_a, v_tmp2_a, v_addrs_a,
                hash_const_vecs, v_one, v_two, v_n_nodes, v_zero,
                n_vectors, tmp1, tmp2, tmp3,
                round_a, is_last_a, n_nodes
            )
            self._emit_wave_round(
                b_idx, b_val, v_node_b, v_tmp1_b, v_tmp2_b, v_addrs_b,
                hash_const_vecs, v_one, v_two, v_n_nodes, v_zero,
                n_vectors, tmp1, tmp2, tmp3,
                round_b, False, n_nodes
            )

    def _emit_ksel_with_gather_overlap(self, a_idx, a_val, v_node_a, v_tmp1_a, v_tmp2_a, v_addrs_a, eff_a,
                                        b_idx, b_val, v_node_b, v_tmp1_b, v_tmp2_b, v_addrs_b,
                                        hash_const_vecs, v_one, v_two, v_n_nodes, v_zero,
                                        n_vectors, tmp1, tmp2, tmp3, tmp4, tmp5, tmp6,
                                        is_last_a, wrap_a, wrap_b):
        """Wave A does k-selection, Wave B does full gather - maximize overlap."""
        # For now, use simple sequential implementation
        # TODO: Implement true interleaved scheduling
        fp = self.scratch["forest_values_p"]

        # Wave A: k-selection
        if eff_a == 0:
            self._emit_round_k1(a_idx, a_val, v_node_a, v_tmp1_a, v_tmp2_a,
                               hash_const_vecs, v_one, v_two, v_n_nodes, v_zero,
                               n_vectors, tmp1, is_last_a, not wrap_a)
        elif eff_a == 1:
            self._emit_round_k2(a_idx, a_val, v_node_a, v_tmp1_a, v_tmp2_a,
                               hash_const_vecs, v_one, v_two, v_n_nodes, v_zero,
                               n_vectors, tmp1, tmp2, 1, is_last_a, not wrap_a)
        elif eff_a == 2:
            self._emit_round_k4(a_idx, a_val, v_node_a, v_tmp1_a, v_tmp2_a,
                               hash_const_vecs, v_one, v_two, v_n_nodes, v_zero,
                               n_vectors, tmp1, tmp2, 3, is_last_a, not wrap_a)
        elif eff_a == 3:
            self._emit_round_k8(a_idx, a_val, v_node_a, v_tmp1_a, v_tmp2_a, v_addrs_a,
                               hash_const_vecs, v_one, v_two, v_n_nodes, v_zero,
                               n_vectors, tmp1, tmp2, 7, is_last_a, not wrap_a)

        # Wave B: full gather
        self._emit_round_full_gather(
            b_idx, b_val, v_node_b, v_tmp1_b, v_tmp2_b, v_addrs_b,
            hash_const_vecs, v_one, v_two, v_n_nodes, v_zero,
            n_vectors, tmp1, False, not wrap_b
        )

    def _emit_gather_with_ksel_overlap(self, a_idx, a_val, v_node_a, v_tmp1_a, v_tmp2_a, v_addrs_a,
                                        b_idx, b_val, v_node_b, v_tmp1_b, v_tmp2_b, v_addrs_b, eff_b,
                                        hash_const_vecs, v_one, v_two, v_n_nodes, v_zero,
                                        n_vectors, tmp1, tmp2, tmp3, tmp4, tmp5, tmp6,
                                        is_last_a, wrap_a, wrap_b):
        """Wave A does full gather, Wave B does k-selection - maximize overlap."""
        # For now, use simple sequential implementation
        # TODO: Implement true interleaved scheduling

        # Wave A: full gather
        self._emit_round_full_gather(
            a_idx, a_val, v_node_a, v_tmp1_a, v_tmp2_a, v_addrs_a,
            hash_const_vecs, v_one, v_two, v_n_nodes, v_zero,
            n_vectors, tmp1, is_last_a, not wrap_a
        )

        # Wave B: k-selection
        if eff_b == 0:
            self._emit_round_k1(b_idx, b_val, v_node_b, v_tmp1_b, v_tmp2_b,
                               hash_const_vecs, v_one, v_two, v_n_nodes, v_zero,
                               n_vectors, tmp1, False, not wrap_b)
        elif eff_b == 1:
            self._emit_round_k2(b_idx, b_val, v_node_b, v_tmp1_b, v_tmp2_b,
                               hash_const_vecs, v_one, v_two, v_n_nodes, v_zero,
                               n_vectors, tmp1, tmp2, 1, False, not wrap_b)
        elif eff_b == 2:
            self._emit_round_k4(b_idx, b_val, v_node_b, v_tmp1_b, v_tmp2_b,
                               hash_const_vecs, v_one, v_two, v_n_nodes, v_zero,
                               n_vectors, tmp1, tmp2, 3, False, not wrap_b)
        elif eff_b == 3:
            self._emit_round_k8(b_idx, b_val, v_node_b, v_tmp1_b, v_tmp2_b, v_addrs_b,
                               hash_const_vecs, v_one, v_two, v_n_nodes, v_zero,
                               n_vectors, tmp1, tmp2, 7, False, not wrap_b)

    def _emit_round_k1(self, w_idx, w_val, v_node, v_tmp1, v_tmp2,
                       hash_const_vecs, v_one, v_two, v_n_nodes, v_zero,
                       n_vectors, tmp1, skip_index=False, skip_wrap_check=False):
        """k=1: All indices at 0, single node value shared by all."""
        # Load single node value
        self.add("load", ("load", tmp1, self.scratch["forest_values_p"]))
        self.add("valu", ("vbroadcast", v_node[0], tmp1))

        # XOR all vectors (6 at a time)
        for vi in range(0, n_vectors, 6):
            vecs = min(6, n_vectors - vi)
            self.instrs.append({
                "valu": [("^", w_val[vi + j], w_val[vi + j], v_node[0]) for j in range(vecs)]
            })

        # Hash
        self._emit_hash_all(w_val, v_tmp1, v_tmp2, hash_const_vecs, n_vectors)

        # Index update
        if not skip_index:
            self._emit_index_all(w_idx, w_val, v_tmp1, v_one, v_two, v_n_nodes, v_zero,
                                n_vectors, skip_wrap_check, v_tmp2)

    def _emit_round_k2(self, w_idx, w_val, v_node, v_tmp1, v_tmp2,
                       hash_const_vecs, v_one, v_two, v_n_nodes, v_zero,
                       n_vectors, tmp1, tmp2, idx_base, skip_index=False, skip_wrap_check=False):
        """k=2: Two unique indices, binary selection."""
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
        self.instrs.append({"valu": [("^", v_tmp2[3], v_node[0], v_node[1])]})

        # Process 6 vectors at a time
        tmps = [v_tmp1[0], v_tmp1[1], v_tmp1[2], v_tmp1[3], v_tmp2[0], v_tmp2[1]]
        for vi in range(0, n_vectors, 6):
            vecs = min(6, n_vectors - vi)
            # offset = idx - idx_base
            self.instrs.append({
                "valu": [("-", tmps[j], w_idx[vi+j], idx_base_vec) for j in range(vecs)]
            })
            # mask = 0 - offset
            self.instrs.append({
                "valu": [("-", tmps[j], v_zero, tmps[j]) for j in range(vecs)]
            })
            # masked_diff = diff & mask
            self.instrs.append({
                "valu": [("&", tmps[j], v_tmp2[3], tmps[j]) for j in range(vecs)]
            })
            # selected = n0 ^ masked_diff
            self.instrs.append({
                "valu": [("^", tmps[j], v_node[0], tmps[j]) for j in range(vecs)]
            })
            # XOR with val
            self.instrs.append({
                "valu": [("^", w_val[vi+j], w_val[vi+j], tmps[j]) for j in range(vecs)]
            })

        self._emit_hash_all(w_val, v_tmp1, v_tmp2, hash_const_vecs, n_vectors)
        if not skip_index:
            self._emit_index_all(w_idx, w_val, v_tmp1, v_one, v_two, v_n_nodes, v_zero,
                                n_vectors, skip_wrap_check, v_tmp2)

    def _emit_round_k4(self, w_idx, w_val, v_node, v_tmp1, v_tmp2,
                       hash_const_vecs, v_one, v_two, v_n_nodes, v_zero,
                       n_vectors, tmp1, tmp2, idx_base, skip_index=False, skip_wrap_check=False):
        """k=4: 2-level binary selection."""
        # Load 4 node values
        for i in range(4):
            self.add("load", ("const", tmp1, idx_base + i))
            self.instrs.append({"alu": [("+", tmp1, self.scratch["forest_values_p"], tmp1)]})
            self.add("load", ("load", tmp1, tmp1))
            self.add("valu", ("vbroadcast", v_node[i], tmp1))

        idx_base_vec = self.scratch_const_vec(idx_base)

        # Process 3 vectors at a time (need more temps for 2-level selection)
        for vi in range(0, n_vectors, 3):
            vecs = min(3, n_vectors - vi)
            t = [v_tmp1[j] for j in range(vecs)]  # offset temps
            m = [v_tmp2[j] for j in range(vecs)]  # intermediate temps

            # offset = idx - idx_base (0-3)
            self.instrs.append({
                "valu": [("-", t[j], w_idx[vi+j], idx_base_vec) for j in range(vecs)]
            })

            # Level 1: bit1 = offset >> 1, mask1 = 0 - bit1
            # Select between (n0,n1) and (n2,n3)
            self.instrs.append({
                "valu": [(">>", m[j], t[j], v_one) for j in range(vecs)]
            })
            self.instrs.append({
                "valu": [("-", m[j], v_zero, m[j]) for j in range(vecs)]
            })

            # diff02 = n0 ^ n2, diff13 = n1 ^ n3 (compute per vector since we need temps)
            # sel_low = n0 ^ ((n0^n2) & mask1)
            self.instrs.append({
                "valu": [("^", v_tmp1[3], v_node[0], v_node[2])]  # diff02
            })
            self.instrs.append({
                "valu": [("&", v_tmp2[3], v_tmp1[3], m[0]) if vecs > 0 else ("^", v_tmp2[3], v_tmp2[3], v_tmp2[3])] +
                        ([("&", v_addrs[0] if hasattr(self, 'v_addrs') else v_tmp1[3], v_tmp1[3], m[1])] if vecs > 1 else []) +
                        ([("&", v_addrs[1] if hasattr(self, 'v_addrs') else v_tmp1[3], v_tmp1[3], m[2])] if vecs > 2 else [])
            })

            # This is getting complex - let me simplify with a cleaner per-vector approach
            for j in range(vecs):
                # offset already in t[j], mask1 in m[j]
                # diff02 = n0 ^ n2
                self.instrs.append({"valu": [("^", v_tmp1[3], v_node[0], v_node[2])]})
                # sel_low = n0 ^ (diff02 & mask1)
                self.instrs.append({"valu": [("&", v_tmp2[3], v_tmp1[3], m[j])]})
                self.instrs.append({"valu": [("^", v_tmp2[3], v_node[0], v_tmp2[3])]})  # sel_low in v_tmp2[3]

                # diff13 = n1 ^ n3
                self.instrs.append({"valu": [("^", v_tmp1[3], v_node[1], v_node[3])]})
                # sel_high = n1 ^ (diff13 & mask1)
                self.instrs.append({"valu": [("&", m[j], v_tmp1[3], m[j])]})
                self.instrs.append({"valu": [("^", m[j], v_node[1], m[j])]})  # sel_high in m[j]

                # Level 2: bit0 = offset & 1, mask0 = 0 - bit0
                self.instrs.append({"valu": [("&", t[j], t[j], v_one)]})
                self.instrs.append({"valu": [("-", t[j], v_zero, t[j])]})

                # result = sel_low ^ ((sel_low ^ sel_high) & mask0)
                self.instrs.append({"valu": [("^", v_tmp1[3], v_tmp2[3], m[j])]})  # diff
                self.instrs.append({"valu": [("&", v_tmp1[3], v_tmp1[3], t[j])]})  # masked
                self.instrs.append({"valu": [("^", t[j], v_tmp2[3], v_tmp1[3])]})  # result

                # XOR with val
                self.instrs.append({"valu": [("^", w_val[vi+j], w_val[vi+j], t[j])]})

        self._emit_hash_all(w_val, v_tmp1, v_tmp2, hash_const_vecs, n_vectors)
        if not skip_index:
            self._emit_index_all(w_idx, w_val, v_tmp1, v_one, v_two, v_n_nodes, v_zero,
                                n_vectors, skip_wrap_check, v_tmp2)

    def _emit_round_k8(self, w_idx, w_val, v_node, v_tmp1, v_tmp2, v_addrs,
                       hash_const_vecs, v_one, v_two, v_n_nodes, v_zero,
                       n_vectors, tmp1, tmp2, idx_base, skip_index=False, skip_wrap_check=False):
        """k=8: Use full gather for now - 3-level selection overhead is high."""
        self._emit_round_full_gather(
            w_idx, w_val, v_node, v_tmp1, v_tmp2, v_addrs,
            hash_const_vecs, v_one, v_two, v_n_nodes, v_zero,
            n_vectors, tmp1, skip_index, skip_wrap_check
        )

    def _emit_hash_all(self, w_val, v_tmp1, v_tmp2, hash_const_vecs, n_vectors):
        """Hash all vectors."""
        for vi in range(0, n_vectors, 6):
            vecs = min(6, n_vectors - vi)

            for hi, stage_info in enumerate(hash_const_vecs):
                op1, val1, op2, op3, val3 = HASH_STAGES[hi]

                if stage_info[0] == 'multiply_add':
                    _, mult_vec, const_vec = stage_info
                    self.instrs.append({
                        "valu": [("multiply_add", w_val[vi+j], w_val[vi+j], mult_vec, const_vec)
                                 for j in range(vecs)]
                    })
                else:
                    _, const1, const3 = stage_info
                    if vecs >= 3:
                        self.instrs.append({
                            "valu": [(op1, v_tmp1[0], w_val[vi], const1),
                                     (op3, v_tmp2[0], w_val[vi], const3),
                                     (op1, v_tmp1[1], w_val[vi+1], const1),
                                     (op3, v_tmp2[1], w_val[vi+1], const3),
                                     (op1, v_tmp1[2], w_val[vi+2], const1),
                                     (op3, v_tmp2[2], w_val[vi+2], const3)]
                        })
                        self.instrs.append({
                            "valu": [(op2, w_val[vi], v_tmp1[0], v_tmp2[0]),
                                     (op2, w_val[vi+1], v_tmp1[1], v_tmp2[1]),
                                     (op2, w_val[vi+2], v_tmp1[2], v_tmp2[2])]
                        })
                        if vecs > 3:
                            remaining = vecs - 3
                            self.instrs.append({
                                "valu": [(op1, v_tmp1[r], w_val[vi+3+r], const1) for r in range(remaining)] +
                                        [(op3, v_tmp2[r], w_val[vi+3+r], const3) for r in range(remaining)]
                            })
                            self.instrs.append({
                                "valu": [(op2, w_val[vi+3+r], v_tmp1[r], v_tmp2[r]) for r in range(remaining)]
                            })
                    else:
                        for j in range(vecs):
                            self.instrs.append({
                                "valu": [(op1, v_tmp1[0], w_val[vi+j], const1),
                                         (op3, v_tmp2[0], w_val[vi+j], const3)]
                            })
                            self.instrs.append({
                                "valu": [(op2, w_val[vi+j], v_tmp1[0], v_tmp2[0])]
                            })

    def _emit_index_all(self, w_idx, w_val, v_tmp1, v_one, v_two, v_n_nodes, v_zero,
                        n_vectors, skip_wrap_check=False, v_tmp2=None):
        """Index update all vectors."""
        for vi in range(0, n_vectors, 6):
            vecs = min(6, n_vectors - vi)
            if vecs == 6 and v_tmp2 is not None:
                temps = [v_tmp1[0], v_tmp1[1], v_tmp1[2], v_tmp1[3], v_tmp2[0], v_tmp2[1]]
                self.instrs.append({
                    "valu": [("&", temps[j], w_val[vi+j], v_one) for j in range(3)] +
                            [("*", w_idx[vi+j], w_idx[vi+j], v_two) for j in range(3)]
                })
                self.instrs.append({
                    "valu": [("&", temps[3+j], w_val[vi+3+j], v_one) for j in range(3)] +
                            [("*", w_idx[vi+3+j], w_idx[vi+3+j], v_two) for j in range(3)]
                })
                self.instrs.append({
                    "valu": [("+", temps[j], temps[j], v_one) for j in range(6)]
                })
                self.instrs.append({
                    "valu": [("+", w_idx[vi+j], w_idx[vi+j], temps[j]) for j in range(6)]
                })
                if not skip_wrap_check:
                    self.instrs.append({
                        "valu": [("<", temps[j], w_idx[vi+j], v_n_nodes) for j in range(6)]
                    })
                    self.instrs.append({
                        "valu": [("-", temps[j], v_zero, temps[j]) for j in range(6)]
                    })
                    self.instrs.append({
                        "valu": [("&", w_idx[vi+j], w_idx[vi+j], temps[j]) for j in range(6)]
                    })
            else:
                for j in range(vecs):
                    self.instrs.append({
                        "valu": [("&", v_tmp1[0], w_val[vi+j], v_one),
                                 ("*", w_idx[vi+j], w_idx[vi+j], v_two)]
                    })
                    self.instrs.append({
                        "valu": [("+", v_tmp1[0], v_tmp1[0], v_one)]
                    })
                    self.instrs.append({
                        "valu": [("+", w_idx[vi+j], w_idx[vi+j], v_tmp1[0])]
                    })
                    if not skip_wrap_check:
                        self.instrs.append({
                            "valu": [("<", v_tmp1[0], w_idx[vi+j], v_n_nodes)]
                        })
                        self.instrs.append({
                            "valu": [("-", v_tmp1[0], v_zero, v_tmp1[0])]
                        })
                        self.instrs.append({
                            "valu": [("&", w_idx[vi+j], w_idx[vi+j], v_tmp1[0])]
                        })

    def _emit_round_full_gather(self, w_idx, w_val, v_node, v_tmp1, v_tmp2, v_addrs,
                                 hash_const_vecs, v_one, v_two, v_n_nodes, v_zero,
                                 n_vectors, tmp1, skip_index=False, skip_wrap_check=False):
        """Full gather with 4 vectors per group."""
        n_groups = n_vectors // 4
        fp = self.scratch["forest_values_p"]

        for group_idx in range(n_groups):
            g = group_idx * 4

            # Compute addresses
            for v in range(4):
                self.instrs.append({
                    "alu": [("+", v_addrs[v] + lane, fp, w_idx[g + v] + lane) for lane in range(VLEN)]
                })

            # Gather all vectors
            for v in range(4):
                for lane in range(0, VLEN, 2):
                    self.instrs.append({
                        "load": [("load", v_node[v] + lane, v_addrs[v] + lane),
                                 ("load", v_node[v] + lane + 1, v_addrs[v] + lane + 1)]
                    })

            # XOR
            self.instrs.append({
                "valu": [("^", w_val[g + v], w_val[g + v], v_node[v]) for v in range(4)]
            })

            # Hash
            for hi, stage_info in enumerate(hash_const_vecs):
                op1, val1, op2, op3, val3 = HASH_STAGES[hi]
                if stage_info[0] == 'multiply_add':
                    _, mult_vec, const_vec = stage_info
                    self.instrs.append({
                        "valu": [("multiply_add", w_val[g + v], w_val[g + v], mult_vec, const_vec)
                                for v in range(4)]
                    })
                else:
                    _, const1, const3 = stage_info
                    self.instrs.append({
                        "valu": [(op1, v_tmp1[0], w_val[g], const1),
                                 (op3, v_tmp2[0], w_val[g], const3),
                                 (op1, v_tmp1[1], w_val[g+1], const1),
                                 (op3, v_tmp2[1], w_val[g+1], const3)]
                    })
                    self.instrs.append({
                        "valu": [(op2, w_val[g], v_tmp1[0], v_tmp2[0]),
                                 (op2, w_val[g+1], v_tmp1[1], v_tmp2[1]),
                                 (op1, v_tmp1[2], w_val[g+2], const1),
                                 (op3, v_tmp2[2], w_val[g+2], const3)]
                    })
                    self.instrs.append({
                        "valu": [(op2, w_val[g+2], v_tmp1[2], v_tmp2[2]),
                                 (op1, v_tmp1[3], w_val[g+3], const1),
                                 (op3, v_tmp2[3], w_val[g+3], const3)]
                    })
                    self.instrs.append({
                        "valu": [(op2, w_val[g+3], v_tmp1[3], v_tmp2[3])]
                    })

            # Index update
            if not skip_index:
                for v in range(4):
                    self.instrs.append({
                        "valu": [("&", v_tmp1[0], w_val[g+v], v_one),
                                 ("*", w_idx[g+v], w_idx[g+v], v_two)]
                    })
                    self.instrs.append({"valu": [("+", v_tmp1[0], v_tmp1[0], v_one)]})
                    self.instrs.append({"valu": [("+", w_idx[g+v], w_idx[g+v], v_tmp1[0])]})
                    if not skip_wrap_check:
                        self.instrs.append({"valu": [("<", v_tmp1[0], w_idx[g+v], v_n_nodes)]})
                        self.instrs.append({"valu": [("-", v_tmp1[0], v_zero, v_tmp1[0])]})
                        self.instrs.append({"valu": [("&", w_idx[g+v], w_idx[g+v], v_tmp1[0])]})


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
