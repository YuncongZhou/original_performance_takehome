"""
Wave-based kernel with careful interleaving.

Key insight: Process vectors in two waves with round offset.
This allows VALU-heavy rounds (k=1, k=2) to overlap with load-heavy rounds.

The crucial observation is:
- k=1 round (round 0, 11): ~109 VALU instructions, 1 load
- k=2 round (round 1, 12): ~134 VALU instructions, 2 loads
- Full gather rounds: ~128 loads + ~50 VALU per 4-vector group

By processing Wave B one round behind Wave A, we can overlap:
- Wave A's full gather loads with Wave B's k=1/k=2 VALU
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


class WaveKernelBuilder:
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

    def emit(self, instr_dict):
        """Emit an instruction with multiple engines."""
        self.instrs.append(instr_dict)

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

    def build_kernel(self, forest_height: int, n_nodes: int, batch_size: int, rounds: int):
        """Build kernel with wave-based processing."""
        n_vectors = batch_size // VLEN  # 32 vectors
        half = n_vectors // 2  # 16 vectors per wave

        # ============ ALLOCATION ============
        # Separate tmp arrays for each wave to avoid interference
        tmp_a = [self.alloc_scratch(f"tmp_a{i}") for i in range(8)]
        tmp_b = [self.alloc_scratch(f"tmp_b{i}") for i in range(8)]
        tmp_init = [self.alloc_scratch(f"tmp_init{i}") for i in range(4)]  # For init/store phases

        init_vars = ["rounds", "n_nodes", "batch_size", "forest_height",
                     "forest_values_p", "inp_indices_p", "inp_values_p"]
        for v in init_vars:
            self.alloc_scratch(v, 1)

        # Indices and values for all vectors
        all_idx = [self.alloc_scratch(f"idx_{i}", VLEN) for i in range(n_vectors)]
        all_val = [self.alloc_scratch(f"val_{i}", VLEN) for i in range(n_vectors)]

        # Working buffers for each wave
        v_node_a = [self.alloc_scratch(f"node_a_{i}", VLEN) for i in range(4)]
        v_tmp1_a = [self.alloc_scratch(f"tmp1_a_{i}", VLEN) for i in range(4)]
        v_tmp2_a = [self.alloc_scratch(f"tmp2_a_{i}", VLEN) for i in range(4)]
        v_addrs_a = [self.alloc_scratch(f"addrs_a_{i}", VLEN) for i in range(4)]

        v_node_b = [self.alloc_scratch(f"node_b_{i}", VLEN) for i in range(4)]
        v_tmp1_b = [self.alloc_scratch(f"tmp1_b_{i}", VLEN) for i in range(4)]
        v_tmp2_b = [self.alloc_scratch(f"tmp2_b_{i}", VLEN) for i in range(4)]
        v_addrs_b = [self.alloc_scratch(f"addrs_b_{i}", VLEN) for i in range(4)]

        # ============ INITIALIZATION ============
        for i, v in enumerate(init_vars):
            self.add("load", ("const", tmp_init[0], i))
            self.add("load", ("load", self.scratch[v], tmp_init[0]))

        v_zero = self.scratch_const_vec(0)
        v_one = self.scratch_const_vec(1)
        v_two = self.scratch_const_vec(2)
        v_n_nodes = self.scratch_const_vec(n_nodes)

        # Hash constants
        hash_consts = []
        hash_mult = {0: 4097, 2: 33, 4: 9}
        for hi, (op1, val1, op2, op3, val3) in enumerate(HASH_STAGES):
            if hi in hash_mult:
                hash_consts.append(('ma', self.scratch_const_vec(hash_mult[hi]),
                                   self.scratch_const_vec(val1)))
            else:
                hash_consts.append(('reg', self.scratch_const_vec(val1),
                                   self.scratch_const_vec(val3)))

        self.add("flow", ("pause",))

        # ============ LOAD ALL DATA ============
        for vi in range(n_vectors):
            self.add("load", ("const", tmp_init[0], vi * VLEN))
            self.emit({"alu": [("+", tmp_init[1], self.scratch["inp_indices_p"], tmp_init[0]),
                               ("+", tmp_init[2], self.scratch["inp_values_p"], tmp_init[0])]})
            self.emit({"load": [("vload", all_idx[vi], tmp_init[1]),
                                ("vload", all_val[vi], tmp_init[2])]})

        # Wave slices
        idx_a, val_a = all_idx[:half], all_val[:half]
        idx_b, val_b = all_idx[half:], all_val[half:]

        fp = self.scratch["forest_values_p"]

        # ============ MAIN LOOP ============
        # Process waves with Wave B one round behind Wave A

        for round_a in range(rounds + 1):
            round_b = round_a - 1

            # What each wave does this iteration
            do_a = (round_a < rounds)
            do_b = (0 <= round_b < rounds)

            if not do_a and not do_b:
                continue

            eff_a = round_a % 11 if round_a < rounds else -1
            eff_b = round_b % 11 if round_b >= 0 else -1
            wrap_a = (round_a == 10) if do_a else False
            wrap_b = (round_b == 10) if do_b else False
            last_a = (round_a == rounds - 1)
            last_b = (round_b == rounds - 1)

            # Determine round types
            # 0: k=1 (VALU-heavy, 1 load)
            # 1: k=2 (VALU-heavy, 2 loads)
            # 2-10: full gather (load-heavy)
            def round_type(eff):
                if eff < 0:
                    return None
                if eff <= 1:
                    return 'ksel'  # k-selection, VALU-heavy
                return 'gather'   # full gather, load-heavy

            type_a = round_type(eff_a) if do_a else None
            type_b = round_type(eff_b) if do_b else None

            # Emit round processing
            if type_a == 'ksel' and type_b == 'gather':
                # Wave A k-selection VALU can overlap with Wave B gather loads
                self._emit_ksel_gather_overlap(
                    idx_a, val_a, v_node_a, v_tmp1_a, v_tmp2_a, v_addrs_a, eff_a, last_a, wrap_a,
                    idx_b, val_b, v_node_b, v_tmp1_b, v_tmp2_b, v_addrs_b, last_b, wrap_b,
                    hash_consts, v_one, v_two, v_n_nodes, v_zero, half, tmp_a, tmp_b, fp
                )
            elif type_a == 'gather' and type_b == 'ksel':
                # Wave A gather loads with Wave B k-selection VALU
                self._emit_gather_ksel_overlap(
                    idx_a, val_a, v_node_a, v_tmp1_a, v_tmp2_a, v_addrs_a, last_a, wrap_a,
                    idx_b, val_b, v_node_b, v_tmp1_b, v_tmp2_b, v_addrs_b, eff_b, last_b, wrap_b,
                    hash_consts, v_one, v_two, v_n_nodes, v_zero, half, tmp_a, tmp_b, fp
                )
            else:
                # Sequential processing for same-type rounds or single wave
                if do_a:
                    self._emit_round(idx_a, val_a, v_node_a, v_tmp1_a, v_tmp2_a, v_addrs_a,
                                    eff_a, last_a, wrap_a, hash_consts, v_one, v_two, v_n_nodes, v_zero,
                                    half, tmp_a, fp)
                if do_b:
                    self._emit_round(idx_b, val_b, v_node_b, v_tmp1_b, v_tmp2_b, v_addrs_b,
                                    eff_b, last_b, wrap_b, hash_consts, v_one, v_two, v_n_nodes, v_zero,
                                    half, tmp_b, fp)

        # ============ STORE ============
        for vi in range(n_vectors):
            self.add("load", ("const", tmp_init[0], vi * VLEN))
            self.emit({"alu": [("+", tmp_init[1], self.scratch["inp_values_p"], tmp_init[0])]})
            self.emit({"store": [("vstore", tmp_init[1], all_val[vi])]})

        self.emit({"flow": [("pause",)]})

    def _emit_round(self, idx, val, v_node, v_tmp1, v_tmp2, v_addrs,
                    eff_round, is_last, needs_wrap, hash_consts, v_one, v_two, v_n_nodes, v_zero,
                    n_vectors, tmp, fp):
        """Emit a single round for one wave."""
        if eff_round == 0:
            self._emit_k1(idx, val, v_node, v_tmp1, v_tmp2, hash_consts,
                         v_one, v_two, v_n_nodes, v_zero, n_vectors, tmp, fp, is_last, needs_wrap)
        elif eff_round == 1:
            self._emit_k2(idx, val, v_node, v_tmp1, v_tmp2, hash_consts,
                         v_one, v_two, v_n_nodes, v_zero, n_vectors, tmp, fp, is_last, needs_wrap)
        else:
            self._emit_full_gather(idx, val, v_node, v_tmp1, v_tmp2, v_addrs, hash_consts,
                                  v_one, v_two, v_n_nodes, v_zero, n_vectors, tmp, fp, is_last, needs_wrap)

    def _emit_ksel_gather_overlap(self, idx_a, val_a, v_node_a, v_tmp1_a, v_tmp2_a, v_addrs_a, eff_a, last_a, wrap_a,
                                   idx_b, val_b, v_node_b, v_tmp1_b, v_tmp2_b, v_addrs_b, last_b, wrap_b,
                                   hash_consts, v_one, v_two, v_n_nodes, v_zero, n_vectors, tmp_a, tmp_b, fp):
        """Wave A does k-selection, Wave B does full gather - TRUE interleaving for overlap."""
        # Collect instructions from both waves, then merge
        saved_instrs = self.instrs

        # Generate Wave A k-selection instructions
        self.instrs = []
        if eff_a == 0:
            self._emit_k1(idx_a, val_a, v_node_a, v_tmp1_a, v_tmp2_a, hash_consts,
                         v_one, v_two, v_n_nodes, v_zero, n_vectors, tmp_a, fp, last_a, wrap_a)
        else:
            self._emit_k2(idx_a, val_a, v_node_a, v_tmp1_a, v_tmp2_a, hash_consts,
                         v_one, v_two, v_n_nodes, v_zero, n_vectors, tmp_a, fp, last_a, wrap_a)
        wave_a_instrs = self.instrs

        # Generate Wave B full gather instructions
        self.instrs = []
        self._emit_full_gather(idx_b, val_b, v_node_b, v_tmp1_b, v_tmp2_b, v_addrs_b, hash_consts,
                              v_one, v_two, v_n_nodes, v_zero, n_vectors, tmp_b, fp, last_b, wrap_b)
        wave_b_instrs = self.instrs

        # Restore and merge
        self.instrs = saved_instrs
        self._merge_instructions(wave_a_instrs, wave_b_instrs)

    def _emit_gather_ksel_overlap(self, idx_a, val_a, v_node_a, v_tmp1_a, v_tmp2_a, v_addrs_a, last_a, wrap_a,
                                   idx_b, val_b, v_node_b, v_tmp1_b, v_tmp2_b, v_addrs_b, eff_b, last_b, wrap_b,
                                   hash_consts, v_one, v_two, v_n_nodes, v_zero, n_vectors, tmp_a, tmp_b, fp):
        """Wave A does full gather, Wave B does k-selection - TRUE interleaving for overlap."""
        # Collect instructions from both waves, then merge
        saved_instrs = self.instrs

        # Generate Wave A full gather instructions
        self.instrs = []
        self._emit_full_gather(idx_a, val_a, v_node_a, v_tmp1_a, v_tmp2_a, v_addrs_a, hash_consts,
                              v_one, v_two, v_n_nodes, v_zero, n_vectors, tmp_a, fp, last_a, wrap_a)
        wave_a_instrs = self.instrs

        # Generate Wave B k-selection instructions
        self.instrs = []
        if eff_b == 0:
            self._emit_k1(idx_b, val_b, v_node_b, v_tmp1_b, v_tmp2_b, hash_consts,
                         v_one, v_two, v_n_nodes, v_zero, n_vectors, tmp_b, fp, last_b, wrap_b)
        else:
            self._emit_k2(idx_b, val_b, v_node_b, v_tmp1_b, v_tmp2_b, hash_consts,
                         v_one, v_two, v_n_nodes, v_zero, n_vectors, tmp_b, fp, last_b, wrap_b)
        wave_b_instrs = self.instrs

        # Restore and merge
        self.instrs = saved_instrs
        self._merge_instructions(wave_a_instrs, wave_b_instrs)

    def _merge_instructions(self, instrs_a, instrs_b):
        """Merge two instruction streams to maximize overlap.

        Strategy: Pair load-heavy instructions from one stream with VALU-heavy from the other.
        """
        # Separate instructions by type
        def classify(instr):
            has_load = 'load' in instr
            has_valu = 'valu' in instr
            if has_load and not has_valu:
                return 'load'
            elif has_valu and not has_load:
                return 'valu'
            else:
                return 'mixed'

        # Try to merge: pair load-only with valu-only
        i, j = 0, 0
        while i < len(instrs_a) and j < len(instrs_b):
            type_a = classify(instrs_a[i])
            type_b = classify(instrs_b[j])

            if type_a == 'load' and type_b == 'valu':
                # Perfect pair: merge them
                merged = dict(instrs_a[i])
                merged.update(instrs_b[j])
                self.instrs.append(merged)
                i += 1
                j += 1
            elif type_a == 'valu' and type_b == 'load':
                # Perfect pair: merge them
                merged = dict(instrs_b[j])
                merged.update(instrs_a[i])
                self.instrs.append(merged)
                i += 1
                j += 1
            elif type_a == 'load' and type_b == 'load':
                # Both are load-only, emit one and try again
                self.instrs.append(instrs_a[i])
                i += 1
            elif type_a == 'valu' and type_b == 'valu':
                # Both are valu-only, emit one and try again
                self.instrs.append(instrs_a[i])
                i += 1
            else:
                # One or both are mixed, emit both separately
                self.instrs.append(instrs_a[i])
                i += 1

        # Emit remaining
        while i < len(instrs_a):
            self.instrs.append(instrs_a[i])
            i += 1
        while j < len(instrs_b):
            self.instrs.append(instrs_b[j])
            j += 1

    def _emit_k1(self, idx, val, v_node, v_tmp1, v_tmp2, hash_consts,
                 v_one, v_two, v_n_nodes, v_zero, n_vectors, tmp, fp, is_last, needs_wrap):
        """k=1: All at index 0."""
        # Load single node
        self.add("load", ("load", tmp[0], fp))
        self.add("valu", ("vbroadcast", v_node[0], tmp[0]))

        # XOR all vectors
        for vi in range(0, n_vectors, 6):
            vecs = min(6, n_vectors - vi)
            self.emit({"valu": [("^", val[vi+j], val[vi+j], v_node[0]) for j in range(vecs)]})

        # Hash
        self._hash_all(val, v_tmp1, v_tmp2, hash_consts, n_vectors)

        # Index update
        if not is_last:
            self._index_all(idx, val, v_tmp1, v_one, v_two, v_n_nodes, v_zero, n_vectors, not needs_wrap, v_tmp2)

    def _emit_k2(self, idx, val, v_node, v_tmp1, v_tmp2, hash_consts,
                 v_one, v_two, v_n_nodes, v_zero, n_vectors, tmp, fp, is_last, needs_wrap):
        """k=2: Indices 1 and 2."""
        # Load two nodes
        self.add("load", ("const", tmp[0], 1))
        self.add("load", ("const", tmp[1], 2))
        self.emit({"alu": [("+", tmp[0], fp, tmp[0]), ("+", tmp[1], fp, tmp[1])]})
        self.emit({"load": [("load", tmp[0], tmp[0]), ("load", tmp[1], tmp[1])]})
        self.emit({"valu": [("vbroadcast", v_node[0], tmp[0]), ("vbroadcast", v_node[1], tmp[1])]})

        idx_base_vec = self.scratch_const_vec(1)
        self.emit({"valu": [("^", v_tmp2[3], v_node[0], v_node[1])]})  # diff

        tmps = [v_tmp1[0], v_tmp1[1], v_tmp1[2], v_tmp1[3], v_tmp2[0], v_tmp2[1]]
        for vi in range(0, n_vectors, 6):
            vecs = min(6, n_vectors - vi)
            self.emit({"valu": [("-", tmps[j], idx[vi+j], idx_base_vec) for j in range(vecs)]})
            self.emit({"valu": [("-", tmps[j], v_zero, tmps[j]) for j in range(vecs)]})
            self.emit({"valu": [("&", tmps[j], v_tmp2[3], tmps[j]) for j in range(vecs)]})
            self.emit({"valu": [("^", tmps[j], v_node[0], tmps[j]) for j in range(vecs)]})
            self.emit({"valu": [("^", val[vi+j], val[vi+j], tmps[j]) for j in range(vecs)]})

        self._hash_all(val, v_tmp1, v_tmp2, hash_consts, n_vectors)
        if not is_last:
            self._index_all(idx, val, v_tmp1, v_one, v_two, v_n_nodes, v_zero, n_vectors, not needs_wrap, v_tmp2)

    def _emit_full_gather(self, idx, val, v_node, v_tmp1, v_tmp2, v_addrs, hash_consts,
                          v_one, v_two, v_n_nodes, v_zero, n_vectors, tmp, fp, is_last, needs_wrap):
        """Full gather for 16 vectors (4 groups of 4)."""
        n_groups = n_vectors // 4

        for gi in range(n_groups):
            g = gi * 4

            # Compute addresses
            for v in range(4):
                self.emit({"alu": [("+", v_addrs[v] + l, fp, idx[g+v] + l) for l in range(VLEN)]})

            # Gather
            for v in range(4):
                for l in range(0, VLEN, 2):
                    self.emit({"load": [("load", v_node[v] + l, v_addrs[v] + l),
                                        ("load", v_node[v] + l + 1, v_addrs[v] + l + 1)]})

            # XOR
            self.emit({"valu": [("^", val[g+v], val[g+v], v_node[v]) for v in range(4)]})

            # Hash for this group
            for hi, stage in enumerate(hash_consts):
                op1, val1, op2, op3, val3 = HASH_STAGES[hi]
                if stage[0] == 'ma':
                    _, mult, const = stage
                    self.emit({"valu": [("multiply_add", val[g+v], val[g+v], mult, const) for v in range(4)]})
                else:
                    _, c1, c3 = stage
                    # Compute for vectors 0,1
                    self.emit({"valu": [(op1, v_tmp1[v], val[g+v], c1) for v in range(2)] +
                                       [(op3, v_tmp2[v], val[g+v], c3) for v in range(2)]})
                    # Combine vectors 0,1 and compute op1 for vectors 2,3
                    self.emit({"valu": [(op2, val[g+v], v_tmp1[v], v_tmp2[v]) for v in range(2)] +
                                       [(op1, v_tmp1[2+v], val[g+2+v], c1) for v in range(2)]})
                    # Compute op3 for vectors 2,3 (separate instruction to avoid RAW hazard)
                    self.emit({"valu": [(op3, v_tmp2[2+v], val[g+2+v], c3) for v in range(2)]})
                    # Combine vectors 2,3
                    self.emit({"valu": [(op2, val[g+2+v], v_tmp1[2+v], v_tmp2[2+v]) for v in range(2)]})

            # Index for this group
            if not is_last:
                for v in range(4):
                    self.emit({"valu": [("&", v_tmp1[0], val[g+v], v_one),
                                        ("*", idx[g+v], idx[g+v], v_two)]})
                    self.emit({"valu": [("+", v_tmp1[0], v_tmp1[0], v_one)]})
                    self.emit({"valu": [("+", idx[g+v], idx[g+v], v_tmp1[0])]})
                    if needs_wrap:
                        self.emit({"valu": [("<", v_tmp1[0], idx[g+v], v_n_nodes)]})
                        self.emit({"valu": [("-", v_tmp1[0], v_zero, v_tmp1[0])]})
                        self.emit({"valu": [("&", idx[g+v], idx[g+v], v_tmp1[0])]})

    def _hash_all(self, val, v_tmp1, v_tmp2, hash_consts, n_vectors):
        """Hash all vectors."""
        for vi in range(0, n_vectors, 6):
            vecs = min(6, n_vectors - vi)
            for hi, stage in enumerate(hash_consts):
                op1, val1, op2, op3, val3 = HASH_STAGES[hi]
                if stage[0] == 'ma':
                    _, mult, const = stage
                    self.emit({"valu": [("multiply_add", val[vi+j], val[vi+j], mult, const) for j in range(vecs)]})
                else:
                    _, c1, c3 = stage
                    if vecs >= 3:
                        self.emit({"valu": [(op1, v_tmp1[j], val[vi+j], c1) for j in range(3)] +
                                           [(op3, v_tmp2[j], val[vi+j], c3) for j in range(3)]})
                        self.emit({"valu": [(op2, val[vi+j], v_tmp1[j], v_tmp2[j]) for j in range(3)]})
                        if vecs > 3:
                            rem = vecs - 3
                            self.emit({"valu": [(op1, v_tmp1[j], val[vi+3+j], c1) for j in range(rem)] +
                                               [(op3, v_tmp2[j], val[vi+3+j], c3) for j in range(rem)]})
                            self.emit({"valu": [(op2, val[vi+3+j], v_tmp1[j], v_tmp2[j]) for j in range(rem)]})
                    else:
                        for j in range(vecs):
                            self.emit({"valu": [(op1, v_tmp1[0], val[vi+j], c1), (op3, v_tmp2[0], val[vi+j], c3)]})
                            self.emit({"valu": [(op2, val[vi+j], v_tmp1[0], v_tmp2[0])]})

    def _index_all(self, idx, val, v_tmp1, v_one, v_two, v_n_nodes, v_zero, n_vectors, skip_wrap, v_tmp2=None):
        """Index update all vectors."""
        for vi in range(0, n_vectors, 6):
            vecs = min(6, n_vectors - vi)
            if vecs == 6 and v_tmp2:
                temps = [v_tmp1[0], v_tmp1[1], v_tmp1[2], v_tmp1[3], v_tmp2[0], v_tmp2[1]]
                self.emit({"valu": [("&", temps[j], val[vi+j], v_one) for j in range(3)] +
                                   [("*", idx[vi+j], idx[vi+j], v_two) for j in range(3)]})
                self.emit({"valu": [("&", temps[3+j], val[vi+3+j], v_one) for j in range(3)] +
                                   [("*", idx[vi+3+j], idx[vi+3+j], v_two) for j in range(3)]})
                self.emit({"valu": [("+", temps[j], temps[j], v_one) for j in range(6)]})
                self.emit({"valu": [("+", idx[vi+j], idx[vi+j], temps[j]) for j in range(6)]})
                if not skip_wrap:
                    self.emit({"valu": [("<", temps[j], idx[vi+j], v_n_nodes) for j in range(6)]})
                    self.emit({"valu": [("-", temps[j], v_zero, temps[j]) for j in range(6)]})
                    self.emit({"valu": [("&", idx[vi+j], idx[vi+j], temps[j]) for j in range(6)]})
            else:
                for j in range(vecs):
                    self.emit({"valu": [("&", v_tmp1[0], val[vi+j], v_one), ("*", idx[vi+j], idx[vi+j], v_two)]})
                    self.emit({"valu": [("+", v_tmp1[0], v_tmp1[0], v_one)]})
                    self.emit({"valu": [("+", idx[vi+j], idx[vi+j], v_tmp1[0])]})
                    if not skip_wrap:
                        self.emit({"valu": [("<", v_tmp1[0], idx[vi+j], v_n_nodes)]})
                        self.emit({"valu": [("-", v_tmp1[0], v_zero, v_tmp1[0])]})
                        self.emit({"valu": [("&", idx[vi+j], idx[vi+j], v_tmp1[0])]})


# Use the original KernelBuilder for export
class KernelBuilder(WaveKernelBuilder):
    pass


BASELINE = 147734

def do_kernel_test(forest_height: int, rounds: int, batch_size: int,
                   seed: int = 123, trace: bool = False, prints: bool = False):
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
    do_kernel_test(10, 16, 256, prints=True)
