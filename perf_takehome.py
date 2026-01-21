"""
# Anthropic's Original Performance Engineering Take-home (Release version)

Copyright Anthropic PBC 2026. Permission is granted to modify and use, but not
to publish or redistribute your solutions so it's hard to find spoilers.

# Task

- Optimize the kernel (in KernelBuilder.build_kernel) as much as possible in the
  available time, as measured by test_kernel_cycles on a frozen separate copy
  of the simulator.

We recommend you look through problem.py next.
"""

from collections import defaultdict
import random
import unittest

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

    def build(self, slots: list[tuple[Engine, tuple]], vliw: bool = False):
        # Simple slot packing that just uses one slot per instruction bundle
        instrs = []
        for engine, slot in slots:
            instrs.append({engine: [slot]})
        return instrs

    def add(self, engine, slot):
        self.instrs.append({engine: [slot]})

    def alloc_scratch(self, name=None, length=1):
        addr = self.scratch_ptr
        if name is not None:
            self.scratch[name] = addr
            self.scratch_debug[addr] = (name, length)
        self.scratch_ptr += length
        assert self.scratch_ptr <= SCRATCH_SIZE, "Out of scratch space"
        return addr

    def scratch_const(self, val, name=None):
        if val not in self.const_map:
            addr = self.alloc_scratch(name)
            self.add("load", ("const", addr, val))
            self.const_map[val] = addr
        return self.const_map[val]

    def build_hash(self, val_hash_addr, tmp1, tmp2, round, i):
        slots = []

        for hi, (op1, val1, op2, op3, val3) in enumerate(HASH_STAGES):
            slots.append(("alu", (op1, tmp1, val_hash_addr, self.scratch_const(val1))))
            slots.append(("alu", (op3, tmp2, val_hash_addr, self.scratch_const(val3))))
            slots.append(("alu", (op2, val_hash_addr, tmp1, tmp2)))
            slots.append(("debug", ("compare", val_hash_addr, (round, i, "hash_stage", hi))))

        return slots

    def build_vhash(self, val_vec, tmp1_vec, tmp2_vec):
        """Build vectorized hash computation for VLEN elements."""
        slots = []
        for hi, (op1, val1, op2, op3, val3) in enumerate(HASH_STAGES):
            # tmp1 = val op1 const1
            slots.append(("valu", (op1, tmp1_vec, val_vec, self.scratch_const_vec(val1))))
            # tmp2 = val op3 const3
            slots.append(("valu", (op3, tmp2_vec, val_vec, self.scratch_const_vec(val3))))
            # val = tmp1 op2 tmp2
            slots.append(("valu", (op2, val_vec, tmp1_vec, tmp2_vec)))
        return slots

    def scratch_const_vec(self, val, name=None):
        """Allocate a vector constant (broadcast scalar to VLEN elements)."""
        key = ("vec", val)
        if key not in self.const_map:
            addr = self.alloc_scratch(name, VLEN)
            # Use vbroadcast to fill vector with scalar constant
            scalar_addr = self.scratch_const(val)
            self.add("valu", ("vbroadcast", addr, scalar_addr))
            self.const_map[key] = addr
        return self.const_map[key]

    def emit_vliw(self, slots):
        """Pack slots into VLIW instruction bundles respecting slot limits."""
        # Group slots by engine type
        by_engine = defaultdict(list)
        for engine, slot in slots:
            by_engine[engine].append(slot)

        # Build instruction bundles
        while any(by_engine.values()):
            bundle = {}
            for engine in list(by_engine.keys()):
                if by_engine[engine]:
                    limit = SLOT_LIMITS.get(engine, 1)
                    take = min(limit, len(by_engine[engine]))
                    bundle[engine] = by_engine[engine][:take]
                    by_engine[engine] = by_engine[engine][take:]
            if bundle:
                self.instrs.append(bundle)

    def build_kernel(
        self, forest_height: int, n_nodes: int, batch_size: int, rounds: int
    ):
        """
        Optimized vectorized kernel with UNROLL=4, handling slot limits properly.
        """
        UNROLL = 4  # Process 4 vectors per iteration (32 elements)

        # ============ ALLOCATION PHASE ============
        tmp1 = self.alloc_scratch("tmp1")
        tmp2 = self.alloc_scratch("tmp2")

        init_vars = [
            "rounds", "n_nodes", "batch_size", "forest_height",
            "forest_values_p", "inp_indices_p", "inp_values_p",
        ]
        for v in init_vars:
            self.alloc_scratch(v, 1)

        # Allocate vectors for UNROLL streams
        v_idx = [self.alloc_scratch(f"v_idx_{i}", VLEN) for i in range(UNROLL)]
        v_val = [self.alloc_scratch(f"v_val_{i}", VLEN) for i in range(UNROLL)]
        v_node_val = [self.alloc_scratch(f"v_node_val_{i}", VLEN) for i in range(UNROLL)]
        v_tmp1 = [self.alloc_scratch(f"v_tmp1_{i}", VLEN) for i in range(UNROLL)]
        v_tmp2 = [self.alloc_scratch(f"v_tmp2_{i}", VLEN) for i in range(UNROLL)]
        v_addrs = [self.alloc_scratch(f"v_addrs_{i}", VLEN) for i in range(UNROLL)]

        loop_i = self.alloc_scratch("loop_i")
        round_ctr = self.alloc_scratch("round_ctr")
        idx_base = [self.alloc_scratch(f"idx_base_{i}") for i in range(UNROLL)]
        val_base = [self.alloc_scratch(f"val_base_{i}") for i in range(UNROLL)]

        n_vectors = batch_size // VLEN

        # ============ INITIALIZATION ============
        for i, v in enumerate(init_vars):
            self.add("load", ("const", tmp1, i))
            self.add("load", ("load", self.scratch[v], tmp1))

        # Constants
        vlen_stride = self.scratch_const(VLEN * UNROLL)
        n_iters_const = self.scratch_const(n_vectors // UNROLL)
        rounds_const = self.scratch_const(rounds)

        v_zero = self.scratch_const_vec(0)
        v_one = self.scratch_const_vec(1)
        v_two = self.scratch_const_vec(2)
        v_n_nodes = self.scratch_const_vec(n_nodes)

        # Offsets for computing additional base addresses
        offsets = [self.scratch_const(i * VLEN) for i in range(UNROLL)]

        hash_const_vecs = []
        for hi, (op1, val1, op2, op3, val3) in enumerate(HASH_STAGES):
            hash_const_vecs.append((self.scratch_const_vec(val1), self.scratch_const_vec(val3)))

        self.add("flow", ("pause",))

        # ============ MAIN LOOP ============
        self.add("load", ("const", round_ctr, 0))
        round_loop_start = len(self.instrs)

        self.add("load", ("const", loop_i, 0))
        batch_loop_start = len(self.instrs)

        # Compute base addresses
        self.instrs.append({"alu": [("*", tmp1, loop_i, vlen_stride)]})
        self.instrs.append({
            "alu": [
                ("+", idx_base[0], self.scratch["inp_indices_p"], tmp1),
                ("+", val_base[0], self.scratch["inp_values_p"], tmp1),
            ]
        })

        # Compute other base addresses (can do up to 12 ALU ops)
        alu_slots = []
        for u in range(1, UNROLL):
            alu_slots.append(("+", idx_base[u], idx_base[0], offsets[u]))
            alu_slots.append(("+", val_base[u], val_base[0], offsets[u]))
        self.instrs.append({"alu": alu_slots})

        # Load all indices and values (2 vloads per cycle)
        for u in range(UNROLL):
            self.instrs.append({
                "load": [
                    ("vload", v_idx[u], idx_base[u]),
                    ("vload", v_val[u], val_base[u]),
                ]
            })

        # Compute gather addresses (8 ALU slots per vector, split across cycles)
        for u in range(UNROLL):
            self.instrs.append({
                "alu": [
                    ("+", v_addrs[u] + vi, self.scratch["forest_values_p"], v_idx[u] + vi)
                    for vi in range(VLEN)
                ]
            })

        # Load node values (2 loads per cycle, 4 cycles per vector, 16 cycles total)
        for vi in range(0, VLEN, 2):
            for u in range(UNROLL):
                self.instrs.append({
                    "load": [
                        ("load", v_node_val[u] + vi, v_addrs[u] + vi),
                        ("load", v_node_val[u] + vi + 1, v_addrs[u] + vi + 1),
                    ]
                })

        # XOR for all vectors (can do up to 6 per cycle)
        for u in range(0, UNROLL, 6):
            end = min(u + 6, UNROLL)
            self.instrs.append({
                "valu": [("^", v_val[i], v_val[i], v_node_val[i]) for i in range(u, end)]
            })

        # Hash stages - split 4 vectors into 2 groups of 2 to fit 6 valu slots
        for hi, (op1, val1, op2, op3, val3) in enumerate(HASH_STAGES):
            const1_vec, const3_vec = hash_const_vecs[hi]
            # First half (vectors 0,1)
            valu_ops = []
            for u in range(0, 2):
                valu_ops.append((op1, v_tmp1[u], v_val[u], const1_vec))
                valu_ops.append((op3, v_tmp2[u], v_val[u], const3_vec))
            self.instrs.append({"valu": valu_ops})
            # Second half (vectors 2,3)
            valu_ops = []
            for u in range(2, 4):
                valu_ops.append((op1, v_tmp1[u], v_val[u], const1_vec))
                valu_ops.append((op3, v_tmp2[u], v_val[u], const3_vec))
            self.instrs.append({"valu": valu_ops})
            # Combine for all 4 vectors (4 ops, fits in 1 cycle)
            valu_ops = []
            for u in range(UNROLL):
                valu_ops.append((op2, v_val[u], v_tmp1[u], v_tmp2[u]))
            self.instrs.append({"valu": valu_ops})

        # Index update: idx = 2*idx + (val&1) + 1
        valu_ops = []
        for u in range(0, 2):
            valu_ops.append(("&", v_tmp1[u], v_val[u], v_one))
            valu_ops.append(("*", v_idx[u], v_idx[u], v_two))
        self.instrs.append({"valu": valu_ops})
        valu_ops = []
        for u in range(2, 4):
            valu_ops.append(("&", v_tmp1[u], v_val[u], v_one))
            valu_ops.append(("*", v_idx[u], v_idx[u], v_two))
        self.instrs.append({"valu": valu_ops})

        valu_ops = []
        for u in range(UNROLL):
            valu_ops.append(("+", v_tmp1[u], v_tmp1[u], v_one))
        self.instrs.append({"valu": valu_ops})

        valu_ops = []
        for u in range(UNROLL):
            valu_ops.append(("+", v_idx[u], v_idx[u], v_tmp1[u]))
        self.instrs.append({"valu": valu_ops})

        # Bounds check: mask = -(idx < n_nodes), idx &= mask
        valu_ops = []
        for u in range(UNROLL):
            valu_ops.append(("<", v_tmp1[u], v_idx[u], v_n_nodes))
        self.instrs.append({"valu": valu_ops})

        valu_ops = []
        for u in range(UNROLL):
            valu_ops.append(("-", v_tmp1[u], v_zero, v_tmp1[u]))
        self.instrs.append({"valu": valu_ops})

        valu_ops = []
        for u in range(UNROLL):
            valu_ops.append(("&", v_idx[u], v_idx[u], v_tmp1[u]))
        self.instrs.append({"valu": valu_ops})

        # Store results (2 vstores per cycle)
        for u in range(UNROLL):
            self.instrs.append({
                "store": [
                    ("vstore", idx_base[u], v_idx[u]),
                    ("vstore", val_base[u], v_val[u]),
                ]
            })

        # Loop control
        self.instrs.append({"flow": [("add_imm", loop_i, loop_i, 1)]})
        self.instrs.append({"alu": [("<", tmp1, loop_i, n_iters_const)]})
        self.instrs.append({"flow": [("cond_jump", tmp1, batch_loop_start)]})

        self.instrs.append({"flow": [("add_imm", round_ctr, round_ctr, 1)]})
        self.instrs.append({"alu": [("<", tmp1, round_ctr, rounds_const)]})
        self.instrs.append({"flow": [("cond_jump", tmp1, round_loop_start)]})

        self.instrs.append({"flow": [("pause",)]})

BASELINE = 147734

def do_kernel_test(
    forest_height: int,
    rounds: int,
    batch_size: int,
    seed: int = 123,
    trace: bool = False,
    prints: bool = False,
):
    print(f"{forest_height=}, {rounds=}, {batch_size=}")
    random.seed(seed)
    forest = Tree.generate(forest_height)
    inp = Input.generate(forest, batch_size, rounds)
    mem = build_mem_image(forest, inp)

    kb = KernelBuilder()
    kb.build_kernel(forest.height, len(forest.values), len(inp.indices), rounds)
    # print(kb.instrs)

    value_trace = {}
    machine = Machine(
        mem,
        kb.instrs,
        kb.debug_info(),
        n_cores=N_CORES,
        value_trace=value_trace,
        trace=trace,
    )
    machine.prints = prints
    for i, ref_mem in enumerate(reference_kernel2(mem, value_trace)):
        machine.run()
        inp_values_p = ref_mem[6]
        if prints:
            print(machine.mem[inp_values_p : inp_values_p + len(inp.values)])
            print(ref_mem[inp_values_p : inp_values_p + len(inp.values)])
        assert (
            machine.mem[inp_values_p : inp_values_p + len(inp.values)]
            == ref_mem[inp_values_p : inp_values_p + len(inp.values)]
        ), f"Incorrect result on round {i}"
        inp_indices_p = ref_mem[5]
        if prints:
            print(machine.mem[inp_indices_p : inp_indices_p + len(inp.indices)])
            print(ref_mem[inp_indices_p : inp_indices_p + len(inp.indices)])
        # Updating these in memory isn't required, but you can enable this check for debugging
        # assert machine.mem[inp_indices_p:inp_indices_p+len(inp.indices)] == ref_mem[inp_indices_p:inp_indices_p+len(inp.indices)]

    print("CYCLES: ", machine.cycle)
    print("Speedup over baseline: ", BASELINE / machine.cycle)
    return machine.cycle


class Tests(unittest.TestCase):
    def test_ref_kernels(self):
        """
        Test the reference kernels against each other
        """
        random.seed(123)
        for i in range(10):
            f = Tree.generate(4)
            inp = Input.generate(f, 10, 6)
            mem = build_mem_image(f, inp)
            reference_kernel(f, inp)
            for _ in reference_kernel2(mem, {}):
                pass
            assert inp.indices == mem[mem[5] : mem[5] + len(inp.indices)]
            assert inp.values == mem[mem[6] : mem[6] + len(inp.values)]

    def test_kernel_trace(self):
        # Full-scale example for performance testing
        do_kernel_test(10, 16, 256, trace=True, prints=False)

    # Passing this test is not required for submission, see submission_tests.py for the actual correctness test
    # You can uncomment this if you think it might help you debug
    # def test_kernel_correctness(self):
    #     for batch in range(1, 3):
    #         for forest_height in range(3):
    #             do_kernel_test(
    #                 forest_height + 2, forest_height + 4, batch * 16 * VLEN * N_CORES
    #             )

    def test_kernel_cycles(self):
        do_kernel_test(10, 16, 256)


# To run all the tests:
#    python perf_takehome.py
# To run a specific test:
#    python perf_takehome.py Tests.test_kernel_cycles
# To view a hot-reloading trace of all the instructions:  **Recommended debug loop**
# NOTE: The trace hot-reloading only works in Chrome. In the worst case if things aren't working, drag trace.json onto https://ui.perfetto.dev/
#    python perf_takehome.py Tests.test_kernel_trace
# Then run `python watch_trace.py` in another tab, it'll open a browser tab, then click "Open Perfetto"
# You can then keep that open and re-run the test to see a new trace.

# To run the proper checks to see which thresholds you pass:
#    python tests/submission_tests.py

if __name__ == "__main__":
    unittest.main()
