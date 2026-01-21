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
        Optimized vectorized kernel with heavy VLIW pipelining.
        Key insight: Overlap operations across multiple vector groups.
        """
        UNROLL = 4  # Process 4 vectors per iteration (32 elements)
        # With 256 elements / 8 VLEN / 4 UNROLL = 8 iterations per round

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

        # Compute base addresses and overlap with early loads
        # Cycle 1: compute tmp1
        self.instrs.append({"alu": [("*", tmp1, loop_i, vlen_stride)]})

        # Cycle 2: compute base[0]
        self.instrs.append({
            "alu": [
                ("+", idx_base[0], self.scratch["inp_indices_p"], tmp1),
                ("+", val_base[0], self.scratch["inp_values_p"], tmp1),
            ]
        })

        # Cycle 3: compute base[1..3] + load vec 0
        alu_slots = []
        for u in range(1, UNROLL):
            alu_slots.append(("+", idx_base[u], idx_base[0], offsets[u]))
            alu_slots.append(("+", val_base[u], val_base[0], offsets[u]))
        self.instrs.append({
            "alu": alu_slots,
            "load": [
                ("vload", v_idx[0], idx_base[0]),
                ("vload", v_val[0], val_base[0]),
            ]
        })

        # Load remaining vecs while computing gather addresses for vec 0
        # Cycle 4: load vec 1 + compute gather addrs for vec 0
        self.instrs.append({
            "load": [
                ("vload", v_idx[1], idx_base[1]),
                ("vload", v_val[1], val_base[1]),
            ],
            "alu": [
                ("+", v_addrs[0] + vi, self.scratch["forest_values_p"], v_idx[0] + vi)
                for vi in range(VLEN)
            ]
        })

        # Cycle 5: load vec 2 + compute gather addrs for vec 1
        self.instrs.append({
            "load": [
                ("vload", v_idx[2], idx_base[2]),
                ("vload", v_val[2], val_base[2]),
            ],
            "alu": [
                ("+", v_addrs[1] + vi, self.scratch["forest_values_p"], v_idx[1] + vi)
                for vi in range(VLEN)
            ]
        })

        # Cycle 6: load vec 3 + compute gather addrs for vec 2
        self.instrs.append({
            "load": [
                ("vload", v_idx[3], idx_base[3]),
                ("vload", v_val[3], val_base[3]),
            ],
            "alu": [
                ("+", v_addrs[2] + vi, self.scratch["forest_values_p"], v_idx[2] + vi)
                for vi in range(VLEN)
            ]
        })

        # ====== INTERLEAVED LOAD + COMPUTE SCHEDULE ======
        # Cycle 7: compute gather addrs for vec 3 + start loading node_val for vec 0
        self.instrs.append({
            "alu": [
                ("+", v_addrs[3] + vi, self.scratch["forest_values_p"], v_idx[3] + vi)
                for vi in range(VLEN)
            ],
            "load": [
                ("load", v_node_val[0] + 0, v_addrs[0] + 0),
                ("load", v_node_val[0] + 1, v_addrs[0] + 1),
            ]
        })

        # Continue loading Group A's node values (vectors 0,1)
        for vi in range(2, VLEN, 2):
            self.instrs.append({
                "load": [
                    ("load", v_node_val[0] + vi, v_addrs[0] + vi),
                    ("load", v_node_val[0] + vi + 1, v_addrs[0] + vi + 1),
                ]
            })
        for vi in range(0, VLEN, 2):
            self.instrs.append({
                "load": [
                    ("load", v_node_val[1] + vi, v_addrs[1] + vi),
                    ("load", v_node_val[1] + vi + 1, v_addrs[1] + vi + 1),
                ]
            })

        # XOR for Group A
        self.instrs.append({
            "valu": [("^", v_val[0], v_val[0], v_node_val[0]),
                     ("^", v_val[1], v_val[1], v_node_val[1])]
        })

        # Interleaved: Hash Group A while loading Group B
        # Group B needs 8 load pairs: 4 for vec2, 4 for vec3
        load_pair_idx = 0  # 0-7 for the 8 load pairs
        for hi, (op1, val1, op2, op3, val3) in enumerate(HASH_STAGES):
            const1_vec, const3_vec = hash_const_vecs[hi]

            # Hash stage first ops for Group A (vectors 0,1) - 4 valu ops
            valu_ops = [(op1, v_tmp1[0], v_val[0], const1_vec),
                        (op3, v_tmp2[0], v_val[0], const3_vec),
                        (op1, v_tmp1[1], v_val[1], const1_vec),
                        (op3, v_tmp2[1], v_val[1], const3_vec)]

            # Try to add Group B loads in same cycle
            if load_pair_idx < 8:
                vec_idx = 2 + (load_pair_idx // 4)  # 0-3 -> vec 2, 4-7 -> vec 3
                vi = (load_pair_idx % 4) * 2  # 0,2,4,6
                self.instrs.append({
                    "valu": valu_ops,
                    "load": [
                        ("load", v_node_val[vec_idx] + vi, v_addrs[vec_idx] + vi),
                        ("load", v_node_val[vec_idx] + vi + 1, v_addrs[vec_idx] + vi + 1),
                    ]
                })
                load_pair_idx += 1
            else:
                self.instrs.append({"valu": valu_ops})

            # Hash stage combine for Group A (vectors 0,1) - 2 valu ops
            valu_ops = [(op2, v_val[0], v_tmp1[0], v_tmp2[0]),
                        (op2, v_val[1], v_tmp1[1], v_tmp2[1])]

            # Try to add Group B loads in same cycle
            if load_pair_idx < 8:
                vec_idx = 2 + (load_pair_idx // 4)
                vi = (load_pair_idx % 4) * 2
                self.instrs.append({
                    "valu": valu_ops,
                    "load": [
                        ("load", v_node_val[vec_idx] + vi, v_addrs[vec_idx] + vi),
                        ("load", v_node_val[vec_idx] + vi + 1, v_addrs[vec_idx] + vi + 1),
                    ]
                })
                load_pair_idx += 1
            else:
                self.instrs.append({"valu": valu_ops})

        # Finish any remaining Group B loads
        while load_pair_idx < 8:
            vec_idx = 2 + (load_pair_idx // 4)
            vi = (load_pair_idx % 4) * 2
            self.instrs.append({
                "load": [
                    ("load", v_node_val[vec_idx] + vi, v_addrs[vec_idx] + vi),
                    ("load", v_node_val[vec_idx] + vi + 1, v_addrs[vec_idx] + vi + 1),
                ]
            })
            load_pair_idx += 1

        # XOR for Group B
        self.instrs.append({
            "valu": [("^", v_val[2], v_val[2], v_node_val[2]),
                     ("^", v_val[3], v_val[3], v_node_val[3])]
        })

        # Hash Group B (vectors 2,3)
        for hi, (op1, val1, op2, op3, val3) in enumerate(HASH_STAGES):
            const1_vec, const3_vec = hash_const_vecs[hi]
            valu_ops = [(op1, v_tmp1[2], v_val[2], const1_vec),
                        (op3, v_tmp2[2], v_val[2], const3_vec),
                        (op1, v_tmp1[3], v_val[3], const1_vec),
                        (op3, v_tmp2[3], v_val[3], const3_vec)]
            self.instrs.append({"valu": valu_ops})
            valu_ops = [(op2, v_val[2], v_tmp1[2], v_tmp2[2]),
                        (op2, v_val[3], v_tmp1[3], v_tmp2[3])]
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

        # Store results + update loop counter (overlap store with ALU)
        # First store
        self.instrs.append({
            "store": [
                ("vstore", idx_base[0], v_idx[0]),
                ("vstore", val_base[0], v_val[0]),
            ],
            "flow": [("add_imm", loop_i, loop_i, 1)]
        })
        # Second store + compare
        self.instrs.append({
            "store": [
                ("vstore", idx_base[1], v_idx[1]),
                ("vstore", val_base[1], v_val[1]),
            ],
            "alu": [("<", tmp1, loop_i, n_iters_const)]
        })
        # Third store
        self.instrs.append({
            "store": [
                ("vstore", idx_base[2], v_idx[2]),
                ("vstore", val_base[2], v_val[2]),
            ]
        })
        # Fourth store + jump
        self.instrs.append({
            "store": [
                ("vstore", idx_base[3], v_idx[3]),
                ("vstore", val_base[3], v_val[3]),
            ],
            "flow": [("cond_jump", tmp1, batch_loop_start)]
        })

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
