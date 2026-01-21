"""
Index deduplication kernel.
Key insight: All elements start at idx=0, so early rounds have few unique indices.
- Round 0: 1 gather, broadcast to all 256 elements
- Round 1: 2 gathers, select using VALU (mask-based, not vselect)
- Round 2+: progressively more until full gather

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
        Kernel with index deduplication for early rounds.
        Structure:
        - Keep all 32 vectors in scratch across rounds (no load/store per iteration)
        - Round 0: single gather + broadcast
        - Rounds 1+: normal gather with pipelining
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

        # Working buffers for 4 vectors per group
        v_node = [self.alloc_scratch(f"node_{i}", VLEN) for i in range(4)]
        v_tmp1 = [self.alloc_scratch(f"tmp1_{i}", VLEN) for i in range(4)]
        v_tmp2 = [self.alloc_scratch(f"tmp2_{i}", VLEN) for i in range(4)]
        v_addrs = [self.alloc_scratch(f"addrs_{i}", VLEN) for i in range(4)]

        round_ctr = self.alloc_scratch("round_ctr")

        # ============ INITIALIZATION ============
        for i, v in enumerate(init_vars):
            self.add("load", ("const", tmp1, i))
            self.add("load", ("load", self.scratch[v], tmp1))

        rounds_const = self.scratch_const(rounds)
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

        # ============ ROUND 0: All elements at idx=0 ============
        # Single load, broadcast to all vectors
        self.add("load", ("load", tmp1, self.scratch["forest_values_p"]))
        self.add("valu", ("vbroadcast", v_node[0], tmp1))

        # XOR all vectors with the single node value
        for vi in range(0, n_vectors, 2):
            self.instrs.append({
                "valu": [("^", all_val[vi], all_val[vi], v_node[0]),
                         ("^", all_val[vi + 1], all_val[vi + 1], v_node[0])]
            })

        # Hash all vectors (process 3 at a time using 6 valu slots)
        for vi in range(0, n_vectors, 3):
            vecs_this_batch = min(3, n_vectors - vi)
            for hi, (op1, val1, op2, op3, val3) in enumerate(HASH_STAGES):
                const1_vec, const3_vec = hash_const_vecs[hi]
                if vecs_this_batch == 3:
                    self.instrs.append({
                        "valu": [(op1, v_tmp1[0], all_val[vi], const1_vec),
                                 (op3, v_tmp2[0], all_val[vi], const3_vec),
                                 (op1, v_tmp1[1], all_val[vi+1], const1_vec),
                                 (op3, v_tmp2[1], all_val[vi+1], const3_vec),
                                 (op1, v_tmp1[2], all_val[vi+2], const1_vec),
                                 (op3, v_tmp2[2], all_val[vi+2], const3_vec)]
                    })
                    self.instrs.append({
                        "valu": [(op2, all_val[vi], v_tmp1[0], v_tmp2[0]),
                                 (op2, all_val[vi+1], v_tmp1[1], v_tmp2[1]),
                                 (op2, all_val[vi+2], v_tmp1[2], v_tmp2[2])]
                    })
                elif vecs_this_batch == 2:
                    self.instrs.append({
                        "valu": [(op1, v_tmp1[0], all_val[vi], const1_vec),
                                 (op3, v_tmp2[0], all_val[vi], const3_vec),
                                 (op1, v_tmp1[1], all_val[vi+1], const1_vec),
                                 (op3, v_tmp2[1], all_val[vi+1], const3_vec)]
                    })
                    self.instrs.append({
                        "valu": [(op2, all_val[vi], v_tmp1[0], v_tmp2[0]),
                                 (op2, all_val[vi+1], v_tmp1[1], v_tmp2[1])]
                    })
                else:
                    self.instrs.append({
                        "valu": [(op1, v_tmp1[0], all_val[vi], const1_vec),
                                 (op3, v_tmp2[0], all_val[vi], const3_vec)]
                    })
                    self.instrs.append({
                        "valu": [(op2, all_val[vi], v_tmp1[0], v_tmp2[0])]
                    })

        # Index update all vectors (process 2 at a time)
        for vi in range(0, n_vectors, 2):
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

        # ============ ROUNDS 1-15: Normal gather with pipelining ============
        self.add("load", ("const", round_ctr, 1))
        round_loop_start = len(self.instrs)

        n_groups = n_vectors // 4  # 8 groups

        for group_idx in range(n_groups):
            g = group_idx * 4

            # Compute addresses for all 4 vectors
            for v in range(4):
                self.instrs.append({
                    "alu": [("+", v_addrs[v] + vi, self.scratch["forest_values_p"], all_idx[g + v] + vi)
                            for vi in range(VLEN)]
                })

            # Gather A (vec 0,1)
            for vi in range(VLEN):
                self.instrs.append({
                    "load": [("load", v_node[0] + vi, v_addrs[0] + vi),
                             ("load", v_node[1] + vi, v_addrs[1] + vi)]
                })

            # XOR A
            self.instrs.append({
                "valu": [("^", all_val[g], all_val[g], v_node[0]),
                         ("^", all_val[g + 1], all_val[g + 1], v_node[1])]
            })

            # Hash A + Gather B overlapped
            load_idx = 0
            for hi, (op1, val1, op2, op3, val3) in enumerate(HASH_STAGES):
                const1_vec, const3_vec = hash_const_vecs[hi]

                vi = load_idx
                if vi < VLEN:
                    self.instrs.append({
                        "valu": [(op1, v_tmp1[0], all_val[g], const1_vec),
                                 (op3, v_tmp2[0], all_val[g], const3_vec),
                                 (op1, v_tmp1[1], all_val[g + 1], const1_vec),
                                 (op3, v_tmp2[1], all_val[g + 1], const3_vec)],
                        "load": [("load", v_node[2] + vi, v_addrs[2] + vi),
                                 ("load", v_node[3] + vi, v_addrs[3] + vi)]
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
                        "load": [("load", v_node[2] + vi, v_addrs[2] + vi),
                                 ("load", v_node[3] + vi, v_addrs[3] + vi)]
                    })
                    load_idx += 1
                else:
                    self.instrs.append({
                        "valu": [(op2, all_val[g], v_tmp1[0], v_tmp2[0]),
                                 (op2, all_val[g + 1], v_tmp1[1], v_tmp2[1])]
                    })

            while load_idx < VLEN:
                self.instrs.append({
                    "load": [("load", v_node[2] + load_idx, v_addrs[2] + load_idx),
                             ("load", v_node[3] + load_idx, v_addrs[3] + load_idx)]
                })
                load_idx += 1

            # XOR B
            self.instrs.append({
                "valu": [("^", all_val[g + 2], all_val[g + 2], v_node[2]),
                         ("^", all_val[g + 3], all_val[g + 3], v_node[3])]
            })

            # Hash B + Index A
            for hi, (op1, val1, op2, op3, val3) in enumerate(HASH_STAGES):
                const1_vec, const3_vec = hash_const_vecs[hi]

                if hi == 0:
                    self.instrs.append({
                        "valu": [(op1, v_tmp1[2], all_val[g + 2], const1_vec),
                                 (op3, v_tmp2[2], all_val[g + 2], const3_vec),
                                 (op1, v_tmp1[3], all_val[g + 3], const1_vec),
                                 (op3, v_tmp2[3], all_val[g + 3], const3_vec),
                                 ("&", v_node[0], all_val[g], v_one),
                                 ("*", all_idx[g], all_idx[g], v_two)]
                    })
                    self.instrs.append({
                        "valu": [(op2, all_val[g + 2], v_tmp1[2], v_tmp2[2]),
                                 (op2, all_val[g + 3], v_tmp1[3], v_tmp2[3]),
                                 ("&", v_node[1], all_val[g + 1], v_one),
                                 ("*", all_idx[g + 1], all_idx[g + 1], v_two)]
                    })
                elif hi == 1:
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
                    self.instrs.append({
                        "valu": [(op1, v_tmp1[2], all_val[g + 2], const1_vec),
                                 (op3, v_tmp2[2], all_val[g + 2], const3_vec),
                                 (op1, v_tmp1[3], all_val[g + 3], const1_vec),
                                 (op3, v_tmp2[3], all_val[g + 3], const3_vec),
                                 ("<", v_node[0], all_idx[g], v_n_nodes),
                                 ("<", v_node[1], all_idx[g + 1], v_n_nodes)]
                    })
                    self.instrs.append({
                        "valu": [(op2, all_val[g + 2], v_tmp1[2], v_tmp2[2]),
                                 (op2, all_val[g + 3], v_tmp1[3], v_tmp2[3]),
                                 ("-", v_node[0], v_zero, v_node[0]),
                                 ("-", v_node[1], v_zero, v_node[1])]
                    })
                elif hi == 3:
                    self.instrs.append({
                        "valu": [(op1, v_tmp1[2], all_val[g + 2], const1_vec),
                                 (op3, v_tmp2[2], all_val[g + 2], const3_vec),
                                 (op1, v_tmp1[3], all_val[g + 3], const1_vec),
                                 (op3, v_tmp2[3], all_val[g + 3], const3_vec),
                                 ("&", all_idx[g], all_idx[g], v_node[0]),
                                 ("&", all_idx[g + 1], all_idx[g + 1], v_node[1])]
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

            # Index B
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

        # Loop control (rounds 1-15)
        self.instrs.append({"flow": [("add_imm", round_ctr, round_ctr, 1)]})
        self.instrs.append({"alu": [("<", tmp1, round_ctr, rounds_const)]})
        self.instrs.append({"flow": [("cond_jump", tmp1, round_loop_start)]})

        # Store all values
        for vi in range(n_vectors):
            self.add("load", ("const", tmp1, vi * VLEN))
            self.instrs.append({"alu": [("+", tmp2, self.scratch["inp_values_p"], tmp1)]})
            self.instrs.append({"store": [("vstore", tmp2, all_val[vi])]})

        self.instrs.append({"flow": [("pause",)]})


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
