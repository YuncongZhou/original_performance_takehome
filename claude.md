# Performance Optimization Notes

## CRITICAL RULE: NEVER MODIFY TESTS
⚠️ **DO NOT modify anything in the `tests/` folder under ANY circumstances.**

This is a strict requirement for valid submissions. Any modification to test files is considered cheating and will invalidate the solution.

Verify test integrity before submission:
```bash
git diff upstream/main tests/  # Should be empty
python tests/submission_tests.py  # Use this for verification
```

## Goal
Achieve **< 1487 cycles** to beat Claude Opus 4.5's best performance at launch.

## Optimization Philosophy
**SINGULAR FOCUS: Cycle count is the ONLY metric that matters.**

When making changes or reviewing code:
- ✅ **DO** prioritize any optimization that reduces cycles, no matter how small
- ✅ **DO** sacrifice code readability, maintainability, or elegance for performance
- ✅ **DO** use aggressive techniques (unrolling, inlining, manual scheduling)
- ❌ **DON'T** worry about code style, comments, or best practices
- ❌ **DON'T** refactor for readability unless it also improves performance
- ❌ **DON'T** spend time on anything that doesn't directly reduce cycle count

The benchmark is unforgiving. Meeting the **< 1487 cycles** threshold is our sole objective.

## Published Benchmarks (Source of Truth)
- **2164 cycles**: Claude Opus 4 after many hours in the test-time compute harness
- **1790 cycles**: Claude Opus 4.5 in a casual Claude Code session, approximately matching the best human performance in 2 hours
- **1579 cycles**: Claude Opus 4.5 after 2 hours in our test-time compute harness
- **1548 cycles**: Claude Sonnet 4.5 after many more than 2 hours of test-time compute
- **1487 cycles**: Claude Opus 4.5 after 11.5 hours in the harness
- **1363 cycles**: Claude Opus 4.5 in an improved test time compute harness

## Current Status
- **3490 cycles** - Current best implementation
- Need **2.35x improvement** to reach goal
- Speedup over baseline: **42.3x**

## Architecture Constraints
- VLEN = 8 (vector width)
- Slots per cycle: 6 VALU, 2 Load, 2 Store, 12 ALU, 1 Flow
- Scratch size: 1536 words (currently using 817)
- Processing: 256 elements (32 vectors) through 16 rounds

## Key Observations (Updated)
- Total loads: 3239 → minimum 1620 cycles (load-bound)
- Total VALU ops: 8831 → minimum 1472 cycles
- Current instruction breakdown:
  - 649 load-only cycles (wasted VALU slots)
  - 1589 VALU-only cycles (wasted load slots)
  - 1020 overlapped cycles (good utilization)
- To hit <1487 cycles: need <2974 loads (must reduce by 265+)

## Optimizations Applied
1. **Instruction merging** (100 pairs merged): 3514 → 3414 cycles
2. **multiply_add for index update**: 3414 → 3394 cycles

## Critical Insight
To achieve <1487 cycles, we MUST:
1. **Reduce loads** via k-selection for rounds 2-4, 13-15
2. **Overlap k-selection VALU** with remaining full gather loads

K-selection load savings potential:
- k=4 (rounds 2,13): 504 loads saved
- k=8 (rounds 3,14): 496 loads saved
- k=16 (rounds 4,15): 480 loads saved
- Total: 1480 loads saved → 1759 remaining → 880 load cycles

Challenge: k-selection adds VALU overhead that must overlap with other work.

## Round Structure
- Round 0, 11: k=1 (all at idx=0) - already optimized
- Round 1, 12: k=2 (indices 1-2) - already optimized
- Round 2, 13: k=4 (indices 3-6) - k-selection candidate
- Round 3, 14: k=8 (indices 7-14) - k-selection candidate
- Round 4, 15: k=16 (indices 15-30) - k-selection candidate
- Rounds 5-10: Full gather (32+ unique) - load-heavy

## Next Steps (Priority Order)

### 1. Implement Vector-Level Pipelining
Process vectors in staggered fashion:
- Wave A (vectors 0-15) at round R
- Wave B (vectors 16-31) at round R-1
- Overlap Wave A's loads with Wave B's VALU

### 2. K-selection with Proper Overlap
- Load k nodes at round start
- Broadcast to vectors
- VALU-based selection fills slots during other operations

### 3. Global Instruction Scheduling
Build dependency graph of all operations, then schedule to maximize slot utilization.
