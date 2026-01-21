# Performance Optimization Notes

## Goal
Achieve **< 1487 cycles** to beat Claude Opus 4.5's best performance at launch.

## Published Benchmarks (Source of Truth)
- **2164 cycles**: Claude Opus 4 after many hours in the test-time compute harness
- **1790 cycles**: Claude Opus 4.5 in a casual Claude Code session, approximately matching the best human performance in 2 hours
- **1579 cycles**: Claude Opus 4.5 after 2 hours in our test-time compute harness
- **1548 cycles**: Claude Sonnet 4.5 after many more than 2 hours of test-time compute
- **1487 cycles**: Claude Opus 4.5 after 11.5 hours in the harness
- **1363 cycles**: Claude Opus 4.5 in an improved test time compute harness

## Current Status
- **3514 cycles** - Current implementation
- Need **2.36x improvement** to reach goal

## Architecture Constraints
- VLEN = 8 (vector width)
- Slots per cycle: 6 VALU, 2 Load, 2 Store, 12 ALU, 1 Flow
- Scratch size: 1536 words (currently using 817)
- Processing: 256 elements (32 vectors) through 16 rounds

## Key Observations
- VALU utilization: only 42.9% - lots of wasted VALU cycles
- 3085 scalar loads needed, but spread across 3514 cycles
- Theoretical minimum based on loads alone: ~800 cycles
- Theoretical minimum based on VALU alone: ~1200 cycles

## Round Structure
- Round 0, 11: k=1 (all at same index) - already optimized
- Round 1, 12: k=2 (2 unique indices) - already optimized
- Round 2, 13: k=4 (4 unique indices) - OPPORTUNITY
- Round 3, 14: k=8 (8 unique indices) - OPPORTUNITY
- Round 4, 15: k=16 (16 unique indices) - OPPORTUNITY
- Rounds 5-10: Full gather (32+ unique) - need optimization

## Reflection & Top 5 Ideas (Priority Order)

### 1. Wave-based Round Pipelining (HIGH PRIORITY)
Process vectors in two waves (0-15, 16-31) staggered by one round:
- While Wave A does gather (load-heavy) for round R
- Wave B does hash/index (VALU-heavy) for round R-1
- This overlaps the 57% wasted VALU with useful work
- Potential: Could nearly halve total cycles if perfectly balanced

### 2. Aggressive k-selection for rounds 2-4, 13-15
Replace gather with VALU-based selection:
- k=4: 4 loads + VALU select vs 128 loads - save ~100 cycles/round
- k=8: 8 loads + VALU select vs 128 loads - save ~80 cycles/round
- k=16: 16 loads + VALU select vs 128 loads - save ~50 cycles/round
- Total potential savings: ~600 cycles

### 3. Better VALU Packing During Full Gather
Fill the 57% wasted VALU slots during gather loads:
- During 8-cycle Gather A: 48 VALU slots available but unused
- Could precompute hash constants, do address calculations
- Incremental improvement: ~100-200 cycles

### 4. Fused Multi-Round Processing for k-selection rounds
Preload ALL node values for rounds 2,3,4 (28 values total):
- Process rounds 2,3,4 back-to-back without returning to memory
- Only need index updates (VALU) between rounds
- Could save setup/teardown overhead

### 5. Process 6 Vectors per Group (instead of 4)
Better utilize ALU slots during address computation:
- 6 vectors need 48 address ops, fitting in 4 ALU cycles
- Reduces total groups from 8 to 6
- May improve instruction density
