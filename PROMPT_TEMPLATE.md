# Prompt Template for Parallel Optimizer

## CRITICAL MINDSET (Include in EVERY prompt):

**NO CAUTION NEEDED**: This is NOT production code. Our ONLY goal is to decrease cycle count. Everything is on the table:
- Break code style conventions
- Use hacky solutions
- Try extreme approaches
- Make code unreadable if it's faster
- Take risks - we can always revert

The simulator is our judge. If tests pass and cycles decrease, the code is GOOD.

## Sections to Include:

## 1. GOAL & METRICS
- Target cycle count
- Expected savings calculation with math
- Why this approach should work (theoretical justification)

## 2. KEY INSIGHT / ANALYSIS
- The problem this solves
- Mathematical proof or evidence
- Comparison to current approach

## 3. EXISTING CODE CONTEXT
- Specific file paths and line numbers
- Relevant existing methods that implement similar patterns
- Code snippets showing the pattern to follow

Example:
```
Lines 238-332 in perf_takehome.py show _emit_round_k2() which:
- Loads 2 nodes once (not 64 times)
- Broadcasts to all vectors
- Uses VALU for selection
```

## 4. IMPLEMENTATION PLAN
- Step-by-step modifications needed
- Specific locations to change (line numbers)
- Before/after code structure

Example:
```python
# BEFORE (lines 197-205):
else:
    self._emit_round_full_gather(...)

# AFTER:
elif effective_round == 2:
    self._emit_round_k4(...)
elif effective_round == 3:
    self._emit_round_k8(...)
else:
    self._emit_round_full_gather(...)
```

## 5. ALGORITHM OPTIONS
- Multiple approaches to solve the problem
- Trade-offs of each approach
- Recommendation for which to try first

## 6. VALIDATION
- How to test the changes
- Expected cycle count after implementation
- Intermediate milestones

## BAD PROMPT (too vague):
```
"Implement k-selection for rounds 2-4. Goal: < 3300 cycles."
```

## GOOD PROMPT (detailed analysis):
```
"K-SELECTION PRE-CACHING: Full Analysis

WHY THIS WORKS:
Round 2 currently does ~100 loads. With k=4 pre-caching:
- Load 4 unique nodes ONCE
- Broadcast to 32 vectors
- Use VALU for selection
- Total: 4 loads (save 96 loads = 48 cycles per round)

EXISTING PATTERN:
Lines 238-332 show _emit_round_k2() already does this for k=2:
[code snippet]

IMPLEMENTATION:
1. Create _emit_round_k4() following k2 pattern
2. Modify build_kernel() lines 197-205 to dispatch
3. Test with rounds 2,13 first

TARGET: < 3000 cycles
"
```

## REMEMBER:
The receiver is Claude Opus 4.5 - it can do the same reasoning you can!
Give it ALL the context, analysis, and specific implementation details you would want.
