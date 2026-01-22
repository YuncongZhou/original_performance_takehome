#!/bin/zsh
# Parallel Optimization Session Manager
# Usage: ./parallel_optimizer.sh <target_cycles>

set -e

TARGET_CYCLES=${1:-3490}
TIMEOUT_MINUTES=60
PYTHON_CMD="/opt/homebrew/bin/python3.12"
TIMEOUT_CMD="gtimeout"  # GNU timeout from coreutils
MAIN_DIR="$(pwd)"
WORKTREE_BASE="${MAIN_DIR}/../perf-opt"
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
RESULTS_DIR="${MAIN_DIR}/.optimization_results_${TIMESTAMP}"

echo "🚀 Parallel Optimization Manager"
echo "=================================="
echo "Target: < ${TARGET_CYCLES} cycles"
echo "Timeout: ${TIMEOUT_MINUTES} minutes per approach"
echo "Main directory: ${MAIN_DIR}"
echo ""

# Create results directory
mkdir -p "$RESULTS_DIR"

# Define optimization approaches
typeset -A APPROACHES
APPROACHES=(
    k-selection "Implement k=4 selection for rounds 2,13 and k=8 for rounds 3,14. Measure cycles with tests/submission_tests.py. Goal: < ${TARGET_CYCLES} cycles. Only modify perf_takehome.py, never modify tests/."
    wave-pipeline "Implement wave-based pipelining: Wave A (vectors 0-15) at round R, Wave B (vectors 16-31) at round R-1. Overlap Wave A loads with Wave B VALU. Measure cycles with tests/submission_tests.py. Goal: < ${TARGET_CYCLES} cycles. Only modify perf_takehome.py, never modify tests/."
    aggressive-unroll "Aggressively unroll all loops and inline all function calls in the kernel. Eliminate all function call overhead. Measure cycles with tests/submission_tests.py. Goal: < ${TARGET_CYCLES} cycles. Only modify perf_takehome.py, never modify tests/."
    load-prefetch "Implement speculative load prefetching for future rounds. Load round N+1 data while computing round N. Measure cycles with tests/submission_tests.py. Goal: < ${TARGET_CYCLES} cycles. Only modify perf_takehome.py, never modify tests/."
)

# Function to run optimization in a worktree
run_optimization() {
    local name=$1
    local prompt=$2
    local worktree_path="${WORKTREE_BASE}-${name}"
    local result_file="${RESULTS_DIR}/${name}_result.txt"

    echo "[$name] Creating worktree at: $worktree_path"

    # Create worktree with new branch
    git worktree add "$worktree_path" -b "opt-${name}-${TIMESTAMP}" 2>&1 | tee "${RESULTS_DIR}/${name}_setup.log"

    echo "[$name] Running Claude optimization..."
    echo "[$name] Working directory: $worktree_path"
    echo "[$name] Results will be saved to: $result_file"

    # Run claude -p with timeout in background
    (
        cd "$worktree_path"

        # Run claude with the prompt
        $TIMEOUT_CMD ${TIMEOUT_MINUTES}m claude -p "$prompt" > "${result_file}" 2>&1
        exit_code=$?

        echo "[$name] Claude completed with exit code: $exit_code" >> "${result_file}"

        # Extract cycle count from test results
        if $PYTHON_CMD tests/submission_tests.py 2>&1 | tee -a "${result_file}" | grep -o "CYCLES:  [0-9]*" | tail -1 > "${RESULTS_DIR}/${name}_cycles.txt"; then
            cycles=$(cat "${RESULTS_DIR}/${name}_cycles.txt" | awk '{print $2}')
            echo "$cycles" > "${RESULTS_DIR}/${name}_final_cycles.txt"
            echo "[$name] ✓ Completed with $cycles cycles"
        else
            echo "[$name] ✗ Failed to measure cycles"
            echo "FAILED" > "${RESULTS_DIR}/${name}_final_cycles.txt"
        fi

        exit $exit_code
    ) &

    local pid=$!
    echo $pid > "${RESULTS_DIR}/${name}_pid.txt"
    echo "[$name] Started with PID: $pid"
}

# Start all optimizations in parallel
echo "🔄 Starting parallel optimizations..."
echo ""

for approach prompt in ${(kv)APPROACHES}; do
    run_optimization "$approach" "$prompt"
    sleep 2  # Stagger starts to avoid race conditions
done

# Wait for all to complete
echo ""
echo "⏳ Waiting for all optimizations to complete..."
echo "   (max ${TIMEOUT_MINUTES} minutes each)"
echo "   Started at: $(date)"
echo ""

wait

echo ""
echo "   Completed at: $(date)"

# Analyze results
echo ""
echo "📊 Results Summary"
echo "=================================="

best_cycles=999999
best_approach=""
current_cycles=3394  # Known baseline

echo "Baseline (current main): $current_cycles cycles"
echo ""

for approach in ${(k)APPROACHES}; do
    worktree_path="${WORKTREE_BASE}-${approach}"
    cycles_file="${RESULTS_DIR}/${approach}_final_cycles.txt"
    result_file="${RESULTS_DIR}/${approach}_result.txt"

    if [ -f "$cycles_file" ]; then
        cycles=$(cat "$cycles_file")

        if [ "$cycles" = "FAILED" ]; then
            echo "[$approach] ✗ FAILED - see $result_file"
        else
            improvement=$((current_cycles - cycles))
            echo "[$approach] $cycles cycles (improvement: $improvement)"

            if [ "$cycles" -lt "$best_cycles" ] && [ "$cycles" -lt "$TARGET_CYCLES" ]; then
                best_cycles=$cycles
                best_approach=$approach
            fi
        fi
    else
        echo "[$approach] ⏸ No result (timeout or error) - see $result_file"
    fi
done

echo ""
echo "=================================="

# Merge winner if found
if [ -n "$best_approach" ]; then
    echo "🏆 Winner: $best_approach with $best_cycles cycles"
    echo ""
    echo "Merging winning approach to main..."

    git merge "opt-${best_approach}-${TIMESTAMP}" -m "Merge winning optimization: $best_approach ($best_cycles cycles)

Co-Authored-By: Claude Sonnet 4.5 <noreply@anthropic.com>"

    echo "✓ Merged successfully"

    # Update claude.md with new best
    sed -i.bak "s/\*\*[0-9]* cycles\*\* - Current best implementation/**${best_cycles} cycles** - Current best implementation/" claude.md
    git add claude.md
    git commit -m "Update claude.md with new best: ${best_cycles} cycles"

    echo "✓ Updated claude.md"
else
    echo "❌ No approach beat the target of ${TARGET_CYCLES} cycles"
    echo ""
    echo "Check individual results in:"
    for approach in ${(k)APPROACHES}; do
        result_file="${RESULTS_DIR}/${approach}_result.txt"
        if [ -f "$result_file" ]; then
            echo "  - $result_file"
        fi
    done
fi

# Cleanup
echo ""
echo "🧹 Cleaning up worktrees..."

for approach in ${(k)APPROACHES}; do
    worktree_path="${WORKTREE_BASE}-${approach}"
    if [ -d "$worktree_path" ]; then
        git worktree remove "$worktree_path" --force 2>/dev/null || true
        git branch -D "opt-${approach}-${TIMESTAMP}" 2>/dev/null || true
        echo "  ✓ Removed $approach worktree"
    fi
done

echo ""
echo "✓ Optimization complete!"
echo "Results saved in: $RESULTS_DIR"
