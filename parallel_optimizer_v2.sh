#!/bin/zsh
# Dynamic Parallel Optimization Session Manager
# Usage:
#   ./parallel_optimizer_v2.sh add <name> <target_cycles> "<prompt>"
#   ./parallel_optimizer_v2.sh status
#   ./parallel_optimizer_v2.sh wait

set -e

PYTHON_CMD="/opt/homebrew/bin/python3.12"
TIMEOUT_CMD="gtimeout"
TIMEOUT_MINUTES=60
MAIN_DIR="$(pwd)"
WORKTREE_BASE="${MAIN_DIR}/../perf-opt"
STATE_DIR="${MAIN_DIR}/.parallel_opt_state"
TIMESTAMP=$(date +%Y%m%d_%H%M%S)

# Initialize state directory
mkdir -p "$STATE_DIR"

# Color codes for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

print_status() {
    echo -e "${BLUE}[$(date '+%H:%M:%S')]${NC} $1"
}

print_success() {
    echo -e "${GREEN}✓${NC} $1"
}

print_error() {
    echo -e "${RED}✗${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}⚠${NC} $1"
}

# Add a new optimization approach
add_approach() {
    local name=$1
    local target_cycles=$2
    local prompt=$3

    if [ -z "$name" ] || [ -z "$target_cycles" ] || [ -z "$prompt" ]; then
        print_error "Usage: add <name> <target_cycles> \"<prompt>\""
        exit 1
    fi

    # Sanitize name
    local safe_name=$(echo "$name" | tr ' ' '-' | tr -cd '[:alnum:]-_')
    local worktree_path="${WORKTREE_BASE}-${safe_name}"
    local approach_dir="${STATE_DIR}/${safe_name}"

    # Check if already running
    if [ -f "${approach_dir}/pid.txt" ]; then
        local pid=$(cat "${approach_dir}/pid.txt")
        if ps -p $pid > /dev/null 2>&1; then
            print_warning "Approach '$safe_name' is already running (PID: $pid)"
            return 0
        fi
    fi

    mkdir -p "$approach_dir"

    print_status "Starting optimization: $safe_name"
    print_status "Target: < $target_cycles cycles"

    # Create worktree with new branch
    local branch_name="opt-${safe_name}-${TIMESTAMP}"

    if [ -d "$worktree_path" ]; then
        print_warning "Cleaning up existing worktree..."
        git worktree remove "$worktree_path" --force 2>/dev/null || true
    fi

    git worktree add "$worktree_path" -b "$branch_name" 2>&1 | tee "${approach_dir}/setup.log"

    # Save approach metadata
    echo "$target_cycles" > "${approach_dir}/target.txt"
    echo "$prompt" > "${approach_dir}/prompt.txt"
    echo "$branch_name" > "${approach_dir}/branch.txt"
    echo "$worktree_path" > "${approach_dir}/worktree.txt"
    echo "running" > "${approach_dir}/status.txt"

    # Run claude with timeout in background
    (
        cd "$worktree_path"

        # Run optimization
        $TIMEOUT_CMD ${TIMEOUT_MINUTES}m claude -p "$prompt" > "${approach_dir}/output.txt" 2>&1
        exit_code=$?

        echo "Exit code: $exit_code" >> "${approach_dir}/output.txt"

        # Measure cycles
        if $PYTHON_CMD tests/submission_tests.py 2>&1 | tee -a "${approach_dir}/output.txt" | grep -o "CYCLES:  [0-9]*" | tail -1 > "${approach_dir}/cycles_raw.txt"; then
            cycles=$(cat "${approach_dir}/cycles_raw.txt" | awk '{print $2}')
            echo "$cycles" > "${approach_dir}/cycles.txt"

            if [ "$cycles" -lt "$target_cycles" ]; then
                echo "success" > "${approach_dir}/status.txt"
                print_success "[$safe_name] Achieved $cycles cycles (target: < $target_cycles)"
            else
                echo "completed" > "${approach_dir}/status.txt"
                print_warning "[$safe_name] Got $cycles cycles (missed target: < $target_cycles)"
            fi
        else
            echo "failed" > "${approach_dir}/status.txt"
            print_error "[$safe_name] Failed to measure cycles"
        fi

        # Record completion time
        date +%s > "${approach_dir}/completed_at.txt"

    ) &

    local pid=$!
    echo $pid > "${approach_dir}/pid.txt"
    echo $(date +%s) > "${approach_dir}/started_at.txt"

    print_success "Started '$safe_name' (PID: $pid)"
    print_status "Results: ${approach_dir}/output.txt"
}

# Show status of all running optimizations
show_status() {
    echo ""
    echo "═══════════════════════════════════════════════════════════"
    echo "  Parallel Optimization Status"
    echo "═══════════════════════════════════════════════════════════"
    echo ""

    local running=0
    local succeeded=0
    local failed=0
    local completed=0

    if [ ! -d "$STATE_DIR" ] || [ -z "$(ls -A $STATE_DIR 2>/dev/null)" ]; then
        print_warning "No optimizations running"
        return
    fi

    for approach_dir in "$STATE_DIR"/*; do
        if [ ! -d "$approach_dir" ]; then
            continue
        fi

        local name=$(basename "$approach_dir")
        local opt_status=$(cat "${approach_dir}/status.txt" 2>/dev/null || echo "unknown")
        local pid=$(cat "${approach_dir}/pid.txt" 2>/dev/null || echo "?")
        local target=$(cat "${approach_dir}/target.txt" 2>/dev/null || echo "?")
        local cycles=$(cat "${approach_dir}/cycles.txt" 2>/dev/null || echo "?")

        # Check if process still running
        if [ "$opt_status" = "running" ] && ! ps -p $pid > /dev/null 2>&1; then
            opt_status="crashed"
            echo "crashed" > "${approach_dir}/status.txt"
        fi

        # Format output based on status
        case "$opt_status" in
            running)
                local started=$(cat "${approach_dir}/started_at.txt")
                local elapsed=$(( $(date +%s) - started ))
                local mins=$(( elapsed / 60 ))
                echo -e "${BLUE}⏳${NC} ${name} (PID: $pid) - Running for ${mins}m"
                echo "   Target: < $target cycles"
                ((running++))
                ;;
            success)
                echo -e "${GREEN}✓${NC} ${name} - ${cycles} cycles (target: < $target)"
                ((succeeded++))
                ;;
            completed)
                echo -e "${YELLOW}○${NC} ${name} - ${cycles} cycles (missed target: < $target)"
                ((completed++))
                ;;
            failed|crashed)
                echo -e "${RED}✗${NC} ${name} - Failed (see ${approach_dir}/output.txt)"
                ((failed++))
                ;;
        esac
    done

    echo ""
    echo "───────────────────────────────────────────────────────────"
    echo "Summary: $running running, $succeeded succeeded, $completed completed, $failed failed"
    echo "═══════════════════════════════════════════════════════════"
    echo ""
}

# Wait for all optimizations and report best
wait_and_report() {
    print_status "Waiting for all optimizations to complete..."

    while true; do
        local still_running=0

        for approach_dir in "$STATE_DIR"/*; do
            if [ ! -d "$approach_dir" ]; then
                continue
            fi

            local opt_status=$(cat "${approach_dir}/status.txt" 2>/dev/null || echo "unknown")
            local pid=$(cat "${approach_dir}/pid.txt" 2>/dev/null || echo "0")

            if [ "$opt_status" = "running" ] && ps -p $pid > /dev/null 2>&1; then
                ((still_running++))
            fi
        done

        if [ $still_running -eq 0 ]; then
            break
        fi

        sleep 10
    done

    print_success "All optimizations completed!"
    echo ""

    # Find best result
    local best_cycles=999999
    local best_approach=""
    local best_branch=""

    for approach_dir in "$STATE_DIR"/*; do
        if [ ! -d "$approach_dir" ]; then
            continue
        fi

        local name=$(basename "$approach_dir")
        local opt_status=$(cat "${approach_dir}/status.txt" 2>/dev/null || echo "unknown")
        local cycles=$(cat "${approach_dir}/cycles.txt" 2>/dev/null || echo "999999")
        local branch=$(cat "${approach_dir}/branch.txt" 2>/dev/null || echo "")

        if [ "$opt_status" = "success" ] && [ "$cycles" -lt "$best_cycles" ]; then
            best_cycles=$cycles
            best_approach=$name
            best_branch=$branch
        fi
    done

    # Report and merge winner
    if [ -n "$best_approach" ]; then
        echo ""
        echo "🏆 Winner: $best_approach with $best_cycles cycles"
        echo ""
        print_status "Merging winning approach to main..."

        git merge "$best_branch" -m "Merge winning optimization: $best_approach ($best_cycles cycles)

Co-Authored-By: Claude Sonnet 4.5 <noreply@anthropic.com>"

        print_success "Merged successfully"

        # Update claude.md
        local speedup=$(echo "scale=1; 147824 / $best_cycles" | bc)
        sed -i.bak "s/\*\*[0-9]* cycles\*\* - Current best implementation/**${best_cycles} cycles** - Current best implementation/" claude.md
        sed -i.bak "s/Speedup over baseline: \*\*[0-9.]*x\*\*/Speedup over baseline: **${speedup}x**/" claude.md
        git add claude.md
        git commit -m "Update claude.md with new best: ${best_cycles} cycles"

        print_success "Updated claude.md"
    else
        print_warning "No approach succeeded"
    fi

    # Show final status
    show_status
}

# Cleanup completed optimizations
cleanup() {
    print_status "Cleaning up worktrees..."

    for approach_dir in "$STATE_DIR"/*; do
        if [ ! -d "$approach_dir" ]; then
            continue
        fi

        local name=$(basename "$approach_dir")
        local worktree=$(cat "${approach_dir}/worktree.txt" 2>/dev/null || echo "")
        local branch=$(cat "${approach_dir}/branch.txt" 2>/dev/null || echo "")

        if [ -n "$worktree" ] && [ -d "$worktree" ]; then
            git worktree remove "$worktree" --force 2>/dev/null || true
            print_success "Removed $name worktree"
        fi

        if [ -n "$branch" ]; then
            git branch -D "$branch" 2>/dev/null || true
        fi

        rm -rf "$approach_dir"
    done

    print_success "Cleanup complete"
}

# Main command dispatcher
case "${1:-help}" in
    add)
        add_approach "$2" "$3" "$4"
        ;;
    status)
        show_status
        ;;
    wait)
        wait_and_report
        ;;
    cleanup)
        cleanup
        ;;
    help|*)
        echo "Dynamic Parallel Optimization Manager"
        echo ""
        echo "Usage:"
        echo "  $0 add <name> <target_cycles> \"<prompt>\""
        echo "      Add and start a new optimization approach"
        echo ""
        echo "  $0 status"
        echo "      Show status of all running optimizations"
        echo ""
        echo "  $0 wait"
        echo "      Wait for all to complete and merge winner"
        echo ""
        echo "  $0 cleanup"
        echo "      Clean up all worktrees and state"
        echo ""
        echo "Examples:"
        echo "  $0 add aggressive-cache 3300 \"Implement aggressive pre-caching...\""
        echo "  $0 status"
        echo "  $0 wait"
        ;;
esac
