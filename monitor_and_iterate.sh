#!/bin/zsh
# Continuous monitoring and iteration script

STATE_DIR=".parallel_opt_state"
CHECK_INTERVAL=30  # seconds

echo "🔄 Starting continuous optimization monitor..."
echo "Goal: < 1487 cycles (target: ~1000 cycles)"
echo ""

iteration=1

while true; do
    echo "═══════════════════════════════════════════"
    echo "Iteration $iteration - $(date '+%H:%M:%S')"
    echo "═══════════════════════════════════════════"

    # Check status of all tasks
    running=0
    completed=0
    success=0

    if [ -d "$STATE_DIR" ]; then
        for task_dir in "$STATE_DIR"/*; do
            if [ -d "$task_dir" ]; then
                name=$(basename "$task_dir")
                status=$(cat "$task_dir/status.txt" 2>/dev/null || echo "running")

                if [ "$status" = "running" ]; then
                    ((running++))
                elif [ "$status" = "completed" ] || [ "$status" = "failed" ]; then
                    ((completed++))
                elif [ "$status" = "success" ]; then
                    ((success++))
                    cycles=$(cat "$task_dir/cycles.txt" 2>/dev/null)
                    echo "🎉 SUCCESS: $name achieved $cycles cycles!"
                fi
            fi
        done
    fi

    echo "Status: $running running, $completed completed, $success successful"

    # If all tasks are done, this script should exit (Claude will analyze and launch new ones)
    if [ $running -eq 0 ] && [ -d "$STATE_DIR" ] && [ -n "$(ls -A $STATE_DIR 2>/dev/null)" ]; then
        echo ""
        echo "✓ All tasks completed. Ready for analysis and next iteration."
        break
    fi

    sleep $CHECK_INTERVAL
    ((iteration++))
done
