#!/bin/zsh
# Continuous optimization loop

echo "🚀 Continuous Optimizer Started"
echo "Goal: < 1487 cycles, Ultimate: ~1000 cycles"
echo "Current baseline: 3490 cycles"
echo ""

while true; do
    echo "═══════════════════════════════════════════════"
    echo "$(date '+%H:%M:%S') - Checking task status..."
    
    # Check if any tasks are still running
    running=0
    completed_with_results=0
    
    for task_dir in .parallel_opt_state/*/; do
        [ -d "$task_dir" ] || continue
        name=$(basename "$task_dir")
        status=$(cat "$task_dir/status.txt" 2>/dev/null || echo "running")
        
        if [ "$status" = "running" ]; then
            ((running++))
        elif [ "$status" = "success" ]; then
            cycles=$(cat "$task_dir/cycles.txt" 2>/dev/null)
            echo "🎉 SUCCESS: $name -> $cycles cycles!"
            ((completed_with_results++))
        fi
    done
    
    echo "Running: $running, Successful: $completed_with_results"
    
    # If nothing running, exit so Claude can analyze and start next batch
    if [ $running -eq 0 ]; then
        echo "✓ Batch complete. Exiting for analysis..."
        break
    fi
    
    sleep 30
done
