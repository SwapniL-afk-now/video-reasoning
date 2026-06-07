#!/bin/bash
PID=67566
OUTPUT_DIR="/workspace/output"

echo "=== Monitoring build_vkg.py (PID: $PID) ==="
echo "Started at: $(date)"
echo ""

while true; do
    TIMESTAMP=$(date '+%H:%M:%S')
    
    if ps -p $PID > /dev/null 2>&1; then
        # Process stats
        STATS=$(ps -p $PID -o %cpu,%mem,etime --no-headers 2>/dev/null)
        CPU=$(echo $STATS | awk '{print $1}')
        MEM=$(echo $STATS | awk '{print $2}')
        ELAPSED=$(echo $STATS | awk '{print $3}')
        
        # GPU usage
        GPU_INFO=$(nvidia-smi --query-gpu=utilization.gpu,memory.used --format=csv,noheader 2>/dev/null || echo "N/A")
        
        # Output directory size
        DIR_SIZE=$(du -sh $OUTPUT_DIR 2>/dev/null | cut -f1)
        
        # Check for any new files
        NEW_FILES=$(find $OUTPUT_DIR -type f -newer /tmp/monitor_start 2>/dev/null | wc -l)
        
        echo "[$TIMESTAMP] CPU: ${CPU}% | Mem: ${MEM}% | Elapsed: $ELAPSED | GPU: $GPU_INFO | Size: $DIR_SIZE | New files: $NEW_FILES"
    else
        echo "[$TIMESTAMP] Process completed!"
        echo "Final output:"
        ls -lh $OUTPUT_DIR/
        break
    fi
    
    sleep 10
done
