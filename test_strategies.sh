#!/bin/bash
# Test all strategies

for strategy in direct progressive swpo tiled cpu_offload hybrid auto; do
    echo "Testing strategy: $strategy"
    python -m expandor /home/user/Pictures/backgrounds/42258.jpg \
        --resolution 2x --strategy $strategy --dry-run 2>&1 | \
        grep -E "(ERROR.*Invalid strategy|Would process|DRY RUN COMPLETE)" || \
        echo "FAILED: $strategy"
done