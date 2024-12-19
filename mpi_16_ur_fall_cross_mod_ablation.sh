#!/bin/bash

N=16

mpirun -n $N python3 src/main.py --config ./config/ur_fall/ablation/acce_depth/A30_B30_AB0_label_A_test_B
date
mpirun -n $N python3 src/main.py --config ./config/ur_fall/ablation/acce_depth/A30_B30_AB0_label_B_test_A
date
mpirun -n $N python3 src/main.py --config ./config/ur_fall/ablation/rgb_depth/A30_B30_AB0_label_A_test_B
date
mpirun -n $N python3 src/main.py --config ./config/ur_fall/ablation/rgb_depth/A30_B30_AB0_label_B_test_A
date

echo "Run completed"
