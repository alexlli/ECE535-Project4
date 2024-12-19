#!/bin/bash

#run these four experiements each 4 runs, each run consists of 100 rounds
#run for combos in the following array
# 
#combo = {
#     "UR_Fall": [("acce", "depth"), ("rgb", "depth")]
#}

#each is running in the 'base' environment

N=16

date
mpirun -n $N python3 src/main.py --config ./config/ur_fall/split_ae/acce_depth/A0_B0_AB30_label_A_test_B
date
mpirun -n $N python3 src/main.py --config ./config/ur_fall/split_ae/acce_depth/A0_B0_AB30_label_B_test_A
date
mpirun -n $N python3 src/main.py --config ./config/ur_fall/split_ae/acce_depth/A0_B10_AB30_label_A_test_B
date
mpirun -n $N python3 src/main.py --config ./config/ur_fall/split_ae/acce_depth/A0_B10_AB30_label_B_test_A
date
mpirun -n $N python3 src/main.py --config ./config/ur_fall/split_ae/acce_depth/A10_B0_AB30_label_A_test_B
date
mpirun -n $N python3 src/main.py --config ./config/ur_fall/split_ae/acce_depth/A10_B0_AB30_label_B_test_A
date
mpirun -n $N python3 src/main.py --config ./config/ur_fall/split_ae/acce_depth/A10_B10_AB30_label_A_test_B
date
mpirun -n $N python3 src/main.py --config ./config/ur_fall/split_ae/acce_depth/A10_B10_AB30_label_B_test_A
date
mpirun -n $N python3 src/main.py --config ./config/ur_fall/split_ae/rgb_depth/A0_B0_AB30_label_A_test_B
date
mpirun -n $N python3 src/main.py --config ./config/ur_fall/split_ae/rgb_depth/A0_B0_AB30_label_B_test_A
date
mpirun -n $N python3 src/main.py --config ./config/ur_fall/split_ae/rgb_depth/A0_B10_AB30_label_A_test_B
date
mpirun -n $N python3 src/main.py --config ./config/ur_fall/split_ae/rgb_depth/A0_B10_AB30_label_B_test_A
date
mpirun -n $N python3 src/main.py --config ./config/ur_fall/split_ae/rgb_depth/A10_B0_AB30_label_A_test_B
date
mpirun -n $N python3 src/main.py --config ./config/ur_fall/split_ae/rgb_depth/A10_B0_AB30_label_B_test_A
date
mpirun -n $N python3 src/main.py --config ./config/ur_fall/split_ae/rgb_depth/A10_B10_AB30_label_A_test_B
date
mpirun -n $N python3 src/main.py --config ./config/ur_fall/split_ae/rgb_depth/A10_B10_AB30_label_B_test_A
date
mpirun -n $N python3 src/main.py --config ./config/ur_fall/ablation/acce_depth/A30_B30_AB0_label_A_test_B
date
mpirun -n $N python3 src/main.py --config ./config/ur_fall/ablation/acce_depth/A30_B30_AB0_label_B_test_A
date
mpirun -n $N python3 src/main.py --config ./config/ur_fall/ablation/rgb_depth/A30_B30_AB0_label_A_test_B
date
mpirun -n $N python3 src/main.py --config ./config/ur_fall/ablation/rgb_depth/A30_B30_AB0_label_B_test_A
date

echo "Run completed"

