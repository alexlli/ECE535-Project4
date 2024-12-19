#!/bin/bash

#run these four experiements each 4 runs, each run consists of 100 rounds
#run for combos in the following array
# 
#combo = {
#     "UR_Fall": [("acce", "depth"), ("rgb", "depth")]
#}


#schemes = {"client_A_label_A_test_A": "A30_B0_AB0_label_A_test_A",
#           "client_B_label_B_test_B": "A0_B30_AB0_label_B_test_B",
#           "client_AB_label_AB_test_A": "A0_B0_AB30_label_AB_test_A",
#           "client_AB_label_AB_test_B": "A0_B0_AB30_label_AB_test_B", }

#each is running in the 'base' environment
# Load the necessary modules, not needed if scripts started in 'base' environment
#module load anaconda3/personal
#source activate base
#module load mpi

# print out stating time
date
# Run the MPI job -np is the official flag -n is the equivalent
# run for acce_depth
mpirun -n 64 python3 src/main.py --config ./config/ur_fall/split_ae/acce_depth/A30_B0_AB0_label_A_test_A &
mpirun -n 64 python3 src/main.py --config ./config/ur_fall/split_ae/acce_depth/A0_B30_AB0_label_B_test_B &
mpirun -n 64 python3 src/main.py --config ./config/ur_fall/split_ae/acce_depth/A0_B0_AB30_label_AB_test_A &
mpirun -n 64 python3 src/main.py --config ./config/ur_fall/split_ae/acce_depth/A0_B0_AB30_label_AB_test_B &

# run for rgb_depth
mpirun -n 64 python3 src/main.py --config ./config/ur_fall/split_ae/rgb_depth/A30_B0_AB0_label_A_test_A &
mpirun -n 64 python3 src/main.py --config ./config/ur_fall/split_ae/rgb_depth/A0_B30_AB0_label_B_test_B &
mpirun -n 64 python3 src/main.py --config ./config/ur_fall/split_ae/rgb_depth/A0_B0_AB30_label_AB_test_A &
mpirun -n 64 python3 src/main.py --config ./config/ur_fall/split_ae/rgb_depth/A0_B0_AB30_label_AB_test_B &

#wait for all processes to finish
wait

# 
date
echo "Run completed"

