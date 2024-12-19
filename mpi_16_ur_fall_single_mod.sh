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
mpirun -n 16 python3 src/main.py --config ./config/ur_fall/split_ae/acce_depth/A30_B0_AB0_label_A_test_A 
echo "done 1"
mpirun -n 16 python3 src/main.py --config ./config/ur_fall/split_ae/acce_depth/A0_B30_AB0_label_B_test_B 
echo "done 2"

mpirun -n 16 python3 src/main.py --config ./config/ur_fall/split_ae/acce_depth/A0_B0_AB30_label_AB_test_A 
echo "done 3"

mpirun -n 16 python3 src/main.py --config ./config/ur_fall/split_ae/acce_depth/A0_B0_AB30_label_AB_test_B 
echo "done 4"

# run for rgb_depth
mpirun -n 16 python3 src/main.py --config ./config/ur_fall/split_ae/rgb_depth/A30_B0_AB0_label_A_test_A 
echo "done 5"

mpirun -n 16 python3 src/main.py --config ./config/ur_fall/split_ae/rgb_depth/A0_B30_AB0_label_B_test_B 
echo "done 6"

mpirun -n 16 python3 src/main.py --config ./config/ur_fall/split_ae/rgb_depth/A0_B0_AB30_label_AB_test_A 
echo "done 7"

mpirun -n 16 python3 src/main.py --config ./config/ur_fall/split_ae/rgb_depth/A0_B0_AB30_label_AB_test_B 

# 
date
echo "Run completed"

