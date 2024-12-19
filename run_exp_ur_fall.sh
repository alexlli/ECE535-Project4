#!/bin/bash

# Define an array with the names of 12 dccae configuration files 
# and two ablation configuration file
config_files=(
# acce_depth
    ./config/ur_fall/split_ae/acce_depth/A0_B0_AB30_label_AB_test_A
    ./config/ur_fall/split_ae/acce_depth/A0_B0_AB30_label_AB_test_B
    ./config/ur_fall/split_ae/acce_depth/A0_B0_AB30_label_A_test_B
    ./config/ur_fall/split_ae/acce_depth/A0_B0_AB30_label_B_test_A
    ./config/ur_fall/split_ae/acce_depth/A0_B10_AB30_label_A_test_B
    ./config/ur_fall/split_ae/acce_depth/A0_B10_AB30_label_B_test_A
    ./config/ur_fall/split_ae/acce_depth/A0_B30_AB0_label_B_test_B
    ./config/ur_fall/split_ae/acce_depth/A10_B0_AB30_label_A_test_B
    ./config/ur_fall/split_ae/acce_depth/A10_B0_AB30_label_B_test_A
    ./config/ur_fall/split_ae/acce_depth/A10_B10_AB30_label_A_test_B
    ./config/ur_fall/split_ae/acce_depth/A10_B10_AB30_label_B_test_A
    ./config/ur_fall/split_ae/acce_depth/A30_B0_AB0_label_A_test_A
    ./config/ur_fall/split_ae/rgb_depth/A0_B0_AB30_label_AB_test_A
    ./config/ur_fall/split_ae/rgb_depth/A0_B0_AB30_label_AB_test_B
    ./config/ur_fall/split_ae/rgb_depth/A0_B0_AB30_label_A_test_B
    ./config/ur_fall/split_ae/rgb_depth/A0_B0_AB30_label_B_test_A
    ./config/ur_fall/split_ae/rgb_depth/A0_B10_AB30_label_A_test_B
    ./config/ur_fall/split_ae/rgb_depth/A0_B10_AB30_label_B_test_A
    ./config/ur_fall/split_ae/rgb_depth/A0_B30_AB0_label_B_test_B
    ./config/ur_fall/split_ae/rgb_depth/A10_B0_AB30_label_A_test_B
    ./config/ur_fall/split_ae/rgb_depth/A10_B0_AB30_label_B_test_A
    ./config/ur_fall/split_ae/rgb_depth/A10_B10_AB30_label_A_test_B
    ./config/ur_fall/split_ae/rgb_depth/A10_B10_AB30_label_B_test_A
    ./config/ur_fall/split_ae/rgb_depth/A30_B0_AB0_label_A_test_A
    ./config/ur_fall/ablation/acce_depth/A30_B30_AB0_label_A_test_B
    ./config/ur_fall/ablation/acce_depth/A30_B30_AB0_label_B_test_A
    ./config/ur_fall/ablation/rgb_depth/A30_B30_AB0_label_A_test_B
    ./config/ur_fall/ablation/rgb_depth/A30_B30_AB0_label_B_test_A
)

# Loop through each file in the array
for file in "${config_files[@]}"; do
    if [[ -f "$file" ]]; then
        echo "Run model on $file  each for 100 rounds:"
        
        python3 src/main.py --config "$file"
                
    else
        echo "File $file does not exist."
    fi
done
