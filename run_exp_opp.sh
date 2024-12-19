#!/bin/bash

# Define an array with the names of 12 dccae configuration files 
# and two ablation configuration file
config_files=(
    "./config/opp/dccae/A0_B0_AB30_label_AB_test_A"
    "./config/opp/dccae/A0_B0_AB30_label_AB_test_B"
    "./config/opp/dccae/A0_B0_AB30_label_A_test_B"
    "./config/opp/dccae/A0_B0_AB30_label_B_test_A"
    "./config/opp/dccae/A0_B10_AB30_label_A_test_B"
    "./config/opp/dccae/A0_B10_AB30_label_B_test_A"
    "./config/opp/dccae/A0_B30_AB0_label_B_test_B"
    "./config/opp/dccae/A10_B0_AB30_label_A_test_B"
    "./config/opp/dccae/A10_B0_AB30_label_B_test_A"
    "./config/opp/dccae/A10_B10_AB30_label_A_test_B"
    "./config/opp/dccae/A10_B10_AB30_label_B_test_A"
    "./config/opp/dccae/A30_B0_AB0_label_A_test_A"
    "./config/opp/ablation/A30_B30_AB0_label_A_test_B"
    "./config/opp/ablation/A30_B30_AB0_label_B_test_A"
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
