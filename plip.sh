#!/bin/bash

# Define the target directories and the chain ID
INPUT_DIR="AT1R_Rep1_PDB_Frames"
OUTPUT_DIR="AT1R_Rep1_PLIP_Results"
CHAIN_ID="X" 

# Create the output directory to keep things organized
mkdir -p "$OUTPUT_DIR"

echo "Starting PLIP batch processing..."

# Loop through every single .pdb file in the input directory
for pdb_file in "$INPUT_DIR"/*.pdb; do

    # Extract just the filename without the path and the .pdb extension
    # e.g., converts "AT1R_Rep1_PDB_Frames/frame_0000.pdb" into "frame_0000"
    basename_str=$(basename "$pdb_file" .pdb)
    
    # Run the PLIP engine
    plip -f "$pdb_file" \
         -x \
         -t \
         --intra "$CHAIN_ID" \
         --name "$basename_str" \
         -o "$OUTPUT_DIR" \
         -q

    # Print a quick status update to the terminal
    echo "Processed: $basename_str"
    
done

echo "PLIP analysis complete! Results saved in $OUTPUT_DIR"
