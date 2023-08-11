#!/bin/bash

# Define the range of residue numbers
start_residue=1
end_residue=56
num_residues_per_run=2

# Loop through the range of residue numbers
for ((i=start_residue; i<=end_residue; i=i+num_residues_per_run))
do
    end_residue_run=$((i+num_residues_per_run))
    if [ "$end_residue_run" -gt "$end_residue" ]; then
        end_residue_run="$end_residue"
    fi
    echo "Running extraction for residues $i to $end_residue_run"
    python3 official_extraction.py "official_confid_20ns_${i}-${end_residue_run}.json"
    #echo "config file official_confid_20ns_${i}-${end_residue_run}.json"
    #python3 official_extraction.py "official_confid_20ns_${i}-${end_residue_run}.json"
    echo "Finished running extraction for residues $i to $end_residue_run"
done


