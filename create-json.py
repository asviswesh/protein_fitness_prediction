import json

#starting residue
start_residue_run = 1
#ending residue
end_residue_run = 56
#number of residues in each run
num_residues_per_run = 4

# Specify the file paths
input_file_path = "official_config_20ns.json"
output_file_path_template = "official_config_20ns_{}.json"

# Read the existing JSON data from the input file
with open(input_file_path, "r") as json_file:
    existing_data = json.load(json_file)

# Extract the existing values
start_residue = existing_data.get("start_residue", 1)
end_residue = existing_data.get("end_residue", 1)


# Iterate over the desired range (1 to 56, inclusive)
for start_residue in range(start_residue_run, end_residue_run, num_residues_per_run):
    
    # Create a copy of the existing data to modify for each age
    data = existing_data.copy()
    end_residue = start_residue + num_residues_per_run
    if end_residue > end_residue_run:
        end_residue = end_residue_run
    data["start_residue"] = start_residue
    data["end_residue"] = end_residue
    output_file_suffix = str(start_residue) + "-" + str(end_residue)
    output_csv = "/home/annika/md_sims/official_extraction/convergence_values_20_" + output_file_suffix + ".csv"
    data["new_csv_name"] = output_csv

    # Create the file path with the age in the name
    output_file_path = output_file_path_template.format(output_file_suffix)

    # Write the data to the new JSON file
    with open(output_file_path, "w") as json_file:
        json.dump(data, json_file, indent=4)

    print(f"JSON file created for residue {start_residue} successfully: {output_file_path}")
