import json

start_residue_run = 0
end_residue_run = 56
num_residues_per_run = 8

input_file_path = "official_config.json"
output_file_path_template = "official_config_20ns_{}.json"

with open(input_file_path, "r") as json_file:
    existing_data = json.load(json_file)

start_residue = existing_data.get("start_residue", 1)
end_residue = existing_data.get("end_residue", 1)

for start_residue in range(start_residue_run, end_residue_run, num_residues_per_run):
    data = existing_data.copy()
    end_residue = start_residue + num_residues_per_run
    if end_residue > end_residue_run:
        end_residue = end_residue_run
    start_residue = start_residue + 1
    data["start_residue"] = start_residue
    data["end_residue"] = end_residue
    data["num_residues"] = num_residues_per_run
    output_file_suffix = str(start_residue) + "-" + str(end_residue)
    output_csv = "/home/annika/md_sims/official_extraction/convergence_values_20_" + output_file_suffix + ".csv"
    data["new_csv_name"] = output_csv

    output_file_path = output_file_path_template.format(output_file_suffix)

    with open(output_file_path, "w") as json_file:
        json.dump(data, json_file, indent=4)

    print(f"JSON file created for residue {start_residue} successfully: {output_file_path}")
