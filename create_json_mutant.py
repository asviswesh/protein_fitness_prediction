import argparse
import json

parser = argparse.ArgumentParser()
parser.add_argument("md_filename")
args = parser.parse_args()

input_file_path = "official_config_20ns_all.json"
output_file_path_template = "/home/annika/md_sims/official_extraction/official_config_20ns_{}.json"

with open(input_file_path, "r") as json_file:
    existing_data = json.load(json_file)

data = existing_data.copy()
md_name = args.md_filename
data["md_mutant_name"] = md_name
output_csv = "/home/annika/md_sims/official_extraction/convergence_values_20_" + md_name + ".csv"
data["new_csv_name"] = output_csv

output_file_path = output_file_path_template.format(md_name)
print(output_file_path)

with open(output_file_path, "w") as json_file:
    json.dump(data, json_file, indent=4)

print(f"JSON file created for mutant {md_name}")
