import argparse
import json
import re

WILD_TYPE_SEQUENCE = "VDGV"
INPUT_FILE_PATH = "official_config_20ns_all.json"

def create_combo_string(sample_file):
    specific_numbers = ['39', '40', '41', '54']
    split_list = sample_file.split('_')
    combo_string = ""
    for number in specific_numbers:
        found_num = False
        for string in split_list:
            if number in string:
                found_num = True
                combo_string += string[2]
                break
        if not found_num:
            replace_index = specific_numbers.index(number)
            combo_string += WILD_TYPE_SEQUENCE[replace_index]
    return combo_string

parser = argparse.ArgumentParser()
parser.add_argument("md_filename")
args = parser.parse_args()

output_file_path_template = "/home/annika/md_sims/official_extraction/official_config_20ns_{}.json"

with open(INPUT_FILE_PATH, "r") as json_file:
    existing_data = json.load(json_file)

data = existing_data.copy()
md_name = args.md_filename
data["md_mutant_filename"] = md_name
md_combo = create_combo_string(md_name)
data["combo"] = md_combo
output_csv = "/home/annika/md_sims/official_extraction/convergence_values_20_" + md_combo + ".csv"
data["new_csv_name"] = output_csv

output_file_path = output_file_path_template.format(md_combo)

with open(output_file_path, "w") as json_file:
    json.dump(data, json_file, indent=4)

print(f"JSON file created for mutant {md_combo}")
