import argparse
import csv

def create_train_dataset(base_path, sheet_name):
    csv_path = base_path + sheet_name
    if '.csv' not in sheet_name:
        raise ValueError("Data is not in the .csv file.")
    with open(csv_path, 'w') as train_csv:
        col_names = ['Variants', 'Fitness', 'sequence']
        writer = csv.DictWriter(train_csv, fieldnames=col_names)
        writer.writeheader()
        with open('/home/annika/mlde/four_mutations_full_data.csv', 'r') as full_data:
            reader_obj = csv.DictReader(full_data)
            for line in reader_obj:
                if line.get('HD') == '1':
                    line.pop('HD')
                    line.pop('Count input')
                    line.pop('Count selected')
                    line.pop('keep')
                    line.pop('one_vs_rest')
                    line.pop('one_vs_rest_validation')
                    line.pop('two_vs_rest')
                    line.pop('two_vs_rest_validation')
                    line.pop('three_vs_rest')
                    line.pop('three_vs_rest_validation')
                    line.pop('sampled')
                    line.pop('sampled_validation')
                    line.pop('low_vs_high')
                    line.pop('low_vs_high_validation')
                    writer.writerow(line)


def create_test_dataset(base_path, sheet_name, need_fitness):
    base_path = '/home/annika/mlde/'
    csv_path = base_path + sheet_name
    if '.csv' not in sheet_name:
        raise ValueError("Must specify .csv when giving filename.")
    with open(csv_path, 'w') as test_csv:
        col_names = ['Variants', 'Fitness', 'sequence']
        writer = csv.DictWriter(test_csv, fieldnames=col_names)
        writer.writeheader()
        with open('/home/annika/mlde/four_mutations_full_data.csv', 'r') as full_data:
            reader_obj = csv.DictReader(full_data)
            for line in reader_obj:
                if line.get('HD') != '0' or line.get('HD') == 1:
                    if need_fitness:
                        line.pop("Fitness")
                    line.pop('HD')
                    line.pop('Count input')
                    line.pop('Count selected')
                    line.pop('keep')
                    line.pop('one_vs_rest')
                    line.pop('one_vs_rest_validation')
                    line.pop('two_vs_rest')
                    line.pop('two_vs_rest_validation')
                    line.pop('three_vs_rest')
                    line.pop('three_vs_rest_validation')
                    line.pop('sampled')
                    line.pop('sampled_validation')
                    line.pop('low_vs_high')
                    line.pop('low_vs_high_validation')
                    writer.writerow(line)

parser = argparse.ArgumentParser()
parser.add_argument("base_path")
args = parser.parse_args()
base_path = args.base_path

create_train_dataset(base_path, 'gb1_train.csv')
create_test_dataset(base_path, 'gb1_test.csv', True)
create_test_dataset(base_path, 'gb1_test_with_fitness.csv', False)
