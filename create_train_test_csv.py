import csv


def get_rows_for_csv(csv_file):
    with open(csv_file) as csv_file:
        csv_reader = csv.reader(csv_file, delimiter=',')
        list_of_column_names = []
        for row in csv_reader:
            list_of_column_names.append(row)
            break
    return list_of_column_names[0]


def create_train_dataset(sheet_name):
    base_path = '/home/annika/mlde/'
    csv_path = base_path + sheet_name
    if '.csv' not in sheet_name:
        raise ValueError("Data is not in the .csv file.")
    with open(csv_path, 'w') as train_csv:
        col_names = get_rows_for_csv(
            '/home/annika/mlde/four_mutations_full_data.csv')
        writer = csv.DictWriter(train_csv, fieldnames=col_names[:6])
        writer.writeheader()
        with open('/home/annika/mlde/four_mutations_full_data.csv', 'r') as full_data:
            reader_obj = csv.DictReader(full_data)
            for line in reader_obj:
                if line.get('HD') == '1':
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


def create_test_dataset(sheet_name):
    base_path = '/home/annika/mlde/'
    csv_path = base_path + sheet_name
    if '.csv' not in sheet_name:
        raise ValueError("Data is not in the .csv file.")
    with open(csv_path, 'w') as test_csv:
        col_names = get_rows_for_csv(
            '/home/annika/mlde/four_mutations_full_data.csv')
        writer = csv.DictWriter(test_csv, fieldnames=col_names[:6])
        writer.writeheader()
        with open('/home/annika/mlde/four_mutations_full_data.csv', 'r') as full_data:
            reader_obj = csv.DictReader(full_data)
            for line in reader_obj:
                if line.get('HD') == '2':
                    line.pop("Fitness")
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


create_train_dataset('gb1_train.csv')
create_test_dataset('gb1_test.csv')
