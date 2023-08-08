import csv
import esm
import matplotlib.pyplot as plt
import subprocess
import torch

model, alphabet = esm.pretrained.esm2_t33_650M_UR50D()
batch_converter = alphabet.get_batch_converter()
model.eval()

subprocess.run(["python3", "/home/annika/mlde/create_train_test_csv.py"])

def generate_embeddings(train_or_test: str):
    data = []
    if train_or_test == 'train':
        with open('/home/annika/mlde/gb1_train.csv', 'r') as train_csv:
            train_reader = csv.DictReader(train_csv)
            for line in train_reader:
                data.append((line['Variants'], line['sequence']))
    elif train_or_test == 'test':
        with open('/home/annika/mlde/gb1_test.csv', 'r') as train_csv:
            test_reader = csv.DictReader(train_csv)
            for line in test_reader:
                data.append((line['Variants'], line['sequence']))

    _, _, batch_tokens = batch_converter(data)
    batch_lens = (batch_tokens != alphabet.padding_idx).sum(1)

    with torch.no_grad():
        results = model(batch_tokens, repr_layers=[33], return_contacts=True)
    token_representations = results["representations"][33]

    for i, tokens_len in enumerate(batch_lens):
        representation = token_representations[i, 1: tokens_len - 1].mean(0)
        torch.save(
            representation, f'/home/annika/mlde/esm_embeddings/{train_or_test}/{data[i][0]}.npy')

generate_embeddings('train')
generate_embeddings('test')
