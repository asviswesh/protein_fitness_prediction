import pickle
import json
import os
import random
import numpy as np
import pandas as pd
import torch
import torch.utils.data
import torch.nn as nn
from feed_forward_draft import NeuralNet
from pytorchtools import EarlyStopping
from tqdm.auto import tqdm
from sklearn.metrics import ndcg_score
from sklearn.model_selection import train_test_split, KFold
from models import *
from Datasets import Dataset

# from DeCOIL.src.oracle import Oracle
# from DeCOIL.src.encoding_utils import *


def ndcg(y_true, y_pred):
    y_true_normalized = y_true - min(y_true)
    return ndcg_score(y_true_normalized.reshape(1, -1), y_pred.reshape(1, -1))


class MLDESim():
    """Class for training and evaluating MLDE models."""

    def __init__(self, save_path: str, encoding: str, model_class: str, n_samples: int, train_name: str, test_name: str, validation_name: str, first_append: bool, feat_to_predict: str, neural_network: bool) -> None:
        """
        Args:
            save_path : path to save results
            encoding : encoding type
            model_class : model class
            n_samples : number of samples to train on
            train_name: name of train dataset
            test_name: name of test dataset
            validation_name: type of validation to use on the dataset
            first_append: if we need to create a brand new csv file or not.
        """
        self.feat_to_predict = feat_to_predict
        self.train_name = train_name
        self.test_name = test_name
        self.validation_name = validation_name
        self.neural_network = neural_network
        self.save_path = save_path
        self.num_workers = 1

        self.n_splits = 5
        self.n_subsets = 24
        self.n_samples = n_samples

        self.n_solutions = 31

        self.save_model = True
        self.first_append = first_append

        self.model_class = model_class
        self.means = np.zeros((self.n_solutions, self.n_subsets))

        # Sample and fix a random seed
        self.seed = 42
        np.random.seed(self.seed)
        random.seed(self.seed)

        self.train_fitness_df = pd.read_csv(
            '/home/annika/mlde/' + self.train_name)
        self.test_fitness_df = pd.read_csv(
            '/home/annika/mlde/' + self.test_name)
        self.train_dataset = Dataset(
            dataframe=self.train_fitness_df, dataset_type="train", to_predict=self.feat_to_predict)
        self.test_dataset = Dataset(
            dataframe=self.test_fitness_df, dataset_type="test", to_predict=self.feat_to_predict)

        self.train_dataset.encode_X(encoding=encoding)
        self.test_dataset.encode_X(encoding=encoding)

        self.X_train_all = np.array(self.train_dataset.X)
        self.X_test = np.array(self.test_dataset.X)

        self.y_train_all = np.array(self.train_dataset.y)

        self.all_combos = self.train_dataset.all_combos
        self.n_sites = self.train_dataset.n_residues

    def train_all(self):
        '''
        Loops through all libraries to be sampled from (n_solutions) and for each solution trains n_subsets of models. Each model is an ensemble of n_splits models, each trained on 90% of the subset selected randomly.

        Output: results for each of the models
        '''
        with tqdm() as pbar:
            pbar.reset(self.n_solutions * self.n_subsets * self.n_splits)
            pbar.set_description('Training and evaluating')
            final_preds = np.zeros((self.n_samples, self.n_splits))
            for k in range(self.n_solutions):
                if self.save_model:
                    save_dir = os.path.join(self.save_path)
                    if not os.path.exists(save_dir):
                        os.makedirs(save_dir)

                if self.validation_name == 'k-fold':
                    kf = KFold(n_splits=self.n_splits)
                    for i, (train_index, test_index) in enumerate(kf.split(self.X_train_all)):
                        if self.n_splits > 1:
                            train_x, validation_x = train_index, test_index
                        else:
                            train_x = self.X_train_all
                            validation_x = self.X_train_all  # used for validation if desired
                        X_train = self.X_train_all[train_x]
                        y_train = self.y_train_all[train_x]
                        X_validation = self.X_train_all[validation_x]
                        y_validation = self.y_train_all[validation_x]
                        y_preds, clf = self.train_single(
                            X_train, y_train, X_validation, y_validation)
                        if self.save_model:
                            filename = 'split' + str(i) + '.model'
                            pickle.dump(
                                clf, open(os.path.join(save_dir, filename), 'wb'))
                        final_preds[:, i] = y_preds
                        pbar.update()
                    means = np.mean(final_preds, axis=1)
                    y_preds = means[:]
                    if self.first_append:
                        filename = self.save_path + 'results.csv'
                        delimiter = ','
                        column_names = [self.feat_to_predict]
                        np.savetxt(filename, y_preds, delimiter=delimiter,
                                   header=delimiter.join(column_names), comments='')
                    else:
                        y_preds = y_preds.flatten()
                        df = pd.read_csv(self.save_path + 'results.csv')
                        new_series = pd.Series(y_preds)
                        df[self.feat_to_predict] = new_series
                        df.to_csv(self.save_path + 'results.csv', index=False)
                else:
                    for i in range(self.n_splits):
                        if self.n_splits > 1:
                            train_x, validation_x = train_test_split(
                                self.X_train_all, test_size=0.1, random_state=i)
                        else:
                            train_x = self.X_train_all
                            validation_x = self.X_train_all
                        X_train = self.X_train_all[train_x]
                        y_train = self.y_train_all[train_x]
                        X_validation = self.X_train_all[validation_x]
                        y_validation = self.y_train_all[validation_x]
                        y_preds, clf = self.train_single(
                            X_train, y_train, X_validation, y_validation)
                        if self.save_model:
                            filename = 'split' + str(i) + '.model'
                            pickle.dump(
                                clf, open(os.path.join(save_dir, filename), 'wb'))
                        final_preds[:, i] = y_preds
                        pbar.update()

                    means = np.mean(final_preds, axis=1)
                    y_preds = means[:]
                    if self.first_append:
                        # Define the filename and delimiter for the CSV file
                        filename = self.save_path + 'results.csv'
                        delimiter = ','
                        # Define column names
                        column_names = [self.feat_to_predict]
                        # Write the NumPy array to the CSV file with column names
                        np.savetxt(filename, y_preds, delimiter=delimiter,
                                   header=delimiter.join(column_names), comments='')
                    else:
                        y_preds = y_preds.flatten()
                        df = pd.read_csv(self.save_path + 'results.csv')
                        new_series = pd.Series(y_preds)
                        # Add the new column to the DataFrame
                        df[self.feat_to_predict] = new_series
                        # Write the updated DataFrame back to the CSV file
                        df.to_csv(self.save_path + 'results.csv', index=False)
        pbar.close

        return

    def train_single(self, X_train: np.ndarray, y_train: np.ndarray, X_validation: np.ndarray, y_validation: np.ndarray):
        '''
        Trains a single supervised ML model. Returns the predictions on the training set and the trained model.
        '''
        if self.model_class == 'boosting':
            clf = get_model(
                self.model_class,
                model_kwargs={'nthread': self.num_workers}, sequence_length=56, vocab_size=69)
            eval_set = [(X_validation, y_validation)]
            clf[0].fit(X_train, y_train, eval_set=eval_set, verbose=False)
        else:
            clf = get_model(
                self.model_class,
                model_kwargs={}, sequence_length=56, vocab_size=69)
            clf[0].fit(X_train, y_train)
        y_preds = clf[0].predict(self.X_test)

        return y_preds, clf

    def run_neural_network(self, learning_rate, num_epochs):
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(device)
        # For fold results
        results = {}
        # Plot train and validation loss and save the figures!
        # Set fixed random number seed
        torch.manual_seed(42)
        # Loss and optimizer
        # might want to use MeanSquareError (check documentation)
        k_folds = 5
        criterion = nn.MSELoss(reduction='sum')
        # Device configuration
        x_train_tensor = torch.Tensor(self.X_train_all)
        y_train_tensor = torch.Tensor(self.y_train_all)
        total_trainset = torch.utils.data.TensorDataset(
            x_train_tensor, y_train_tensor)
        trainloader = torch.utils.data.DataLoader(
            total_trainset, batch_size=128, shuffle=True)

        test_tensor = torch.Tensor(self.test_dataset.X)
        testset = torch.utils.data.TensorDataset(test_tensor)
        testloader = torch.utils.data.DataLoader(
            testset, batch_size=128, shuffle=False)
        print(type(testloader))

        self.input_size = self.X_train_all.shape[1]
        print(type(self.input_size))
        self.hidden_size1 = 600
        print(type(self.hidden_size1))
        self.hidden_size2 = 30
        self.learning_rate = learning_rate
        self.num_epochs = num_epochs

        model = NeuralNet(self.input_size, self.hidden_size1,
                          self.hidden_size2, 1).to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=self.learning_rate)

        # total_data = ConcatDataset(trainset, testset)

        # Define the K-fold Cross Validator
        kfold = KFold(n_splits=k_folds, shuffle=True)
        # Train the model
        real_total_step = len(trainloader)

        for fold, (train_ids, validation_ids) in enumerate(kfold.split(total_trainset)):
            print(f"Fold{fold}")
            # Sample elements randomly from a given list of ids, no replacement.
            train_subsampler = torch.utils.data.SubsetRandomSampler(train_ids)
            validation_subsampler = torch.utils.data.SubsetRandomSampler(
                validation_ids)

            # Define data loaders for training and testing data in this fold
            trainloader = torch.utils.data.DataLoader(
                total_trainset,
                batch_size=10, sampler=train_subsampler)
            validationloader = torch.utils.data.DataLoader(
                total_trainset,
                batch_size=10, sampler=validation_subsampler)

            train_subsampler = torch.utils.data.SubsetRandomSampler(train_ids)
            validation_subsampler = torch.utils.data.SubsetRandomSampler(
                validation_ids)
            # Define data loaders for training and testing data in this fold
            trainloader = torch.utils.data.DataLoader(
                total_trainset,
                batch_size=10, sampler=train_subsampler)
            validationloader = torch.utils.data.DataLoader(
                total_trainset,
                batch_size=10, sampler=validation_subsampler)
            # to track the training loss as the model trains
            train_losses = []
            # to track the validation loss as the model trains
            valid_losses = []
            # to track the average training loss per epoch as the model trains
            avg_train_losses = []
            # to track the average validation loss per epoch as the model trains
            avg_valid_losses = []
            # try somewhere from 10 to 100 for number of epochs
            early_stopping = EarlyStopping(verbose=True)
            for epoch in range(self.num_epochs):
                for i, (data, quantity_to_predict) in enumerate(trainloader):
                    # Forward pass
                    outputs = model(data.to(device))
                    loss = criterion(outputs, quantity_to_predict.to(device))
                    # Backward and optimize
                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()
                    train_losses.append(loss.item())

                for data, target in validationloader:
                    # forward pass: compute predicted outputs by passing inputs to the model
                    output = model(data.to(device))
                    # calculate the loss
                    loss = criterion(output, target.to(device))
                    # record validation loss
                    valid_losses.append(loss.item())
                    # if (i+1) % 100 == 0:
                    #     # Compile losses manually and then compute the mean.
                    #     print ('Epoch [{}/{}], Step [{}/{}], Loss: {:.4f}'
                    #         .format(epoch+1, self.num_epochs, i+1, real_total_step, loss.item()))
                # print training/validation statistics
                # calculate average loss over an epoch
                train_loss = np.average(train_losses)
                valid_loss = np.average(valid_losses)
                avg_train_losses.append(train_loss)
                avg_valid_losses.append(valid_loss)

                epoch_len = len(str(self.num_epochs))

                print_msg = (f'[{epoch:>{epoch_len}}/{self.num_epochs:>{epoch_len}}] ' +
                             f'train_loss: {train_loss:.5f} ' +
                             f'valid_loss: {valid_loss:.5f}')

                print(print_msg)

                # clear lists to track next epoch
                train_losses = []
                valid_losses = []

                # early_stopping needs the validation loss to check if it has decresed,
                # and if it has, it will make a checkpoint of the current model
                early_stopping(valid_loss, model)

                if early_stopping.early_stop:
                    print("Early stopping")
                    break
        # move the testing and average across all five loops.
        with torch.no_grad():
            correct = 0
            total = 0
            print(testloader)
            # for i, (data, quantity_to_predict) in enumerate(testloader):
            for data in testloader:
                # data = torch.Tensor(data).int()
                outputs = model(data[0].to(device))
                print(outputs.shape)
                outputs = outputs.cpu()
                if self.first_append:
                    # Define the filename and delimiter for the CSV file
                    filename = self.save_path + 'results.csv'
                    delimiter = ','
                    # Define column names
                    column_names = [self.feat_to_predict]
                    # Write the NumPy array to the CSV file with column names
                    np.savetxt(filename, outputs, delimiter=delimiter,
                               header=delimiter.join(column_names), comments='')
                else:
                    outputs = outputs.flatten()
                    # np.squeeze(new_column)
                    # Convert the NumPy array to a Pandas Series
                    df = pd.read_csv(self.save_path + 'results.csv')
                    new_series = pd.Series(outputs)

                    # Add the new column to the DataFrame
                    df[self.feat_to_predict] = new_series

                    # Write the updated DataFrame back to the CSV file
                    df.to_csv(self.save_path + 'results.csv', index=False)

                # total += things_to_predict.size(0)
                # correct += (predicted == things_to_predict).sum().item()

            # print('Accuracy of the network on sequence data: {} %'.format(100 * correct / total))

            # Save the model checkpoint
            torch.save(model.state_dict(), 'model.ckpt')