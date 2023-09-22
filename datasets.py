from encoding_utils import *
import pandas as pd

encoding_dict = {
    'one-hot' : generate_onehot,
    'georgiev' : generate_georgiev
}

class Dataset():
    """
    Prepares a given dataset for MLDE
    """
    def __init__(self, dataframe, dataset_type, to_predict):
        self.data = dataframe
        self.N = len(self.data)
        self.dataset_type = dataset_type
        self.to_predict = to_predict
        if self.dataset_type == "train":
            self.data = self.data.dropna(subset=[self.to_predict])
        self.all_combos = self.data['Variants'].values
        self.data[self.to_predict] = self.data[self.to_predict].values
        self.y = self.data[self.to_predict]

    def encode_X(self, encoding: str):
        """
        Encodes the input features based on the encoding type.
        """
        if encoding == 'one-hot':
            self.X = np.array(encoding_dict[encoding](self.all_combos)) 
            self.X = self.X.reshape(self.X.shape[0],-1) 

        self.input_dim = self.X.shape[1]
        self.n_residues = self.input_dim/len(ALL_AAS)
