# Leveraging Structural and Biophysical Information for Machine Learning-Assisted Protein Engineering

This repository contains the code used to generate relevant feature data based on Molecular Dynamics simulation runs and machine learning baseline tests based on one-hot encoded sequence data. Research conducted under Professor Frances Arnold (Caltech).

---

Below is an explanation of each of the files presented in this repository <br />

**create_json_mutant.py** - Creates a custom .json configuration file for feature extraction based on .pdb TRIAD file one wishes to extract MD-related features on. <br />

To generate a custom .json configuration file, run the following
```bash
python3 create_json_mutant.py [triad_struct_filename]
```
All TRIAD files that can be inputted into this program must be named in a similar manner to what is presented below
```bash
2GI9_prepared_designed_9_A_40L.pdb
```
**create_test_train_csv.py** - Creates the train and test datasets for GB1 sequential data with approaches based on the work of [Dallago et. al](https://www.biorxiv.org/content/10.1101/2021.11.09.467890v2.full). The datasets for baseline machine learning tests were created using the one vs. rest approach highlighted in the paper.

**mlde_test.py** - Runs the machine learning baseline tests. See documentation in **train_eval.py** for how to customize calling the MLDESim class for other baseline tests.

**official_extraction.py** - Extracts data from molecular-dynamics simulations based on a .json configuration file (that can be created via **create_json_mutant.py**) and places relevant residue-specific features in a .csv file for the specified mutant.

To generate a dataset for a customized mutant, run the following
```bash
python3 official_extraction.py [json_configuration_file]
```
**pytorch_models.py** - Contains custom neural network models for machine learning baseline tests. Currently has two implementations of a convolutional neural network and a feed forward neural network.

**run-md.py** - Runs a 20 nanosecond (production step time) molecular dynamics simulation for a custom protein, and generates relevant files for the feature extraction pipeline.
To generate a dataset for a customized mutant, run the following
```bash
python3 run-md.py [triad_struct_filename]
```
**train_eval.py** - Contains the architecture for encoding sequence data via one-hot encodings, training, and testing machine learning models needed for baseline tests.
