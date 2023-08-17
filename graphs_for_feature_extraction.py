from collections import Counter
import MDAnalysis as mda
import mdtraj as md
import os
import numpy as np
import pandas as pd
import subprocess

# Path to obtain all MD related files for feature extraction
base_md_file_path = '/home/annika/md_sims/'

# List of residues we will be extracting data for.
# for later: for real feature extraction, automate this to the file itself.
residues_to_consider_sub = [i for i in range(1, 57)]
# residues_to_consider = [39]
# Dictionary corresponding to the amount of dihedral angles found
# on each of the amino acids.
dihedral_angles_dict = {
    "ALA": 0,
    "CYS": 1,
    "ASP": 2,
    "GLU": 3,
    "PHE": 2,
    "GLY": 0,
    "HIS": 2,
    "ILE": 2,
    "LYS": 4,
    "LEU": 2,
    "MET": 3,
    "ASN": 2,
    "PRO": 0,
    "GLN": 3,
    "ARG": 4,
    "SER": 1,
    "THR": 1,
    "VAL": 1,
    "TRP": 2,
    "TYR": 2
}

# Dictionary corresponding to one-letter representation of amino acids.
one_letter_rep_dict = {
    "ALA": 'A',
    "CYS": 'C',
    "ASP": 'D',
    "GLU": 'E',
    "PHE": 'F',
    "GLY": 'G',
    "HIS": 'H',
    "ILE": 'I',
    "LYS": 'K',
    "LEU": 'L',
    "MET": 'M',
    "ASN": 'N',
    "PRO": 'P',
    "GLN": 'Q',
    "ARG": 'R',
    "SER": 'S',
    "THR": 'T',
    "VAL": 'V',
    "TRP": 'W',
    "TYR": 'Y'
}

# Feature strings for datframe organization
residue_type = 'Residue Type'
time_step = 'Timestep'
c_alpha_disp = 'Displacement of Backbone C-Alpha'
centroid_res_disp = 'Displacement of Residue Centroid'
side_chain_disp = 'Displacement of Side-Chain Centroid'
dist_backbone_side_chain = 'Distance between Backbone and Furthest Side-chain Atom'
sasa_res_string = 'Dynamic SA'
entropy = 'Entropy'
mean_pot_energy = 'Mean Potential Energy'
var_pot_energy = 'Variance in Potential Energy'
rad_gyration = 'Radius of Gyration'


def create_state_library(uni, res):
    curr_state = 0
    chi_list = ["chi1", "chi2", "chi3", "chi4"]
    chi_angles_list = [60, -60, 180]
    total_states = get_num_states(uni, res.resids[0])
    state_lib = {}
    if total_states == 3:
        for i in range(len(chi_angles_list)):
            if curr_state != total_states:
                new_entry = {curr_state: {chi_list[0]: chi_angles_list[i]}}
                state_lib.update(new_entry)
                curr_state += 1
    elif total_states == 9:
        for i in range(len(chi_angles_list)):
            for j in range(len(chi_angles_list)):
                if curr_state != total_states:
                    new_entry = {curr_state: {chi_list[0]: chi_angles_list[i],
                                              chi_list[1]: chi_angles_list[j]}}
                    state_lib.update(new_entry)
                    curr_state += 1
    elif total_states == 27:
        for i in range(len(chi_angles_list)):
            for j in range(len(chi_angles_list)):
                for k in range(len(chi_angles_list)):
                    if curr_state != total_states:
                        state_lib.update({curr_state: {chi_list[0]: chi_angles_list[i],
                                                       chi_list[1]: chi_angles_list[j],
                                                       chi_list[2]: chi_angles_list[k]}})
                    curr_state += 1
    elif total_states == 81:
        for i in range(len(chi_angles_list)):
            for j in range(len(chi_angles_list)):
                for k in range(len(chi_angles_list)):
                    for l in range(len(chi_angles_list)):
                        if curr_state != total_states:
                            state_lib.update({curr_state: {chi_list[0]: chi_angles_list[i],
                                                           chi_list[1]: chi_angles_list[j],
                                                           chi_list[2]: chi_angles_list[k],
                                                           chi_list[3]: chi_angles_list[l]}})
    return state_lib


# Helper function to get the number of states for a particular
# amino acid.
def get_num_states(uni, res_num):
    residue = select_residue(uni, res_num)
    residue_name = residue.resnames[0]
    num_dihedral_angles = dihedral_angles_dict.get(residue_name)
    return pow(3, num_dihedral_angles)


# Helper function for converting list to dictionary storing
# the state as a probability of how often the protein is
# likely to be in that state.
def create_frequency_dict(state_list, num_states):
    frequency_counter = Counter(state_list)
    frequency_dict = dict(frequency_counter)
    for state in frequency_dict.keys():
        probability_value = frequency_dict.get(state)/num_states
        probability_entry = {state: probability_value}
        frequency_dict.update(probability_entry)
    return frequency_dict


def cys_calc_chi1(res):
    N = res.select_atoms("name N").positions[0]
    CA = res.select_atoms("name CA").positions[0]
    CB = res.select_atoms("name CB").positions[0]
    SG = res.select_atoms("name SG").positions[0]
    v1 = N - CA
    v2 = CB - CA
    v3 = SG - CB
    chi1 = np.degrees(
        np.arccos(
            np.dot(np.cross(v1, v2), np.cross(v2, v3))
            / (np.linalg.norm(np.cross(v1, v2)) * np.linalg.norm(np.cross(v2, v3)))
        )
    )
    return chi1


def asp_calc_chi1_chi2(res):
    N = res.select_atoms("name N").positions[0]
    CA = res.select_atoms("name CA").positions[0]
    CB = res.select_atoms("name CB").positions[0]
    CG = res.select_atoms("name CG").positions[0]
    OD2 = res.select_atoms("name OD2").positions[0]
    v1 = N - CA
    v2 = CB - CA
    v3 = CG - CB
    v4 = OD2 - CG
    chi1 = np.degrees(
        np.arccos(
            np.dot(np.cross(v1, v2), np.cross(v2, v3))
            / (np.linalg.norm(np.cross(v1, v2)) * np.linalg.norm(np.cross(v2, v3)))
        )
    )
    chi2 = np.degrees(
        np.arccos(
            np.dot(np.cross(v2, v3), np.cross(v3, v4))
            / (np.linalg.norm(np.cross(v2, v3)) * np.linalg.norm(np.cross(v3, v4)))
        )
    )
    return chi1, chi2


def glu_calc_chi1_chi2_chi3(res):
    N = res.select_atoms("name N").positions[0]
    CA = res.select_atoms("name CA").positions[0]
    CB = res.select_atoms("name CB").positions[0]
    CG = res.select_atoms("name CG").positions[0]
    CD = res.select_atoms("name CD").positions[0]
    OE2 = res.select_atoms("name OE2").positions[0]
    v1 = N - CA
    v2 = CB - CA
    v3 = CG - CB
    v4 = CD - CG
    v5 = OE2 - CD
    chi1 = np.degrees(
        np.arccos(
            np.dot(np.cross(v1, v2), np.cross(v2, v3))
            / (np.linalg.norm(np.cross(v1, v2)) * np.linalg.norm(np.cross(v2, v3)))
        )
    )
    chi2 = np.degrees(
        np.arccos(
            np.dot(np.cross(v2, v3), np.cross(v3, v4))
            / (np.linalg.norm(np.cross(v2, v3)) * np.linalg.norm(np.cross(v3, v4)))
        )
    )
    chi3 = np.degrees(
        np.arccos(
            np.dot(np.cross(v3, v4), np.cross(v4, v5))
            / (np.linalg.norm(np.cross(v3, v4)) * np.linalg.norm(np.cross(v4, v5)))
        )
    )
    return chi1, chi2, chi3


def phe_calc_chi1_chi2(res):
    N = res.select_atoms("name N").positions[0]
    CA = res.select_atoms("name CA").positions[0]
    CB = res.select_atoms("name CB").positions[0]
    CG = res.select_atoms("name CG").positions[0]
    CD2 = res.select_atoms("name CD2").positions[0]
    v1 = N - CA
    v2 = CB - CA
    v3 = CG - CB
    v4 = CD2 - CG
    chi1 = np.degrees(
        np.arccos(
            np.dot(np.cross(v1, v2), np.cross(v2, v3))
            / (np.linalg.norm(np.cross(v1, v2)) * np.linalg.norm(np.cross(v2, v3)))
        )
    )
    chi2 = np.degrees(
        np.arccos(
            np.dot(np.cross(v2, v3), np.cross(v3, v4))
            / (np.linalg.norm(np.cross(v2, v3)) * np.linalg.norm(np.cross(v3, v4)))
        )
    )
    return chi1, chi2


def his_calc_chi1_chi2(res):
    N = res.select_atoms("name N").positions[0]
    CA = res.select_atoms("name CA").positions[0]
    CB = res.select_atoms("name CB").positions[0]
    CG = res.select_atoms("name CG").positions[0]
    CD2 = res.select_atoms("name CD2").positions[0]
    v1 = N - CA
    v2 = CB - CA
    v3 = CG - CB
    v4 = CD2 - CG
    chi1 = np.degrees(
        np.arccos(
            np.dot(np.cross(v1, v2), np.cross(v2, v3))
            / (np.linalg.norm(np.cross(v1, v2)) * np.linalg.norm(np.cross(v2, v3)))
        )
    )
    chi2 = np.degrees(
        np.arccos(
            np.dot(np.cross(v2, v3), np.cross(v3, v4))
            / (np.linalg.norm(np.cross(v2, v3)) * np.linalg.norm(np.cross(v3, v4)))
        )
    )
    return chi1, chi2


def ile_calc_chi1_chi2(res):
    N = res.select_atoms("name N").positions[0]
    CA = res.select_atoms("name CA").positions[0]
    CB = res.select_atoms("name CB").positions[0]
    CG2 = res.select_atoms("name CG2").positions[0]
    CD1 = res.select_atoms("name CD1").positions[0]
    v1 = N - CA
    v2 = CB - CA
    v3 = CG2 - CB
    v4 = CD1 - CG2
    chi1 = np.degrees(
        np.arccos(
            np.dot(np.cross(v1, v2), np.cross(v2, v3))
            / (np.linalg.norm(np.cross(v1, v2)) * np.linalg.norm(np.cross(v2, v3)))
        )
    )
    chi2 = np.degrees(
        np.arccos(
            np.dot(np.cross(v2, v3), np.cross(v3, v4))
            / (np.linalg.norm(np.cross(v2, v3)) * np.linalg.norm(np.cross(v3, v4)))
        )
    )
    return chi1, chi2


def lys_calc_chi1_chi2_chi3_chi4(res):
    N = res.select_atoms("name N").positions[0]
    CA = res.select_atoms("name CA").positions[0]
    CB = res.select_atoms("name CB").positions[0]
    CG = res.select_atoms("name CG").positions[0]
    CD = res.select_atoms("name CD").positions[0]
    CE = res.select_atoms("name CE").positions[0]
    NZ = res.select_atoms("name NZ").positions[0]
    v1 = N - CA
    v2 = CB - CA
    v3 = CG - CB
    v4 = CD - CG
    v5 = CE - CD
    v6 = NZ - CE
    chi1 = np.degrees(
        np.arccos(
            np.dot(np.cross(v1, v2), np.cross(v2, v3))
            / (np.linalg.norm(np.cross(v1, v2)) * np.linalg.norm(np.cross(v2, v3)))
        )
    )
    chi2 = np.degrees(
        np.arccos(
            np.dot(np.cross(v2, v3), np.cross(v3, v4))
            / (np.linalg.norm(np.cross(v2, v3)) * np.linalg.norm(np.cross(v3, v4)))
        )
    )
    chi3 = np.degrees(
        np.arccos(
            np.dot(np.cross(v3, v4), np.cross(v4, v5))
            / (np.linalg.norm(np.cross(v3, v4)) * np.linalg.norm(np.cross(v4, v5)))
        )
    )
    chi4 = np.degrees(
        np.arccos(
            np.dot(np.cross(v4, v5), np.cross(v5, v6))
            / (np.linalg.norm(np.cross(v4, v5)) * np.linalg.norm(np.cross(v5, v6)))
        )
    )
    return chi1, chi2, chi3, chi4


def leu_calc_chi1_chi2(res):
    N = res.select_atoms("name N").positions[0]
    CA = res.select_atoms("name CA").positions[0]
    CB = res.select_atoms("name CB").positions[0]
    CG = res.select_atoms("name CG").positions[0]
    CD2 = res.select_atoms("name CD2").positions[0]
    v1 = N - CA
    v2 = CB - CA
    v3 = CG - CB
    v4 = CD2 - CG
    chi1 = np.degrees(
        np.arccos(
            np.dot(np.cross(v1, v2), np.cross(v2, v3))
            / (np.linalg.norm(np.cross(v1, v2)) * np.linalg.norm(np.cross(v2, v3)))
        )
    )
    chi2 = np.degrees(
        np.arccos(
            np.dot(np.cross(v2, v3), np.cross(v3, v4))
            / (np.linalg.norm(np.cross(v2, v3)) * np.linalg.norm(np.cross(v3, v4)))
        )
    )
    return chi1, chi2


def met_calc_chi1_chi2_chi3(res):
    N = res.select_atoms("name N").positions[0]
    CA = res.select_atoms("name CA").positions[0]
    CB = res.select_atoms("name CB").positions[0]
    CG = res.select_atoms("name CG").positions[0]
    SD = res.select_atoms("name SD").positions[0]
    CE = res.select_atoms("name CE").positions[0]
    v1 = N - CA
    v2 = CB - CA
    v3 = CG - CB
    v4 = SD - CG
    v5 = CE - SD
    chi1 = np.degrees(
        np.arccos(
            np.dot(np.cross(v1, v2), np.cross(v2, v3))
            / (np.linalg.norm(np.cross(v1, v2)) * np.linalg.norm(np.cross(v2, v3)))
        )
    )
    chi2 = np.degrees(
        np.arccos(
            np.dot(np.cross(v2, v3), np.cross(v3, v4))
            / (np.linalg.norm(np.cross(v2, v3)) * np.linalg.norm(np.cross(v3, v4)))
        )
    )
    chi3 = np.degrees(
        np.arccos(
            np.dot(np.cross(v3, v4), np.cross(v4, v5))
            / (np.linalg.norm(np.cross(v3, v4)) * np.linalg.norm(np.cross(v4, v5)))
        )
    )
    return chi1, chi2, chi3


def asn_calc_chi1_chi2(res):
    N = res.select_atoms("name N").positions[0]
    CA = res.select_atoms("name CA").positions[0]
    CB = res.select_atoms("name CB").positions[0]
    CG = res.select_atoms("name CG").positions[0]
    ND2 = res.select_atoms("name ND2").positions[0]
    v1 = N - CA
    v2 = CB - CA
    v3 = CG - CB
    v4 = ND2 - CG
    chi1 = np.degrees(
        np.arccos(
            np.dot(np.cross(v1, v2), np.cross(v2, v3))
            / (np.linalg.norm(np.cross(v1, v2)) * np.linalg.norm(np.cross(v2, v3)))
        )
    )
    chi2 = np.degrees(
        np.arccos(
            np.dot(np.cross(v2, v3), np.cross(v3, v4))
            / (np.linalg.norm(np.cross(v2, v3)) * np.linalg.norm(np.cross(v3, v4)))
        )
    )
    return chi1, chi2


def gln_calc_chi1_chi2_chi3(res):
    N = res.select_atoms("name N").positions[0]
    CA = res.select_atoms("name CA").positions[0]
    CB = res.select_atoms("name CB").positions[0]
    CG = res.select_atoms("name CG").positions[0]
    CD = res.select_atoms("name CD").positions[0]
    NE2 = res.select_atoms("name NE2").positions[0]
    v1 = N - CA
    v2 = CB - CA
    v3 = CG - CB
    v4 = CD - CG
    v5 = NE2 - CD
    chi1 = np.degrees(
        np.arccos(
            np.dot(np.cross(v1, v2), np.cross(v2, v3))
            / (np.linalg.norm(np.cross(v1, v2)) * np.linalg.norm(np.cross(v2, v3)))
        )
    )
    chi2 = np.degrees(
        np.arccos(
            np.dot(np.cross(v2, v3), np.cross(v3, v4))
            / (np.linalg.norm(np.cross(v2, v3)) * np.linalg.norm(np.cross(v3, v4)))
        )
    )
    chi3 = np.degrees(
        np.arccos(
            np.dot(np.cross(v3, v4), np.cross(v4, v5))
            / (np.linalg.norm(np.cross(v3, v4)) * np.linalg.norm(np.cross(v4, v5)))
        )
    )
    return chi1, chi2, chi3


def arg_calc_chi1_chi2_chi3_chi4(res):
    N = res.select_atoms("name N").positions[0]
    CA = res.select_atoms("name CA").positions[0]
    CB = res.select_atoms("name CB").positions[0]
    CG = res.select_atoms("name CG").positions[0]
    CD = res.select_atoms("name CD").positions[0]
    NE = res.select_atoms("name NE").positions[0]
    CZ = res.select_atoms("name CZ").positions[0]
    v1 = N - CA
    v2 = CB - CA
    v3 = CG - CB
    v4 = CD - CG
    v5 = NE - CD
    v6 = CZ - NE
    chi1 = np.degrees(
        np.arccos(
            np.dot(np.cross(v1, v2), np.cross(v2, v3))
            / (np.linalg.norm(np.cross(v1, v2)) * np.linalg.norm(np.cross(v2, v3)))
        )
    )
    chi2 = np.degrees(
        np.arccos(
            np.dot(np.cross(v2, v3), np.cross(v3, v4))
            / (np.linalg.norm(np.cross(v2, v3)) * np.linalg.norm(np.cross(v3, v4)))
        )
    )
    chi3 = np.degrees(
        np.arccos(
            np.dot(np.cross(v3, v4), np.cross(v4, v5))
            / (np.linalg.norm(np.cross(v3, v4)) * np.linalg.norm(np.cross(v4, v5)))
        )
    )
    chi4 = np.degrees(
        np.arccos(
            np.dot(np.cross(v4, v5), np.cross(v5, v6))
            / (np.linalg.norm(np.cross(v4, v5)) * np.linalg.norm(np.cross(v5, v6)))
        )
    )
    return chi1, chi2, chi3, chi4


def ser_calc_chi1(res):
    N = res.select_atoms("name N").positions[0]
    CA = res.select_atoms("name CA").positions[0]
    CB = res.select_atoms("name CB").positions[0]
    OG = res.select_atoms("name OG").positions[0]
    v1 = N - CA
    v2 = CB - CA
    v3 = OG - CB
    chi1 = np.degrees(
        np.arccos(
            np.dot(np.cross(v1, v2), np.cross(v2, v3))
            / (np.linalg.norm(np.cross(v1, v2)) * np.linalg.norm(np.cross(v2, v3)))
        )
    )
    return chi1


def thr_calc_chi1(res):
    N = res.select_atoms("name N").positions[0]
    CA = res.select_atoms("name CA").positions[0]
    CB = res.select_atoms("name CB").positions[0]
    OG1 = res.select_atoms("name OG1").positions[0]
    v1 = N - CA
    v2 = CB - CA
    v3 = OG1 - CB
    chi1 = np.degrees(
        np.arccos(
            np.dot(np.cross(v1, v2), np.cross(v2, v3))
            / (np.linalg.norm(np.cross(v1, v2)) * np.linalg.norm(np.cross(v2, v3)))
        )
    )
    return chi1


def val_calc_chi1(res):
    N = res.select_atoms("name N").positions[0]
    CA = res.select_atoms("name CA").positions[0]
    CB = res.select_atoms("name CB").positions[0]
    CG2 = res.select_atoms("name CG2").positions[0]
    v1 = N - CA
    v2 = CB - CA
    v3 = CG2 - CB
    chi1 = np.degrees(
        np.arccos(
            np.dot(np.cross(v1, v2), np.cross(v2, v3))
            / (np.linalg.norm(np.cross(v1, v2)) * np.linalg.norm(np.cross(v2, v3)))
        )
    )
    return chi1


def trp_calc_chi1_chi2(res):
    N = res.select_atoms("name N").positions[0]
    CA = res.select_atoms("name CA").positions[0]
    CB = res.select_atoms("name CB").positions[0]
    CG = res.select_atoms("name CG").positions[0]
    CD1 = res.select_atoms("name CD1").positions[0]
    v1 = N - CA
    v2 = CB - CA
    v3 = CG - CB
    v4 = CD1 - CG
    chi1 = np.degrees(
        np.arccos(
            np.dot(np.cross(v1, v2), np.cross(v2, v3))
            / (np.linalg.norm(np.cross(v1, v2)) * np.linalg.norm(np.cross(v2, v3)))
        )
    )
    chi2 = np.degrees(
        np.arccos(
            np.dot(np.cross(v2, v3), np.cross(v3, v4))
            / (np.linalg.norm(np.cross(v2, v3)) * np.linalg.norm(np.cross(v3, v4)))
        )
    )
    return chi1, chi2


def tyr_calc_chi1_chi2(res):
    N = res.select_atoms("name N").positions[0]
    CA = res.select_atoms("name CA").positions[0]
    CB = res.select_atoms("name CB").positions[0]
    CG = res.select_atoms("name CG").positions[0]
    CD1 = res.select_atoms("name CD1").positions[0]
    v1 = N - CA
    v2 = CB - CA
    v3 = CG - CB
    v4 = CD1 - CG
    chi1 = np.degrees(
        np.arccos(
            np.dot(np.cross(v1, v2), np.cross(v2, v3))
            / (np.linalg.norm(np.cross(v1, v2)) * np.linalg.norm(np.cross(v2, v3)))
        )
    )
    chi2 = np.degrees(
        np.arccos(
            np.dot(np.cross(v2, v3), np.cross(v3, v4))
            / (np.linalg.norm(np.cross(v2, v3)) * np.linalg.norm(np.cross(v3, v4)))
        )
    )
    return chi1, chi2


def assign_one_state(uni, res, chi1):
    one_state_library = create_state_library(uni, res)
    dist = []
    for i in range(len(one_state_library)):
        dist.append(
            np.sqrt(
                (chi1 - one_state_library[i]["chi1"]) ** 2
            )
        )
    return np.argmin(dist)


def assign_two_state(uni, res, chi1, chi2):
    two_state_library = create_state_library(uni, res)
    dist = []
    for i in range(len(two_state_library)):
        dist.append(
            np.sqrt(
                (chi1 - two_state_library[i]["chi1"]) ** 2
                + (chi2 - two_state_library[i]["chi2"]) ** 2
            )
        )
    return np.argmin(dist)


def assign_three_state(uni, res, chi1, chi2, chi3):
    three_state_library = create_state_library(uni, res)
    dist = []
    for i in range(len(three_state_library)):
        dist.append(
            np.sqrt(
                (chi1 - three_state_library[i]["chi1"]) ** 2
                + (chi2 - three_state_library[i]["chi2"]) ** 2
                + (chi3 - three_state_library[i]["chi3"]) ** 2
            )
        )
    return np.argmin(dist)


def assign_four_state(uni, res, chi1, chi2, chi3, chi4):
    four_state_library = create_state_library(uni, res)
    dist = []
    for i in range(len(four_state_library)):
        dist.append(
            np.sqrt(
                (chi1 - four_state_library[i]["chi1"]) ** 2
                + (chi2 - four_state_library[i]["chi2"]) ** 2
                + (chi3 - four_state_library[i]["chi3"]) ** 2
                + (chi4 - four_state_library[i]["chi4"]) ** 2
            )
        )
    return np.argmin(dist)


def create_probability_distribution(uni, res):
    states_through_sim = []
    for ts in uni.trajectory:
        if res.resnames[0] == "ALA" or res.resnames[0] == "GLY" or res.resnames[0] == "PRO":
            temp_state = 0
        elif res.resnames[0] == "CYS":
            chi1, chi2 = cys_calc_chi1(res)
            temp_state = assign_one_state(uni, res, chi1)
        elif res.resnames[0] == "ASP":
            chi1, chi2 = asp_calc_chi1_chi2(res)
            temp_state = assign_two_state(uni, res, chi1, chi2)
        elif res.resnames[0] == "GLU":
            chi1, chi2, chi3 = glu_calc_chi1_chi2_chi3(res)
            temp_state = assign_three_state(uni, res, chi1, chi2, chi3)
        elif res.resnames[0] == "PHE":
            chi1, chi2 = phe_calc_chi1_chi2(res)
            temp_state = assign_two_state(uni, res, chi1, chi2)
        elif res.resnames[0] == "HIS":
            chi1, chi2 = his_calc_chi1_chi2(res)
            temp_state = assign_two_state(uni, res, chi1, chi2)
        elif res.resnames[0] == "ILE":
            chi1, chi2 = ile_calc_chi1_chi2(res)
            temp_state = assign_two_state(uni, res, chi1, chi2)
        elif res.resnames[0] == "LYS":
            chi1, chi2, chi3, chi4 = lys_calc_chi1_chi2_chi3_chi4(res)
            temp_state = assign_four_state(uni, res, chi1, chi2, chi3, chi4)
        elif res.resnames[0] == "LEU":
            chi1, chi2 = leu_calc_chi1_chi2(res)
            temp_state = assign_two_state(uni, res, chi1, chi2)
        elif res.resnames[0] == "MET":
            chi1, chi2, chi3 = met_calc_chi1_chi2_chi3(res)
            temp_state = assign_three_state(uni, res, chi1, chi2, chi3)
        elif res.resnames[0] == "ASN":
            chi1, chi2 = asn_calc_chi1_chi2(res)
            temp_state = assign_two_state(uni, res, chi1, chi2)
        elif res.resnames[0] == "GLN":
            chi1, chi2, chi3 = gln_calc_chi1_chi2_chi3(res)
            temp_state = assign_three_state(uni, res, chi1, chi2, chi3)
        elif res.resnames[0] == "ARG":
            chi1, chi2, chi3, chi4 = arg_calc_chi1_chi2_chi3_chi4(res)
            temp_state = assign_four_state(uni, res, chi1, chi2, chi3, chi4)
        elif res.resnames[0] == "SER":
            chi1 = ser_calc_chi1(res)
            temp_state = assign_one_state(uni, res, chi1)
        elif res.resnames[0] == "THR":
            chi1 = thr_calc_chi1(res)
            temp_state = assign_one_state(uni, res, chi1)
        elif res.resnames[0] == "VAL":
            chi1 = val_calc_chi1(res)
            temp_state = assign_one_state(uni, res, chi1)
        elif res.resnames[0] == "TRP":
            chi1, chi2 = trp_calc_chi1_chi2(res)
            temp_state = assign_two_state(uni, res, chi1, chi2)
        elif res.resnames[0] == "TYR":
            chi1, chi2 = tyr_calc_chi1_chi2(res)
            temp_state = assign_two_state(uni, res, chi1, chi2)
        states_through_sim.append(temp_state)
    num_states = get_num_states(uni, res.resids[0])
    return create_frequency_dict(states_through_sim, num_states)


def calculate_entropy(uni, res):
    entropy_value = 0
    state_number = 0
    probability_dist_residue = create_probability_distribution(uni, res)
    num_states = get_num_states(uni, res.resids[0])
    for _ in range(num_states):
        p_xi = probability_dist_residue.get(state_number)
        try:
            entropy_value += (p_xi * np.log(p_xi))
        except:
            entropy_value += 0
        state_number += 1
    return entropy_value


def select_residue(uni, res_num):
    residue_string = f'resid {res_num}'
    residue = uni.select_atoms(residue_string)
    return residue


def obtain_atom_pos_vector(uni, res_num, atom_name, timestep):
    residue = select_residue(uni, res_num)
    for ts in uni.trajectory:
        if ts.frame == timestep:
            return residue.select_atoms(f'name {atom_name}').positions[0]


def calculate_disp_backbone_side_chain(uni, res_num, side_chain_atom, timestep):
    backbone_vector = obtain_atom_pos_vector(uni, res_num, 'CA', timestep)
    side_chain_vector = obtain_atom_pos_vector(
        uni, res_num, side_chain_atom, timestep)
    return np.linalg.norm(np.subtract(backbone_vector, side_chain_vector))


def obtain_disp_backbone_side_chain(uni, res_num, timestep):
    res = select_residue(uni, res_num)
    if res.resnames[0] == "ALA":
        return calculate_disp_backbone_side_chain(uni, res_num, 'CB', timestep)
    elif res.resnames[0] == "GLY":
        return calculate_disp_backbone_side_chain(uni, res_num, 'H2', timestep)
    elif res.resnames[0] == "PRO":
        return calculate_disp_backbone_side_chain(uni, res_num, 'CG', timestep)
    elif res.resnames[0] == "CYS":
        return calculate_disp_backbone_side_chain(uni, res_num, 'SG', timestep)
    elif res.resnames[0] == "ASP":
        return calculate_disp_backbone_side_chain(uni, res_num, 'OD2', timestep)
    elif res.resnames[0] == "GLU":
        return calculate_disp_backbone_side_chain(uni, res_num, 'OE2', timestep)
    elif res.resnames[0] == "PHE":
        return calculate_disp_backbone_side_chain(uni, res_num, 'CZ', timestep)
    elif res.resnames[0] == "HIS":
        return calculate_disp_backbone_side_chain(uni, res_num, 'NE2', timestep)
    elif res.resnames[0] == "ILE":
        return calculate_disp_backbone_side_chain(uni, res_num, 'CD1', timestep)
    elif res.resnames[0] == "LYS":
        return calculate_disp_backbone_side_chain(uni, res_num, 'NZ', timestep)
    elif res.resnames[0] == "LEU":
        return calculate_disp_backbone_side_chain(uni, res_num, 'CD1', timestep)
    elif res.resnames[0] == "MET":
        return calculate_disp_backbone_side_chain(uni, res_num, 'CE', timestep)
    elif res.resnames[0] == "ASN":
        return calculate_disp_backbone_side_chain(uni, res_num, 'ND2', timestep)
    elif res.resnames[0] == "GLN":
        return calculate_disp_backbone_side_chain(uni, res_num, 'NE2', timestep)
    elif res.resnames[0] == "ARG":
        return calculate_disp_backbone_side_chain(uni, res_num, 'NH1', timestep)
    elif res.resnames[0] == "SER":
        return calculate_disp_backbone_side_chain(uni, res_num, 'OG', timestep)
    elif res.resnames[0] == "THR":
        return calculate_disp_backbone_side_chain(uni, res_num, 'CG2', timestep)
    elif res.resnames[0] == "VAL":
        return calculate_disp_backbone_side_chain(uni, res_num, 'CG1', timestep)
    elif res.resnames[0] == "TRP":
        return calculate_disp_backbone_side_chain(uni, res_num, 'CH2', timestep)
    elif res.resnames[0] == "TYR":
        return calculate_disp_backbone_side_chain(uni, res_num, 'OH', timestep)


def obtain_ending_centroid_disp_res(uni, res_num, timestep=4999):
    residue = select_residue(uni, res_num)
    mean_list = []
    for ts in uni.trajectory:
        if ts.frame == timestep:
            for i in range(residue.n_atoms):
                mean_list.append(residue.atoms[i].position)
            return np.mean(mean_list, axis=0)


def obtain_centroid_disp_res(uni, res_num, timestep):
    residue = select_residue(uni, res_num)
    displacement_list = []
    centroid_ref = obtain_ending_centroid_disp_res(uni, res_num)
    for ts in uni.trajectory:
        if ts.frame == timestep:
            for i in range(residue.n_atoms):
                displacement_list.append(residue.atoms[i].position)
            curr_res_centroid = np.mean(displacement_list, axis=0)
            return np.linalg.norm(np.subtract(curr_res_centroid, centroid_ref))


def obtain_backbone_disp_res(uni, res_num, timestep):
    residue = select_residue(uni, res_num)
    atom_reference = residue.select_atoms('name CA').positions[0]
    for ts in uni.trajectory:
        if ts.frame == timestep:
            return np.linalg.norm(np.subtract(
                residue.select_atoms('name CA').positions[0], atom_reference))


def obtain_ending_side_chain_centroid_pos(uni, res_num, timestep=4999):
    residue = select_residue(uni, res_num)
    backbone_atoms = ['N', 'CA', 'C', 'O', 'OH']
    res_atom_list = list(residue.atoms.names)
    mean_list = []
    for ts in uni.trajectory:
        if ts.frame == timestep:
            for i in range(residue.n_atoms):
                if res_atom_list[i] not in backbone_atoms:
                    mean_list.append(residue.atoms[i].position)
            return np.mean(mean_list, axis=0)


def obtain_side_chain_centroid_disp(uni, res_num, timestep):
    residue = select_residue(uni, res_num)
    backbone_atoms = ['N', 'CA', 'C', 'O', 'OH']
    res_atom_list = list(residue.atoms.names)
    displacement_list = []
    side_chain_ref = obtain_ending_side_chain_centroid_pos(uni, res_num)
    for ts in uni.trajectory:
        if ts.frame == timestep:
            for i in range(residue.n_atoms):
                if res_atom_list[i] not in backbone_atoms:
                    displacement_list.append(residue.atoms[i].position)
            side_chain_centroid = np.mean(displacement_list, axis=0)
            return np.linalg.norm(np.subtract(side_chain_centroid, side_chain_ref))


def compute_dynamic_surface_area():
    traj = md.load(convert_traj_path, topology_file_path)
    sasa = md.shrake_rupley(traj, mode='residue')
    return sasa


def sasa_res_of_interest(uni, residue):
    sasa_vals = compute_dynamic_surface_area()
    sasa_list = []
    for ts in uni.trajectory:
        frame = ts.frame
        sasa_list.append(sasa_vals[frame, residue])
    return sasa_list


def obtain_mean_var_potential_energies(log_file):
    potential_energy_list = []
    with open(log_file, 'r') as md_logfile:
        for line in md_logfile:
            if line[0] != "#":
                potential_energy_val = float(list(line.split(','))[1])
                potential_energy_list.append(potential_energy_val)
    np_pot_energy_list = np.asarray(potential_energy_list)
    return (np.mean(np_pot_energy_list), np.var(np_pot_energy_list))


def obtain_rad_gyration(uni, timestep):
    protein = uni.select_atoms("protein")
    for ts in uni.trajectory:
        if ts.frame == timestep:
            return protein.radius_of_gyration()


def update_timestep_features(timestep, residue, uni, final_frame, sasa_vals, log_file_dir):
    feature_dict = {}
    res_sel = select_residue(uni, residue)
    sasa_val = sasa_vals[timestep]
    feature_dict.update(
        {f'{residue_type}': one_letter_rep_dict.get(res_sel.resnames[0])})
    feature_dict.update({f'{time_step}': timestep + 1})
    feature_dict.update({f'{sasa_res_string}': sasa_val})
    feature_dict.update(
        {f'{c_alpha_disp}': obtain_backbone_disp_res(uni, residue, timestep)})
    feature_dict.update(
        {f'{centroid_res_disp}': obtain_centroid_disp_res(uni, residue, timestep)})
    feature_dict.update(
        {f'{side_chain_disp}': obtain_side_chain_centroid_disp(uni, residue, timestep)})
    feature_dict.update(
        {f'{dist_backbone_side_chain}': obtain_disp_backbone_side_chain(uni, residue, timestep)})
    feature_dict.update({f'{entropy}': calculate_entropy(uni, res_sel)})
    feature_dict.update(
        {f'{mean_pot_energy}': obtain_mean_var_potential_energies(log_file_dir)[0]})
    feature_dict.update(
        {f'{var_pot_energy}': obtain_mean_var_potential_energies(log_file_dir)[0]})
    feature_dict.update(
        {f'{rad_gyration}': obtain_rad_gyration(uni, timestep)})
    feature_data = pd.DataFrame(feature_dict, index=[residue])
    final_frame = pd.concat([feature_data, final_frame])
    return final_frame


def create_final_dataframe(uni, res_list, log_file_dir):
    md_dict = {}
    md_data = pd.DataFrame(md_dict)
    print(len(uni.trajectory))
    for i in range(len(res_list)):
        sasa_list = sasa_res_of_interest(uni, res_list[i])
        for timestep in range(len(uni.trajectory)):
            print((i + 1), (timestep + 1))
            if (timestep + 1) % 125 == 0:
                md_data = update_timestep_features(
                    timestep, res_list[i], universe, md_data, sasa_list, log_file_dir)
    return md_data.sort_index()


md_base_filepath = f'{base_md_file_path}some_test_result/'
log_file_path = f'{md_base_filepath}prod.log'
trajectory_file_path = f'{md_base_filepath}prod.dcd'
topology_file_path = f'{md_base_filepath}end_annika.pdb'
convert_traj_path = trajectory_file_path[:len(
    trajectory_file_path) - 4] + '.h5'
subprocess.run(["mdconvert", trajectory_file_path, "-t",
               topology_file_path, "-o", convert_traj_path])
universe = mda.Universe(topology_file_path, trajectory_file_path)
md_data_final = create_final_dataframe(
    universe, residues_to_consider_sub, log_file_path)
md_data_final.to_csv(
    '/home/annika/md_sims/timestep_dependent_features_gb1_with_more_disp.csv')
print("File saved!")
