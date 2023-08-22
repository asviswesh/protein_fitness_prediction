from collections import Counter
from MDAnalysis.analysis import align
from MDAnalysis.analysis.dihedrals import Dihedral
from MDAnalysis import transformations
import argparse
import csv
import json
import os
import math
import MDAnalysis as mda
import numpy as np
import pandas as pd
import time


class Residue:
    def __init__(self, uni, res_num):
        self.res_num = res_num
        self.residue = select_residue(uni, res_num)
        self.protein = uni.select_atoms("protein")
        self.num_frames = len(uni.trajectory)
        self.res_atom_list = list(self.residue.atoms.names)
        self.backbone_pos = self.residue.select_atoms('name CA').positions[0]
        self.farthest_side_chain_array = self.residue.select_atoms(
            'name N').positions
        self.rad_of_gyration = self.protein.radius_of_gyration()
        self.backbone_list = ['N', 'CA', 'C', 'O', 'OH']
        self.water_list = ['H1', 'O', 'H2']
        self.chi_list = ["chi1", "chi2", "chi3", "chi4"]
        self.chi_angles_list = [60, -60, 180]

    def get_furthest_position(self):
        self.farthest_side_chain = self.residue.select_atoms(
            self.furthest_atom_type).positions[0]
        return self.farthest_side_chain

    def calc_side_chain_centroid_pos(self):
        self.displacement_list = []
        for i in range(self.residue.n_atoms):
            if self.res_atom_list[i] not in self.backbone_list or self.res_atom_list[i] not in self.water_list:
                self.displacement_list.append(self.residue.atoms[i].position)
        return np.mean(self.displacement_list, axis=0)

    def calc_residue_centroid(self):
        self.displacement_list = []
        for i in range(self.residue.n_atoms):
            if self.res_atom_list[i] not in self.water_list:
                self.displacement_list.append(self.residue.atoms[i].position)
        return np.mean(self.displacement_list, axis=0)

    def create_one_state_library(self):
        curr_state = 0
        state_lib = {}
        for i in range(len(self.chi_angles_list)):
            if curr_state != 3:
                new_entry = {curr_state: {
                    self.chi_list[0]: self.chi_angles_list[i]}}
                state_lib.update(new_entry)
                curr_state += 1
        return state_lib

    def create_two_state_library(self):
        curr_state = 0
        state_lib = {}
        for i in range(len(self.chi_angles_list)):
            for j in range(len(self.chi_angles_list)):
                if curr_state != 9:
                    new_entry = {curr_state: {self.chi_list[0]: self.chi_angles_list[i],
                                              self.chi_list[1]: self.chi_angles_list[j]}}
                    state_lib.update(new_entry)
                    curr_state += 1
        return state_lib

    def create_three_state_library(self):
        curr_state = 0
        state_lib = {}
        for i in range(len(self.chi_angles_list)):
            for j in range(len(self.chi_angles_list)):
                for k in range(len(self.chi_angles_list)):
                    if curr_state != 27:
                        state_lib.update({curr_state: {self.chi_list[0]: self.chi_angles_list[i],
                                                       self.chi_list[1]: self.chi_angles_list[j],
                                                       self.chi_list[2]: self.chi_angles_list[k]}})
                    curr_state += 1
        return state_lib

    def create_four_state_library(self):
        curr_state = 0
        state_lib = {}
        for i in range(len(self.chi_angles_list)):
            for j in range(len(self.chi_angles_list)):
                for k in range(len(self.chi_angles_list)):
                    for l in range(len(self.chi_angles_list)):
                        if curr_state != 81:
                            state_lib.update({curr_state: {self.chi_list[0]: self.chi_angles_list[i],
                                                           self.chi_list[1]: self.chi_angles_list[j],
                                                           self.chi_list[2]: self.chi_angles_list[k],
                                                           self.chi_list[3]: self.chi_angles_list[l]}})
                        curr_state += 1
        return state_lib

    def assign_zero_state(self):
        return 0

    def assign_one_state(self, chi1):
        if chi1 < -120:
            chi1 += 360
        one_state_library = self.create_one_state_library()
        chi1_dist = []
        for i in range(len(one_state_library)):
            chi1_dist.append(abs(chi1 - one_state_library[i]["chi1"]))
        return np.argmin(chi1_dist)

    def assign_two_state(self, chi1, chi2):
        if chi1 < -120:
            chi1 += 360
        if chi2 < -120:
            chi2 += 360
        two_state_library = self.create_two_state_library()
        chi1_dist = []
        chi2_dist = []
        for i in range(len(two_state_library)):
            chi1_dist.append(abs(chi1 - two_state_library[i]["chi1"]))
            chi2_dist.append(abs(chi2 - two_state_library[i]["chi2"]))
        chi1_index = np.argmin(chi1_dist)
        chi2_index = np.argmin(chi2_dist)
        chi1_angle = two_state_library.get(chi1_index).get('chi1')
        chi2_angle = two_state_library.get(chi2_index).get('chi2')
        temp_chi_dict = {'chi1': chi1_angle, 'chi2': chi2_angle}
        state_val = [
            i for i in two_state_library if two_state_library[i] == temp_chi_dict]
        return state_val[0]

    def assign_three_state(self, chi1, chi2, chi3):
        if chi1 < -120:
            chi1 += 360
        if chi2 < -120:
            chi2 += 360
        if chi3 < -120:
            chi3 += 360
        three_state_library = self.create_three_state_library()
        chi1_dist = []
        chi2_dist = []
        chi3_dist = []
        for i in range(len(three_state_library)):
            chi1_dist.append(abs(chi1 - three_state_library[i]["chi1"]))
            chi2_dist.append(abs(chi2 - three_state_library[i]["chi2"]))
            chi3_dist.append(abs(chi3 - three_state_library[i]["chi3"]))
        chi1_index = np.argmin(chi1_dist)
        chi2_index = np.argmin(chi2_dist)
        chi3_index = np.argmin(chi3_dist)
        chi1_angle = three_state_library.get(chi1_index).get('chi1')
        chi2_angle = three_state_library.get(chi2_index).get('chi2')
        chi3_angle = three_state_library.get(chi3_index).get('chi3')
        temp_chi_dict = {'chi1': chi1_angle,
                         'chi2': chi2_angle,
                         'chi3': chi3_angle}
        state_val = [
            i for i in three_state_library if three_state_library[i] == temp_chi_dict]
        return state_val[0]

    def assign_four_state(self, chi1, chi2, chi3, chi4):
        if chi1 < -120:
            chi1 += 360
        if chi2 < -120:
            chi2 += 360
        if chi3 < -120:
            chi3 += 360
        if chi4 < -120:
            chi4 += 360
        four_state_library = self.create_four_state_library()
        chi1_dist = []
        chi2_dist = []
        chi3_dist = []
        chi4_dist = []
        for i in range(len(four_state_library)):
            chi1_dist.append(abs(chi1 - four_state_library[i]["chi1"]))
            chi2_dist.append(abs(chi2 - four_state_library[i]["chi2"]))
            chi3_dist.append(abs(chi3 - four_state_library[i]["chi3"]))
            chi4_dist.append(abs(chi4 - four_state_library[i]["chi4"]))
        chi1_index = np.argmin(chi1_dist)
        chi2_index = np.argmin(chi2_dist)
        chi3_index = np.argmin(chi3_dist)
        chi4_index = np.argmin(chi4_dist)
        chi1_angle = four_state_library.get(chi1_index).get('chi1')
        chi2_angle = four_state_library.get(chi2_index).get('chi2')
        chi3_angle = four_state_library.get(chi3_index).get('chi3')
        chi4_angle = four_state_library.get(chi4_index).get('chi4')
        temp_chi_dict = {'chi1': chi1_angle,
                         'chi2': chi2_angle,
                         'chi3': chi3_angle,
                         'chi4': chi4_angle}
        state_val = [
            i for i in four_state_library if four_state_library[i] == temp_chi_dict]
        return state_val[0]

    def create_frequency_dict(self, state_list, num_states, uni):
        frequency_counter = Counter(state_list)
        frequency_dict = dict(frequency_counter)
        for state in range(num_states):
            if state in frequency_dict.keys():
                probability_value = frequency_dict.get(
                    state)/(len(uni.trajectory))
                probability_entry = {state: probability_value}
                frequency_dict.update(probability_entry)
        return frequency_dict

    def create_probability_distribution_one(self, uni, chi1_list):
        states_through_sim = []
        for ts in uni.trajectory:
            states_through_sim.append(self.assign_state(chi1_list[ts.frame]))
        return self.create_frequency_dict(states_through_sim, self.num_states, uni)

    def create_probability_distribution_two(self, uni, chi1_list, chi2_list):
        states_through_sim = []
        for ts in uni.trajectory:
            states_through_sim.append(self.assign_state(
                chi1_list[ts.frame], chi2_list[ts.frame]))
        return self.create_frequency_dict(states_through_sim, self.num_states, uni)

    def create_probability_distribution_three(self, uni, chi1_list, chi2_list, chi3_list):
        states_through_sim = []
        for ts in uni.trajectory:
            states_through_sim.append(self.assign_state(
                chi1_list[ts.frame], chi2_list[ts.frame], chi3_list[ts.frame]))
        return self.create_frequency_dict(states_through_sim, self.num_states, uni)

    def create_probability_distribution_four(self, uni, chi1_list, chi2_list, chi3_list, chi4_list):
        states_through_sim = []
        for ts in uni.trajectory:
            states_through_sim.append(self.assign_state(
                chi1_list[ts.frame], chi2_list[ts.frame], chi3_list[ts.frame], chi4_list[ts.frame]))
        return self.create_frequency_dict(states_through_sim, self.num_states, uni)

    def calculate_entropy_one(self, uni, chi1_list):
        entropy_value = 0
        probability_dist_residue = self.create_probability_distribution_one(
            uni, chi1_list)
        for key in probability_dist_residue.keys():
            p_xi = probability_dist_residue.get(key)
            entropy_value += (p_xi * np.log(p_xi))
        return entropy_value

    def calculate_entropy_two(self, uni, chi1_list, chi2_list):
        entropy_value = 0
        probability_dist_residue = self.create_probability_distribution_two(
            uni, chi1_list, chi2_list)
        for key in probability_dist_residue.keys():
            p_xi = probability_dist_residue.get(key)
            entropy_value += (p_xi * np.log(p_xi))
        return entropy_value

    def calculate_entropy_three(self, uni, chi1_list, chi2_list, chi3_list):
        entropy_value = 0
        probability_dist_residue = self.create_probability_distribution_three(
            uni, chi1_list, chi2_list, chi3_list)
        for key in probability_dist_residue.keys():
            p_xi = probability_dist_residue.get(key)
            entropy_value += (p_xi * np.log(p_xi))
        return entropy_value

    def calculate_entropy_four(self, uni, chi1_list, chi2_list, chi3_list, chi4_list):
        entropy_value = 0
        probability_dist_residue = self.create_probability_distribution_four(
            uni, chi1_list, chi2_list, chi3_list, chi4_list)
        for key in probability_dist_residue.keys():
            p_xi = probability_dist_residue.get(key)
            entropy_value += (p_xi * np.log(p_xi))
        return entropy_value


class Ala(Residue):
    def __init__(self, uni, res_num) -> None:
        super(Ala, self).__init__(uni, res_num)
        self.furthest_atom_type = 'name CB'
        self.assign_state = self.assign_zero_state
        self.num_states = 1
        self.dihedral_atom_list = []


class Cys(Residue):
    def __init__(self, uni, res_num) -> None:
        super(Cys, self).__init__(uni, res_num)
        self.furthest_atom_type = 'name SG'
        self.assign_state = self.assign_one_state
        self.num_states = 3
        self.dihedral_atom_list = ['N', 'CA', 'CB', 'SG']


class Asp(Residue):
    def __init__(self, uni, res_num) -> None:
        super(Asp, self).__init__(uni, res_num)
        self.furthest_atom_type = 'name OD2'
        self.assign_state = self.assign_two_state
        self.num_states = 9
        self.dihedral_atom_list = ['N', 'CA', 'CB', 'CG', 'OD2']


class Glu(Residue):
    def __init__(self, uni, res_num) -> None:
        super(Glu, self).__init__(uni, res_num)
        self.furthest_atom_type = 'name OE2'
        self.assign_state = self.assign_three_state
        self.num_states = 27
        self.dihedral_atom_list = ['N', 'CA', 'CB', 'CG', 'CD', 'OE2']


class Phe(Residue):
    def __init__(self, uni, res_num) -> None:
        super(Phe, self).__init__(uni, res_num)
        self.furthest_atom_type = 'name CZ'
        self.assign_state = self.assign_two_state
        self.num_states = 9
        self.dihedral_atom_list = ['N', 'CA', 'CB', 'CG', 'CD2']


class Gly(Residue):
    def __init__(self, uni, res_num) -> None:
        super(Gly, self).__init__(uni, res_num)
        self.furthest_atom_type = 'name H2'
        self.assign_state = self.assign_zero_state
        self.num_states = 1
        self.dihedral_atom_list = []


class His(Residue):
    def __init__(self, uni, res_num) -> None:
        super(His, self).__init__(uni, res_num)
        self.furthest_atom_type = 'name NE2'
        self.assign_state = self.assign_two_state
        self.num_states = 9
        self.dihedral_atom_list = ['N', 'CA', 'CB', 'CG', 'CD2']


class Ile(Residue):
    def __init__(self, uni, res_num) -> None:
        super(Ile, self).__init__(uni, res_num)
        self.furthest_atom_type = 'name CD1'
        self.assign_state = self.assign_two_state
        self.num_states = 9
        self.dihedral_atom_list = ['N', 'CA', 'CB', 'CG1', 'CD1']


class Lys(Residue):
    def __init__(self, uni, res_num) -> None:
        super(Lys, self).__init__(uni, res_num)
        self.furthest_atom_type = 'name NZ'
        self.assign_state = self.assign_four_state
        self.num_states = 81
        self.dihedral_atom_list = ['N', 'CA', 'CB', 'CG', 'CD', 'CE', 'NZ']


class Leu(Residue):
    def __init__(self, uni, res_num) -> None:
        super(Leu, self).__init__(uni, res_num)
        self.furthest_atom_type = 'name CD1'
        self.assign_state = self.assign_two_state
        self.num_states = 9
        self.dihedral_atom_list = ['N', 'CA', 'CB', 'CG', 'CD2']


class Met(Residue):
    def __init__(self, uni, res_num) -> None:
        super(Met, self).__init__(uni, res_num)
        self.furthest_atom_type = 'name CE'
        self.assign_state = self.assign_three_state
        self.num_states = 27
        self.dihedral_atom_list = ['N', 'CA', 'CB', 'CG', 'SD', 'CE']


class Asn(Residue):
    def __init__(self, uni, res_num) -> None:
        super(Asn, self).__init__(uni, res_num)
        self.furthest_atom_type = 'name ND2'
        self.assign_state = self.assign_two_state
        self.num_states = 9
        self.dihedral_atom_list = ['N', 'CA', 'CB', 'CG', 'ND2']


class Pro(Residue):
    def __init__(self, uni, res_num) -> None:
        super(Pro, self).__init__(uni, res_num)
        self.furthest_atom_type = 'name ND2'
        self.assign_state = self.assign_zero_state
        self.num_states = 1
        self.dihedral_atom_list = []


class Gln(Residue):
    def __init__(self, uni, res_num) -> None:
        super(Gln, self).__init__(uni, res_num)
        self.furthest_atom_type = 'name NE2'
        self.assign_state = self.assign_three_state
        self.num_states = 27
        self.dihedral_atom_list = ['N', 'CA', 'CB', 'CG', 'CD', 'NE2']


class Arg(Residue):
    def __init__(self, uni, res_num) -> None:
        super(Arg, self).__init__(uni, res_num)
        self.furthest_atom_type = 'name NH1'
        self.assign_state = self.assign_four_state
        self.num_states = 81
        self.dihedral_atom_list = ['N', 'CA', 'CB', 'CG', 'CD', 'NE', 'CZ']


class Ser(Residue):
    def __init__(self, uni, res_num) -> None:
        super(Ser, self).__init__(uni, res_num)
        self.furthest_atom_type = 'name OG'
        self.assign_state = self.assign_one_state
        self.num_states = 3
        self.dihedral_atom_list = ['N', 'CA', 'CB', 'OG']


class Thr(Residue):
    def __init__(self, uni, res_num) -> None:
        super(Thr, self).__init__(uni, res_num)
        self.furthest_atom_type = 'name CG2'
        self.assign_state = self.assign_one_state
        self.num_states = 3
        self.dihedral_atom_list = ['N', 'CA', 'CB', 'OG1']


class Val(Residue):
    def __init__(self, uni, res_num) -> None:
        super(Val, self).__init__(uni, res_num)
        self.furthest_atom_type = 'name CG1'
        self.assign_state = self.assign_one_state
        self.num_states = 3
        self.dihedral_atom_list = ['N', 'CA', 'CB', 'CG2']


class Trp(Residue):
    def __init__(self, uni, res_num) -> None:
        super(Trp, self).__init__(uni, res_num)
        self.furthest_atom_type = 'name CH2'
        self.assign_state = self.assign_two_state
        self.num_states = 9
        self.dihedral_atom_list = ['N', 'CA', 'CB', 'CG', 'CD2']


class Tyr(Residue):
    def __init__(self, uni, res_num) -> None:
        super(Tyr, self).__init__(uni, res_num)
        self.furthest_atom_type = 'name OH'
        self.assign_state = self.assign_two_state
        self.num_states = 9
        self.dihedral_atom_list = ['N', 'CA', 'CB', 'CG', 'CD1']


def find_csv_path(file_path):
    for final_file_or_dir in os.listdir(file_path):
        if '.csv' in final_file_or_dir:
            csv_path = file_path + final_file_or_dir
            return csv_path
    return False


def get_rows_for_csv(csv_file):
    with open(csv_file) as csv_file:
        csv_reader = csv.reader(csv_file, delimiter=',')
        list_of_column_names = []
        for row in csv_reader:
            list_of_column_names.append(row)
            break
    return list_of_column_names[0]


def obtain_cols_to_add(file_path, features_list):
    possible_csv_path = find_csv_path(file_path)
    if isinstance(find_csv_path(file_path), bool):
        return features_list
    else:
        revised_feat_list = []
        curr_rows = get_rows_for_csv(possible_csv_path)
        for i in range(len(feature_list)):
            if feature_list[i] not in curr_rows:
                revised_feat_list.append(feature_list[i])
        return revised_feat_list


def create_header_string(feat_list):
    header_string = ''
    for feature in feat_list:
        header_string += feature + ','
    if header_string[0].isalpha():
        return header_string[:len(header_string) - 1]
    return header_string[1:len(header_string) - 1]


def select_residue(uni, res_num):
    residue_string = f'resid {res_num}'
    residue = uni.select_atoms(residue_string)
    return residue


def center_protein_in_box(uni):
    with mda.Writer("/home/annika/md_sims/official_extraction/20_ns_sims_new/new_no_in_memory_center20.dcd", len(uni.atoms)) as dcd_writer:
        for ts in uni.trajectory:
            protein_atoms = uni.select_atoms("protein")
            ag = uni.atoms
            mda.transformations.unwrap(ag)
            centroid = protein_atoms.centroid()
            # Obtain box vectors and calculate center of solvent box.
            box_vectors = ts.triclinic_dimensions
            box_center = np.sum(box_vectors, axis=0) / 2
            translation = box_center - centroid
            # Move the protein to the center of solvent box.
            protein_atoms.translate(translation)
            # Write coordinates to the file.
            dcd_writer.write(uni.atoms)
    print("Everything is centered in the box.")


# Minimizing RMSD.
def superimpose_to_last_frame(uni, ref_uni, top_file):
    for ts in uni.trajectory[-1]:
        for ts in ref_uni.trajectory[-1]:
            mod_traj_filename = '/home/annika/md_sims/official_extraction/20_ns_sims_new/adding_new_wrap_superimposition_20.dcd'
            # Setting select='protein' reduces diffusion.
            aligner = align.AlignTraj(
                uni, ref_uni, select='protein', filename=mod_traj_filename).run()
            uni = mda.Universe(top_file, mod_traj_filename)
            print("Finished superimposition.")
            return uni


def obtain_dihedral_angles(atom_list, res_num_list):
    ags = []
    uni_residues = [select_residue(universe, i) for i in res_num_list]
    for i in range(len(uni_residues)):
        ags.append(uni_residues[i].atoms.select_atoms(
            f"name {atom_list[i][0]} or name {atom_list[i][1]} or name {atom_list[i][2]} or name {atom_list[i][3]}"))
    R = Dihedral(ags).run()
    return R.results.angles


# Paths to obtain all MD related files for feature extraction
base_md_file_path = '/home/annika/md_sims/'
md_base_filepath = f'{base_md_file_path}official_extraction/20_ns_sims_new/'
log_file_path = f'{md_base_filepath}prod_20ns.log'
trajectory_file_path = f'{md_base_filepath}prod_20ns.dcd'
topology_file_path = f'{md_base_filepath}end_20ns.pdb'

# Recording start time.
start_time = time.time()

# Loading data from JSON File
parser = argparse.ArgumentParser()
parser.add_argument("json_filename")
args = parser.parse_args()

json_filename = args.json_filename
# with open('/home/annika/md_sims/official_extraction/official_config.json') as config_file:
with open(json_filename) as config_file:
    data = json.load(config_file)

start_residue = data["start_residue"]
end_residue = data["end_residue"]
num_residues = end_residue - start_residue + 1
time_interval = data["time_interval"]
feature_list = data["features"]
# Quick assertion check to ensure user gives non-zero time interval value.
assert time_interval > 0
feature_list = data["features"]
if 'Residue' not in feature_list or ' Timestep' not in feature_list:
    raise ValueError(
        "List of features in config file must contain 'Residue' and ' Timestep' respectively")
new_csv = data["new_csv"]

real_feat_list = []

universe = mda.Universe(topology_file_path, trajectory_file_path)
protein = universe.select_atoms("protein")
protein.guess_bonds()
num_frames = len(universe.trajectory)
# Obtaining simplified universe based on timestep collection.
if time_interval > 1:
    universe.transfer_to_memory(start=0, stop=len(
        universe.trajectory) + 1, step=time_interval)
    # universe.transfer_to_memory(start=time_interval-1, step=time_interval)
num_timesteps = math.floor(num_frames/time_interval)
final_arr_x_dim = num_timesteps * num_residues
final_arr_y_dim = len(feature_list)

res_name_vals = np.empty((num_residues, num_timesteps), dtype=object)
backbone_pos_vals = np.zeros((num_residues, num_timesteps, 3))
side_chain_centroid_pos_vals = np.zeros((num_residues, num_timesteps, 3))
residue_chain_pos_vals = np.zeros((num_residues, num_timesteps, 3))
farthest_side_chain_vals = np.zeros((num_residues, num_timesteps, 3))
rad_gyration_vals = np.zeros((num_residues, num_timesteps))
all_chi_vals = np.zeros((num_residues, num_timesteps, 4), dtype=object)
entropy_vals = np.zeros((1, num_residues))
state_vals = np.zeros((num_residues, num_timesteps))
mean_pot_energy = np.empty((1,))
var_pot_energy = np.empty((1,))
calculated_mean_var = False
final_frame = np.empty((final_arr_x_dim, final_arr_y_dim), dtype=object)

final_features = feature_list[3:len(feature_list) - 4]
num_residues = end_residue - start_residue + 1
feature_set = len(final_features) * num_residues * 2 + 5
feature_set = np.zeros((1, feature_set), dtype=object)
res_list = [i for i in range(start_residue, end_residue + 1)]
real_feature_list = ['Mutant name']
for i in range(len(res_list)):
    for feature in final_features:
        if 'Entropy' in feature:
            feat_string = f'Entropy Residue {res_list[i]}'
            real_feature_list.append(feat_string)
        else:
            mean_string = 'Mean' + feature + f' Residue {res_list[i]}'
            variance_string = 'Variance' + feature + f' Residue {res_list[i]}'
            real_feature_list.append(mean_string)
            real_feature_list.append(variance_string)
real_feature_list.append("Mean Potential Energy")
real_feature_list.append("Variation Potential Energy")
real_feature_list.append("Mean Radius of Gyration")
real_feature_list.append("Variance in Radius of Gyration")

if not new_csv:
    real_feat_list = obtain_cols_to_add(
        '/home/annika/md_sims/official_extraction/', feature_list)
    for i in range(len(real_feat_list)):
        if '#' and 'Residue' in real_feat_list[i]:
            real_feat_list.remove(real_feat_list[i])
            break
    final_frame = np.empty(
        (final_arr_x_dim, len(real_feat_list)), dtype=object)

class_dict = {
    "ALA": Ala,
    "CYS": Cys,
    "ASP": Asp,
    "GLU": Glu,
    "PHE": Phe,
    "GLY": Gly,
    "HIS": His,
    "ILE": Ile,
    "LYS": Lys,
    "LEU": Leu,
    "MET": Met,
    "ASN": Asn,
    "PRO": Pro,
    "GLN": Gln,
    "ARG": Arg,
    "SER": Ser,
    "THR": Thr,
    "VAL": Val,
    "TRP": Trp,
    "TYR": Tyr
}
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

# Performing centering and alignment respectively.
if data["center"]:
    center_protein_in_box(universe)
    universe = mda.Universe(
        topology_file_path, '/home/annika/md_sims/official_extraction/20_ns_sims_new/new_no_in_memory_center20.dcd')
if data["superimpose"]:
    universe = superimpose_to_last_frame(
        universe, universe, topology_file_path)

time_index = 0
for ts in universe.trajectory:
    for i in range(len(res_list)):
        res_sel = select_residue(universe, res_list[i])
        res_obj = class_dict[res_sel.resnames[0]](
            universe, res_list[i])
        res_name_vals[i, ts.frame] = one_letter_rep_dict.get(
            res_sel.resnames[0])
        backbone_pos_vals[i, time_index, :] = res_obj.backbone_pos
        side_chain_centroid_pos_vals[i, time_index,
                                     :] = res_obj.calc_side_chain_centroid_pos()
        residue_chain_pos_vals[i, time_index,
                               :] = res_obj.calc_residue_centroid()
        if res_sel.resnames[0] == 'GLY':
            farthest_side_chain_vals[i,
                                     time_index, :] = res_obj.backbone_pos
        else:
            farthest_side_chain_vals[i, time_index,
                                     :] = res_obj.get_furthest_position()
        rad_gyration_vals[i, time_index] = res_obj.rad_of_gyration
    time_index += 1


def calculate_entropy(res_list):
    chi1_list = []
    chi2_list = []
    chi3_list = []
    chi4_list = []

    chi1_index_list = []
    chi2_index_list = []
    chi3_index_list = []
    chi4_index_list = []

    for m in range(len(res_list)):
        res_sel = select_residue(universe, res_list[m])
        res_obj = class_dict[res_sel.resnames[0]](
            universe, res_list[m])
        if res_obj.num_states == 3:
            chi1_index_list.append(res_list[m])
            chi1_list.append(res_obj.dihedral_atom_list)
        elif res_obj.num_states == 9:
            chi1_index_list.append(res_list[m])
            chi2_index_list.append(res_list[m])
            chi1_list.append(res_obj.dihedral_atom_list[0:4])
            chi2_list.append(res_obj.dihedral_atom_list[1:])
        elif res_obj.num_states == 27:
            chi1_index_list.append(res_list[m])
            chi2_index_list.append(res_list[m])
            chi3_index_list.append(res_list[m])
            chi1_list.append(res_obj.dihedral_atom_list[0:4])
            chi2_list.append(res_obj.dihedral_atom_list[1:5])
            chi3_list.append(res_obj.dihedral_atom_list[2:])
        elif res_obj.num_states == 81:
            chi1_index_list.append(res_list[m])
            chi2_index_list.append(res_list[m])
            chi3_index_list.append(res_list[m])
            chi4_index_list.append(res_list[m])
            chi1_list.append(res_obj.dihedral_atom_list[0:4])
            chi2_list.append(res_obj.dihedral_atom_list[1:5])
            chi3_list.append(res_obj.dihedral_atom_list[2:6])
            chi4_list.append(res_obj.dihedral_atom_list[3:])

    chi1_index = 0
    chi2_index = 0
    chi3_index = 0
    chi4_index = 0

    try:
        chi1_dihedrals = obtain_dihedral_angles(chi1_list, chi1_index_list)
    except:
        pass
    try:
        chi2_dihedrals = obtain_dihedral_angles(chi2_list, chi2_index_list)
    except:
        pass
    try:
        chi3_dihedrals = obtain_dihedral_angles(chi3_list, chi3_index_list)
    except:
        pass
    try:
        chi4_dihedrals = obtain_dihedral_angles(chi4_list, chi4_index_list)
    except:
        pass

    for m in range(len(res_list)):
        res_sel = select_residue(universe, res_list[m])
        res_obj = class_dict[res_sel.resnames[0]](universe, res_list[m])
        if res_obj.num_states == 0:
            entropy_vals[:, m] = 0
        elif res_obj.num_states == 3:
            chi1_vals_res = chi1_dihedrals[:, chi1_index]
            entropy_vals[:, m] = abs(res_obj.calculate_entropy_one(
                universe, chi1_vals_res))
            chi1_index += 1
        elif res_obj.num_states == 9:
            chi1_vals_res = chi1_dihedrals[:, chi1_index]
            chi2_vals_res = chi2_dihedrals[:, chi2_index]
            entropy_vals[:, m] = abs(res_obj.calculate_entropy_two(
                universe, chi1_vals_res, chi2_vals_res))
            chi1_index += 1
            chi2_index += 1
        elif res_obj.num_states == 27:
            chi1_vals_res = chi1_dihedrals[:, chi1_index]
            chi2_vals_res = chi2_dihedrals[:, chi2_index]
            chi3_vals_res = chi3_dihedrals[:, chi3_index]
            entropy_vals[:, m] = abs(res_obj.calculate_entropy_three(
                universe, chi1_vals_res, chi2_vals_res, chi3_vals_res))
            chi1_index += 1
            chi2_index += 1
            chi3_index += 1
        elif res_obj.num_states == 81:
            chi1_vals_res = chi1_dihedrals[:, chi1_index]
            chi2_vals_res = chi2_dihedrals[:, chi2_index]
            chi3_vals_res = chi3_dihedrals[:, chi3_index]
            chi4_vals_res = chi4_dihedrals[:, chi4_index]
            entropy_vals[:, m] = abs(res_obj.calculate_entropy_four(
                universe, chi1_vals_res, chi2_vals_res, chi3_vals_res, chi4_vals_res))
            chi1_index += 1
            chi2_index += 1
            chi3_index += 1
            chi4_index += 1
    return entropy_vals


def obtain_entropy_state(res_list):
    chi1_list = []
    chi2_list = []
    chi3_list = []
    chi4_list = []

    chi1_index_list = []
    chi2_index_list = []
    chi3_index_list = []
    chi4_index_list = []

    for m in range(len(res_list)):
        res_sel = select_residue(universe, res_list[m])
        res_obj = class_dict[res_sel.resnames[0]](
            universe, res_list[m])
        if res_obj.num_states == 3:
            chi1_index_list.append(res_list[m])
            chi1_list.append(res_obj.dihedral_atom_list)
        elif res_obj.num_states == 9:
            chi1_index_list.append(res_list[m])
            chi2_index_list.append(res_list[m])
            chi1_list.append(res_obj.dihedral_atom_list[0:4])
            chi2_list.append(res_obj.dihedral_atom_list[1:])
        elif res_obj.num_states == 27:
            chi1_index_list.append(res_list[m])
            chi2_index_list.append(res_list[m])
            chi3_index_list.append(res_list[m])
            chi1_list.append(res_obj.dihedral_atom_list[0:4])
            chi2_list.append(res_obj.dihedral_atom_list[1:5])
            chi3_list.append(res_obj.dihedral_atom_list[2:])
        elif res_obj.num_states == 81:
            chi1_index_list.append(res_list[m])
            chi2_index_list.append(res_list[m])
            chi3_index_list.append(res_list[m])
            chi4_index_list.append(res_list[m])
            chi1_list.append(res_obj.dihedral_atom_list[0:4])
            chi2_list.append(res_obj.dihedral_atom_list[1:5])
            chi3_list.append(res_obj.dihedral_atom_list[2:6])
            chi4_list.append(res_obj.dihedral_atom_list[3:])

    chi1_index = 0
    chi2_index = 0
    chi3_index = 0
    chi4_index = 0

    try:
        chi1_dihedrals = obtain_dihedral_angles(chi1_list, chi1_index_list)
    except:
        pass
    try:
        chi2_dihedrals = obtain_dihedral_angles(chi2_list, chi2_index_list)
    except:
        pass
    try:
        chi3_dihedrals = obtain_dihedral_angles(chi3_list, chi3_index_list)
    except:
        pass
    try:
        chi4_dihedrals = obtain_dihedral_angles(chi4_list, chi4_index_list)
    except:
        pass

    for m in range(len(res_list)):
        res_sel = select_residue(universe, res_list[m])
        res_obj = class_dict[res_sel.resnames[0]](universe, res_list[m])
        for ts in universe.trajectory:
            if res_obj.num_states == 0:
                state_vals[m, ts.frame] = 0
            elif res_obj.num_states == 3:
                chi1_val = chi1_dihedrals[ts.frame, chi1_index]
                state_vals[m, ts.frame] = res_obj.assign_state(chi1_val)
            elif res_obj.num_states == 9:
                chi1_val = chi1_dihedrals[ts.frame, chi1_index]
                chi2_val = chi2_dihedrals[ts.frame, chi2_index]
                state_vals[m, ts.frame] = res_obj.assign_state(
                    chi1_val, chi2_val)
            elif res_obj.num_states == 27:
                chi1_val = chi1_dihedrals[ts.frame, chi1_index]
                chi2_val = chi2_dihedrals[ts.frame, chi2_index]
                chi3_val = chi3_dihedrals[ts.frame, chi3_index]
                state_vals[m, ts.frame] = res_obj.assign_state(
                    chi1_val, chi2_val, chi3_val)
            elif res_obj.num_states == 81:
                chi1_val = chi1_dihedrals[ts.frame, chi1_index]
                chi2_val = chi2_dihedrals[ts.frame, chi2_index]
                chi3_val = chi3_dihedrals[ts.frame, chi3_index]
                chi4_val = chi4_dihedrals[ts.frame, chi4_index]
                state_vals[m, ts.frame] = res_obj.assign_state(
                    chi1_val, chi2_val, chi3_val, chi4_val)
        if res_obj.num_states == 3:
            chi1_index += 1
        elif res_obj.num_states == 9:
            chi1_index += 1
            chi2_index += 1
        elif res_obj.num_states == 27:
            chi1_index += 1
            chi2_index += 1
            chi3_index += 1
        elif res_obj.num_states == 81:
            chi1_index += 1
            chi2_index += 1
            chi3_index += 1
            chi4_index += 1
    return state_vals


def obtain_chi_angles(res_list):
    chi1_list = []
    chi2_list = []
    chi3_list = []
    chi4_list = []

    chi1_index_list = []
    chi2_index_list = []
    chi3_index_list = []
    chi4_index_list = []

    for m in range(len(res_list)):
        res_sel = select_residue(universe, res_list[m])
        res_obj = class_dict[res_sel.resnames[0]](
            universe, res_list[m])
        if res_obj.num_states == 3:
            chi1_index_list.append(res_list[m])
            chi1_list.append(res_obj.dihedral_atom_list)
        elif res_obj.num_states == 9:
            chi1_index_list.append(res_list[m])
            chi2_index_list.append(res_list[m])
            chi1_list.append(res_obj.dihedral_atom_list[0:4])
            chi2_list.append(res_obj.dihedral_atom_list[1:])
        elif res_obj.num_states == 27:
            chi1_index_list.append(res_list[m])
            chi2_index_list.append(res_list[m])
            chi3_index_list.append(res_list[m])
            chi1_list.append(res_obj.dihedral_atom_list[0:4])
            chi2_list.append(res_obj.dihedral_atom_list[1:5])
            chi3_list.append(res_obj.dihedral_atom_list[2:])
        elif res_obj.num_states == 81:
            chi1_index_list.append(res_list[m])
            chi2_index_list.append(res_list[m])
            chi3_index_list.append(res_list[m])
            chi4_index_list.append(res_list[m])
            chi1_list.append(res_obj.dihedral_atom_list[0:4])
            chi2_list.append(res_obj.dihedral_atom_list[1:5])
            chi3_list.append(res_obj.dihedral_atom_list[2:6])
            chi4_list.append(res_obj.dihedral_atom_list[3:])

    chi1_index = 0
    chi2_index = 0
    chi3_index = 0
    chi4_index = 0

    try:
        chi1_dihedrals = obtain_dihedral_angles(chi1_list, chi1_index_list)
    except:
        pass
    try:
        chi2_dihedrals = obtain_dihedral_angles(chi2_list, chi2_index_list)
    except:
        pass
    try:
        chi3_dihedrals = obtain_dihedral_angles(chi3_list, chi3_index_list)
    except:
        pass
    try:
        chi4_dihedrals = obtain_dihedral_angles(chi4_list, chi4_index_list)
    except:
        pass

    for m in range(len(res_list)):
        res_sel = select_residue(universe, res_list[m])
        res_obj = class_dict[res_sel.resnames[0]](universe, res_list[m])
        time_index = 0
        for ts in universe.trajectory:
            if res_obj.num_states == 0:
                all_chi_vals[m, time_index, 0] = "NA"
                all_chi_vals[m, time_index, 1] = "NA"
                all_chi_vals[m, time_index, 2] = "NA"
                all_chi_vals[m, time_index, 3] = "NA"
                time_index += 1
            elif res_obj.num_states == 3:
                chi1_val = chi1_dihedrals[ts.frame, chi1_index]
                all_chi_vals[m, time_index, 0] = chi1_val
                all_chi_vals[m, time_index, 1] = "NA"
                all_chi_vals[m, time_index, 2] = "NA"
                all_chi_vals[m, time_index, 3] = "NA"
                time_index += 1
            elif res_obj.num_states == 9:
                chi1_val = chi1_dihedrals[ts.frame, chi1_index]
                chi2_val = chi2_dihedrals[ts.frame, chi2_index]
                all_chi_vals[m, time_index, 0] = chi1_val
                all_chi_vals[m, time_index, 1] = chi2_val
                all_chi_vals[m, time_index, 2] = "NA"
                all_chi_vals[m, time_index, 3] = "NA"
                time_index += 1
            elif res_obj.num_states == 27:
                chi1_val = chi1_dihedrals[ts.frame, chi1_index]
                chi2_val = chi2_dihedrals[ts.frame, chi2_index]
                chi3_val = chi3_dihedrals[ts.frame, chi3_index]
                all_chi_vals[m, time_index, 0] = chi1_val
                all_chi_vals[m, time_index, 1] = chi2_val
                all_chi_vals[m, time_index, 2] = chi3_val
                all_chi_vals[m, time_index, 3] = "NA"
                time_index += 1
            elif res_obj.num_states == 81:
                chi1_val = chi1_dihedrals[ts.frame, chi1_index]
                chi2_val = chi2_dihedrals[ts.frame, chi2_index]
                chi3_val = chi3_dihedrals[ts.frame, chi3_index]
                chi4_val = chi4_dihedrals[ts.frame, chi4_index]
                all_chi_vals[m, time_index, 0] = chi1_val
                all_chi_vals[m, time_index, 1] = chi2_val
                all_chi_vals[m, time_index, 2] = chi3_val
                all_chi_vals[m, time_index, 3] = chi4_val
                time_index += 1
        if res_obj.num_states == 3:
            chi1_index += 1
        elif res_obj.num_states == 9:
            chi1_index += 1
            chi2_index += 1
        elif res_obj.num_states == 27:
            chi1_index += 1
            chi2_index += 1
            chi3_index += 1
        elif res_obj.num_states == 81:
            chi1_index += 1
            chi2_index += 1
            chi3_index += 1
            chi4_index += 1
    return all_chi_vals


def calc_disp_same_array(arr, final_index_to_update):
    for j in range(arr.shape[0]):
        ref_val = arr[j, arr.shape[1] - 1, :]
        for k in range(arr.shape[1]):
            displacement = np.linalg.norm(np.subtract(arr[j, k, :], ref_val))
            index_to_place = j * num_timesteps + k
            final_frame[index_to_place, final_index_to_update] = displacement


def calc_disp_diff_array(arr1, arr2, final_index_to_update):
    for j in range(arr1.shape[0]):
        for k in range(arr1.shape[1]):
            displacement = np.linalg.norm(
                np.subtract(arr1[j, k, :], arr2[j, k, :]))
            index_to_place = j * num_timesteps + k
            final_frame[index_to_place, final_index_to_update] = displacement


def obtain_x_coords(coords_arr, final_index_to_update):
    assert coords_arr.shape[2] == 3
    for j in range(coords_arr.shape[0]):
        for k in range(coords_arr.shape[1]):
            index_to_place = j * num_timesteps + k
            final_frame[index_to_place,
                        final_index_to_update] = coords_arr[j, k, 0]


def obtain_y_coords(coords_arr, final_index_to_update):
    assert coords_arr.shape[2] == 3
    for j in range(coords_arr.shape[0]):
        for k in range(coords_arr.shape[1]):
            index_to_place = j * num_timesteps + k
            final_frame[index_to_place,
                        final_index_to_update] = coords_arr[j, k, 1]
    return final_frame


def obtain_z_coords(coords_arr, final_index_to_update):
    assert coords_arr.shape[2] == 3
    for j in range(coords_arr.shape[0]):
        for k in range(coords_arr.shape[1]):
            index_to_place = j * num_timesteps + k
            final_frame[index_to_place,
                        final_index_to_update] = coords_arr[j, k, 2]
    return final_frame


def obtain_mean_var_potential_energies(log_file):
    global mean_pot_energy, var_pot_energy, calculated_mean_var
    if calculated_mean_var:
        return (mean_pot_energy, var_pot_energy)
    potential_energy_list = []
    with open(log_file, 'r') as md_logfile:
        for line in md_logfile:
            if line[0] != "#":
                potential_energy_val = float(list(line.split(','))[1])
                potential_energy_list.append(potential_energy_val)
    np_pot_energy_list = np.asarray(potential_energy_list)
    mean_pot_energy = np.mean(np_pot_energy_list)
    var_pot_energy = np.var(np_pot_energy_list)
    calculated_mean_var = True
    return (mean_pot_energy, var_pot_energy)


def place_res_nums(res_list, final_index_to_update):
    for j in range(len(res_list)):
        for k in range(num_timesteps):
            index_to_place = j * num_timesteps + k
            res_num = res_list[j]
            final_frame[index_to_place, final_index_to_update] = res_num


def place_res_name_vals(res_name_arr, final_index_to_update):
    for j in range(res_name_arr.shape[0]):
        for k in range(res_name_arr.shape[1]):
            index_to_place = j * num_timesteps + k
            final_frame[index_to_place,
                        final_index_to_update] = res_name_arr[j, k]


def place_ts_intervals(res_list, final_index_to_update):
    for j in range(len(res_list)):
        for k in range(num_timesteps):
            index_to_place = j * num_timesteps + k
            ts_interval = (k) * time_interval
            # ts_interval = (k + 1) * time_interval
            final_frame[index_to_place, final_index_to_update] = ts_interval


def place_entropy_nums(res_list, final_index_to_update):
    entropy_arr = calculate_entropy(res_list)
    for j in range(len(res_list)):
        for k in range(num_timesteps):
            index_to_place = j * num_timesteps + k
            final_frame[index_to_place,
                        final_index_to_update] = entropy_arr[0, j]


def place_state_vals(res_list, final_index_to_update):
    state_arr = obtain_entropy_state(res_list)
    for j in range(len(res_list)):
        for k in range(num_timesteps):
            index_to_place = j * num_timesteps + k
            final_frame[index_to_place,
                        final_index_to_update] = state_arr[j, k]


def place_chi1_vals(res_list, final_index_to_update):
    chi_arr = obtain_chi_angles(res_list)
    for j in range(len(res_list)):
        for k in range(num_timesteps):
            index_to_place = j * num_timesteps + k
            final_frame[index_to_place,
                        final_index_to_update] = chi_arr[j, k, 0]


def place_chi2_vals(res_list, final_index_to_update):
    chi_arr = obtain_chi_angles(res_list)
    for j in range(len(res_list)):
        for k in range(num_timesteps):
            index_to_place = j * num_timesteps + k
            final_frame[index_to_place,
                        final_index_to_update] = chi_arr[j, k, 1]


def place_chi3_vals(res_list, final_index_to_update):
    chi_arr = obtain_chi_angles(res_list)
    for j in range(len(res_list)):
        for k in range(num_timesteps):
            index_to_place = j * num_timesteps + k
            final_frame[index_to_place,
                        final_index_to_update] = chi_arr[j, k, 2]


def place_chi4_vals(res_list, final_index_to_update):
    chi_arr = obtain_chi_angles(res_list)
    for j in range(len(res_list)):
        for k in range(num_timesteps):
            index_to_place = j * num_timesteps + k
            final_frame[index_to_place,
                        final_index_to_update] = chi_arr[j, k, 3]


def place_mean_pot_energy(res_list, final_index_to_update):
    mean_pot_energy_value = obtain_mean_var_potential_energies(log_file_path)[
        0]
    for j in range(len(res_list)):
        for k in range(num_timesteps):
            index_to_place = j * num_timesteps + k
            final_frame[index_to_place,
                        final_index_to_update] = mean_pot_energy_value


def place_var_pot_energy(res_list, final_index_to_update):
    var_pot_energy_value = obtain_mean_var_potential_energies(log_file_path)[1]
    for j in range(len(res_list)):
        for k in range(num_timesteps):
            index_to_place = j * num_timesteps + k
            final_frame[index_to_place,
                        final_index_to_update] = var_pot_energy_value


def place_rad_gyration_vals(rad_array, final_index_to_update):
    for j in range(rad_array.shape[0]):
        for k in range(rad_array.shape[1]):
            index_to_place = j * num_timesteps + k
            final_frame[index_to_place,
                        final_index_to_update] = rad_array[j, k]


cols_to_funcs_dict = {"Residue": [place_res_nums, res_list],
                      " Residue Name": [place_res_name_vals, res_name_vals],
                      " Timestep": [place_ts_intervals, res_list],
                      " Backbone Displacement": [calc_disp_same_array, backbone_pos_vals],
                      " Residue Centroid Displacement": [calc_disp_same_array, residue_chain_pos_vals],
                      " Side Chain Centroid Displacement": [calc_disp_same_array, side_chain_centroid_pos_vals],
                      " Backbone to Side Chain Displacement": [calc_disp_diff_array, backbone_pos_vals, farthest_side_chain_vals],
                      " Backbone x Position": [obtain_x_coords, backbone_pos_vals],
                      " Backbone y Position": [obtain_y_coords, backbone_pos_vals],
                      " Backbone z Position": [obtain_z_coords, backbone_pos_vals],
                      " Residue Chain Centroid x Position": [obtain_x_coords, residue_chain_pos_vals],
                      " Residue Chain Centroid y Position": [obtain_y_coords, residue_chain_pos_vals],
                      " Residue Chain Centroid z Position": [obtain_z_coords, residue_chain_pos_vals],
                      " Side Chain Centroid x Position": [obtain_x_coords, side_chain_centroid_pos_vals],
                      " Side Chain Centroid y Position": [obtain_y_coords, side_chain_centroid_pos_vals],
                      " Side Chain Centroid z Position": [obtain_z_coords, side_chain_centroid_pos_vals],
                      " Chi1": [place_chi1_vals, res_list],
                      " Chi2": [place_chi2_vals, res_list],
                      " Chi3": [place_chi3_vals, res_list],
                      " Chi4": [place_chi4_vals, res_list],
                      " Entropy": [place_entropy_nums, res_list],
                      " Entropy State": [place_state_vals, res_list],
                      " Mean Potential Energy": [place_mean_pot_energy, res_list],
                      " Variance in Potential Energy": [place_var_pot_energy, res_list],
                      " Radius of Gyration": [place_rad_gyration_vals, rad_gyration_vals]}


def extract_md_features(feat_list):
    for i in range(len(feat_list)):
        print(f"Working on extracting {feat_list[i]}")
        if cols_to_funcs_dict.get(feat_list[i])[0].__name__ == 'calc_disp_diff_array':
            cols_to_funcs_dict[feat_list[i]][0](
                cols_to_funcs_dict[feat_list[i]][1], cols_to_funcs_dict[feat_list[i]][2], i)
        else:
            cols_to_funcs_dict[feat_list[i]][0](
                cols_to_funcs_dict[feat_list[i]][1], i)


if new_csv:
    extract_md_features(feature_list)
    np.savetxt(data["new_csv_name"], final_frame, delimiter=',',
               header=create_header_string(feature_list), fmt='%s')
else:
    extract_md_features(real_feat_list)
    csv_to_add = find_csv_path('/home/annika/md_sims/official_extraction/')
    old_header_list = get_rows_for_csv(csv_to_add)
    new_header_list = old_header_list + real_feat_list
    header_string = create_header_string(
        get_rows_for_csv(csv_to_add)).replace(" ", "")
    existing_data = np.loadtxt(csv_to_add, delimiter=',', dtype=object)
    updated_data = np.concatenate((existing_data, final_frame), axis=1)
    np.savetxt(csv_to_add, updated_data, delimiter=',',
               header=create_header_string(new_header_list), fmt='%s')

time_dep_frame = pd.read_csv(data["new_csv_name"])
counter = 0
feature_set[0, counter] = data["md_mutant_name"]
counter += 1
for i in range(len(res_list)):
    res_frame = time_dep_frame[time_dep_frame['# Residue'] == res_list[i]]
    for j in range(len(final_features)):
        if 'Entropy' in final_features:
            feature_set[0, counter] = np.mean(
                np.asarray(res_frame[final_features[j]]), axis=0)
            counter += 1
        else:
            feature_set[0, counter] = np.mean(
                np.asarray(res_frame[final_features[j]]), axis=0)
            counter += 1
            feature_set[0, counter] = np.var(
                np.asarray(res_frame[final_features[j]]), axis=0)
            counter += 1
res_frame = time_dep_frame[time_dep_frame['# Residue'] == (start_residue)]
feature_set[0, counter] = res_frame[' Mean Potential Energy'][0]
counter += 1
feature_set[0, counter] = res_frame[' Variance in Potential Energy'][0]
counter += 1
feature_set[0, counter] = np.mean(np.asarray(
    res_frame[' Radius of Gyration']), axis=0)
counter += 1
feature_set[0, counter] = np.var(np.asarray(
    res_frame[' Radius of Gyration']), axis=0)

# TODO: change .csv pathname to be including the md_mutant_name.
np.savetxt('/home/annika/md_sims/official_extraction/test2.csv', feature_set, delimiter=',',
           header=create_header_string(real_feature_list), fmt='%s')

print("Feature .csv file saved.")

# Record the end time
end_time = time.time()

# Print out the elapsed time
print("Elapsed time: %.2f seconds" % (end_time - start_time))
