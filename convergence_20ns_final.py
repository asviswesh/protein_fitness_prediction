# %%
import MDAnalysis as mda
from MDAnalysis.analysis import rms
from MDAnalysis.analysis.rms import RMSD
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# %%
ref_traj = '/home/annika/md_sims/official_extraction/20_ns_sims_new/end_20ns.pdb'
new_traj_file = '/home/annika/md_sims/official_extraction/20_ns_sims_new/adding_new_wrap_superimposition_20.dcd'

# %%
u_ref = mda.Universe(ref_traj, new_traj_file)

# %%
rmsd_list = []
bb = u_ref.select_atoms("protein")
print(type(bb))
for i in range(len(u_ref.trajectory)):
    u_ref.trajectory[i]
    A = bb.positions.copy()  # coordinates of first frame
    u_ref.trajectory[-1]         # forward to last frame
    B = bb.positions.copy()  # coordinates of last frame
    rmsd_list.append(rms.rmsd(A, B))

# %%
# Extract RMSD values and timesteps
# rmsd_values = rmsd_analysis.rmsd[:, 2]  # Third column contains the RMSD values
# print(rmsd_values.shape)
rmsd_values = np.asarray(rmsd_list)
timesteps = np.arange(len(rmsd_list)) * 0.0002

# %%
time_step_list = []
for i in range(len(rmsd_values)):
    time_step_list.append((i) * 250 * 0.0002)
print(time_step_list)

# %%
# Plot RMSD values
# Use timesteps for entire 100,000 frameset.
plt.figure(figsize=(10, 6))
plt.plot(time_step_list, rmsd_values, label="RMSD")
plt.xticks(np.arange(min(time_step_list), max(time_step_list)+1, 1.0))
plt.xlabel("Time (ns)", fontsize=12)
plt.ylabel("RMSD fit to final trajectory frame (Å)", fontsize=12)
plt.title("RMSD Convergence Test")
plt.legend()
plt.grid(True)
plt.show()

# %%
avg_rmsd_vals = np.zeros((len(rmsd_list),))
def obtain_avg_rmsd_values(rmsd_vals, avg_rmsd_values, ts_interval):
    index = 0
    step_range = int(avg_rmsd_values.shape[0] / ts_interval)
    for i in range(step_range):
        temp_val = ts_interval * (i + 1)
        another_temp_val = ts_interval * i
        mean_rmsd = np.mean(rmsd_vals[index:temp_val])
        for i in range(ts_interval):
            avg_rmsd_values[another_temp_val + i] = mean_rmsd
        index += ts_interval

# %%
def plot_rolling_mean_rmsd_plot(rmsd_vals, avg_rmsd_val, ts_interval, ts_list):
    obtain_avg_rmsd_values(rmsd_vals, avg_rmsd_val, ts_interval)
    plt.figure(figsize=(10, 6))
    plt.plot(ts_list, avg_rmsd_val, label="RMSD")
    plt.xlabel("Time (ns)")
    plt.xticks(np.arange(min(ts_list), max(ts_list)+1, 1.0))
    plt.ylabel("Mean RMSD (Å)")
    plt.title(f"20 ns Rolling Mean RMSD Convergence Test after every {ts_interval} steps")
    plt.legend()
    plt.grid(True)
    plt.savefig(f"/home/annika/md_sims/official_extraction/convergence_plots_20ns/rmsd_20ns_converge_{ts_interval}_frames.png")

# %%
plot_rolling_mean_rmsd_plot(rmsd_values, avg_rmsd_vals, 50, time_step_list)

# %%
plot_rolling_mean_rmsd_plot(rmsd_values, avg_rmsd_vals, 250, time_step_list)

# %%
plot_rolling_mean_rmsd_plot(rmsd_values, avg_rmsd_vals, 500, time_step_list)

# %%
plot_rolling_mean_rmsd_plot(rmsd_values, avg_rmsd_vals, 1000, time_step_list)

# %%
plot_rolling_mean_rmsd_plot(rmsd_values, avg_rmsd_vals, 2000, time_step_list)

# %%
plot_rolling_mean_rmsd_plot(rmsd_values, avg_rmsd_vals, 4000)

# %%
plot_rolling_mean_rmsd_plot(rmsd_values, avg_rmsd_vals, 5000)

# %%
import Bio.PDB
from Bio.PDB.Polypeptide import PPBuilder
pdb_file_extension = '.pdb'
pdb_file_coords = '/home/annika/md_sims/official_extraction/20_ns_sims_new/end_20ns.pdb'
def get_sequence(pdb_file_coords):
    seq_str = ''
    pdb_to_align_file = pdb_file_coords[:len(pdb_file_coords) - 4] + \
                        pdb_file_extension
    pdb_parser = Bio.PDB.PDBParser(QUIET=True)
    structure = pdb_parser.get_structure('struct', pdb_to_align_file)
    ppb=PPBuilder()
    for pp in ppb.build_peptides(structure):
        seq_str = pp.get_sequence()
    return seq_str
def get_distinct_residues(sequence_string):
    res_dict = {}
    for res_index in range(len(sequence_string)):
        found_res_already = False
        if sequence_string[res_index] in res_dict:
            found_res_already = True
        if not found_res_already:
            res_dict.update({sequence_string[res_index]: (res_index + 1)})
    return list(res_dict.values())
seq_str = get_sequence(pdb_file_coords)
res_list = get_distinct_residues(seq_str)

# %%
from collections import Counter
def create_frequency_dict(state_list):
    frequency_counter = Counter(state_list)
    frequency_dict = dict(frequency_counter)
    for key in frequency_dict.keys():
        probability_value = frequency_dict.get(
            key)/(len(state_list))
        probability_entry = {key: probability_value}
        frequency_dict.update(probability_entry)
    return frequency_dict

# %%
def calculate_entropy(freq_dict):
    entropy_value = 0
    for key in freq_dict:
        p_xi = freq_dict.get(key)
        entropy_value += (p_xi * np.log(p_xi))
    return abs(entropy_value)

# %%
def plot_entropy_vals(res_num, res_frame, ts_interval):
    state_cols = res_frame[' Entropy State']
    state_array = list(state_cols.to_numpy())
    res_name = list((res_frame[' Residue Name']).to_numpy())[0]
    final_state_array = []
    entropy_list = []
    for i in range(len(state_array)):
        if (i + 1) % ts_interval == 0:
            final_state_array.append(state_array[i])
            frequency_dict = create_frequency_dict(final_state_array)
            entropy = calculate_entropy(frequency_dict)
            entropy_list.append(entropy)
    index = 0
    step_range = int(len(state_array) / ts_interval)
    real_entropy_vals = np.empty((len(state_array),))
    for i in range(step_range):
        another_temp_val = ts_interval * i
        for j in range(ts_interval):
            real_entropy_vals[another_temp_val + j] = entropy_list[i]
        index += ts_interval
    real_entropy_value = res_frame[' Entropy']
    plt.figure()
    plt.title(f"Entropy for Residue {res_num} (aka {res_name}) collected at every 250 timesteps for 20 ns")
    plt.plot(time_step_list, real_entropy_vals, label="Entropy")
    # plt.plot(time_step_list, real_entropy_value, label="Final calculated entropy")
    plt.xticks(np.arange(min(time_step_list), max(time_step_list)+1, 2.0), fontsize=8)
    plt.yticks(np.arange(min(real_entropy_vals), max(real_entropy_vals)+1, 0.4), fontsize=8)
    plt.xlabel("Time (ns)")
    plt.ylabel("Entropy value")
    plt.legend()
    plt.grid(True)
    plt.savefig(f"/home/annika/md_sims/official_extraction/convergence_plots_20ns_all/entropy_20ns_{res_num}_converge_250_frames.png")

# %%
time_dep_frame = pd.read_csv('/home/annika/md_sims/official_extraction/convergence_values_20_all.csv')
# res_frame = time_dep_frame[time_dep_frame['# Residue'] == 1.0]
# plot_entropy_vals(1.0, res_frame, 1)
less_sussy_list = [2, 3, 45, 4, 10, 13, 5, 7, 12, 39, 8, 45, 9, 14, 18, 15, 20, 23, 21, 22, 40, 30, 43, 1]
for residue in less_sussy_list:
    res_frame = time_dep_frame[time_dep_frame['# Residue'] == residue]
    plot_entropy_vals(residue, res_frame, 1)



# %%
time_dep_frame = pd.read_csv('/home/annika/md_sims/official_extraction/convergence_values_20_9-16.csv')
res_frame = time_dep_frame[time_dep_frame['# Residue'] == 13.0]
plot_entropy_vals(13, res_frame, 1)

# %%
res_list = [i for i in range(1, 2)]
print(res_list)
time_dep_frame = pd.read_csv('/home/annika/md_sims/official_extraction/convergence_values_20_9-16.csv')
for i in range(len(res_list)):
    res_num = res_list[i] + 0.0
    residue_frame = time_dep_frame[time_dep_frame['# Residue'] == res_num]
    plot_entropy_vals(res_list[i], residue_frame, 1)
    plot_entropy_vals(res_list[i], residue_frame, 125)
    # plot_entropy_vals(res_list[i], residue_frame, 250)
    # plot_entropy_vals(res_list[i], residue_frame, 500)
    # plot_entropy_vals(res_list[i], residue_frame, 1000)
    # plot_entropy_vals(res_list[i], residue_frame, 1250)
    # plot_entropy_vals(res_list[i], residue_frame, 2000)
    # plot_entropy_vals(res_list[i], residue_frame, 4000)
    # plot_entropy_vals(res_list[i], residue_frame, 5000)


# %%
time_dep_frame = pd.read_csv('/home/annika/md_sims/official_extraction/convergence_values_20_9-16.csv')
res_frame = time_dep_frame[time_dep_frame['# Residue'] == 10.0]
res_frame[' Radius of Gyration']

# %%
def plot_rolling_entropy(window_interval, step_time, res_num):
    # window interval - ex. 1 ns
    # step_time - collect every 0.1 ns at 1 ns time intervals.
    global rmsd_list
    x_points_list = []
    entropy_vals_list = []
    res_num = res_num + 0.0
    res_frame = time_dep_frame[time_dep_frame['# Residue'] == res_num]
    interval = int(len(rmsd_list)/(len(rmsd_list)* 0.0002) * window_interval)
    step_time_interval = int(len(rmsd_list)/(len(rmsd_list)* 0.0002) * step_time)
    plt.figure()
    plt.title(f"Rolling entropy for Residue {res_num} Collected after every {window_interval} ns interval for 20 ns")
    for i in range(0, len(rmsd_list) - interval, step_time_interval):
        state_list = list(res_frame[' Entropy State'][i: i + interval])
        freq_dict = create_frequency_dict(state_list)
        print(freq_dict)
        temp_arr = np.arange(len(rmsd_list) + 1) * 0.0002
        temp_ts_list = (np.arange(len(rmsd_list) + 1) * 0.0002)[i: i + interval]
        # print(f"The start time is {temp_ts_list[0]} and the end time is {temp_ts_list[-1]}")
        x_point_to_plot = (temp_ts_list[0] + temp_ts_list[-1]) / 2
        entropy_val = calculate_entropy(freq_dict)
        x_points_list.append(x_point_to_plot)
        entropy_vals_list.append(entropy_val)
    full_state_list = list(res_frame[' Entropy State'])
    freq_dict = create_frequency_dict(full_state_list)
    print(freq_dict)
    entire_entropy = np.asarray([calculate_entropy(freq_dict)] * len(rmsd_list))
    entire_ts_list = (np.arange(len(rmsd_list)) * 0.0002)
    plt.plot(x_points_list, entropy_vals_list, color='red', marker='o', markerfacecolor='blue', markersize=3) 
    plt.plot(entire_ts_list, entire_entropy, color="green")
    plt.grid(True)
    plt.xlabel("Time (ns)")
    plt.ylabel("Entropy value")
    plt.savefig(f"/home/annika/md_sims/official_extraction/rolling_entropy_20ns/entropy_20ns_{res_num}_{step_time}_{window_interval}_frames.png")

# %%
# Entropy flucutates significantly, why is that? Should we also plot radius of gyration.
plot_rolling_entropy(10, 0.02, 12)

# %%
def calculate_scaled_entropy(freq_dict):
    # Entropy calculation working correctly.
    entropy_value = 0
    num_states = len(freq_dict)
    for key in freq_dict:
        p_xi = freq_dict.get(key)
        try:
            print(f"p_xi is {p_xi}, log is {np.log(p_xi)}, and number of states is {num_states}")
            entropy_value += (p_xi * np.log(p_xi)) / np.log(num_states)
            print(f"Adding {(p_xi * np.log(p_xi)) / np.log(num_states)} to the current entropy value.")
            print(f"The current entropy value is {entropy_value}")
        except:
            entropy_value += 0.0
    return abs(entropy_value)

# %%
def plot_scaled_rolling_entropy(window_interval, step_time, res_num):
    # window interval - ex. 1 ns
    # step_time - collect every 0.1 ns at 1 ns time intervals.
    global rmsd_list
    x_points_list = []
    entropy_vals_list = []
    res_num = res_num + 0.0
    res_frame = time_dep_frame[time_dep_frame['# Residue'] == res_num]
    interval = int(len(rmsd_list)/(len(rmsd_list)* 0.0002) * window_interval)
    step_time_interval = int(len(rmsd_list)/(len(rmsd_list)* 0.0002) * step_time)
    plt.figure()
    plt.title(f"Scaled rolling entropy for Residue {res_num} Collected after every {window_interval} ns interval for 20 ns")
    for i in range(0, len(rmsd_list) - interval, step_time_interval):
        state_list = list(res_frame[' Entropy State'][i: i + interval])
        freq_dict = create_frequency_dict(state_list)
        print(freq_dict)
        temp_ts_list = (np.arange(len(rmsd_list)) * 0.0002)[i: i + interval]
        x_point_to_plot = (temp_ts_list[0] + temp_ts_list[-1]) / 2
        entropy_val = calculate_scaled_entropy(freq_dict)
        x_points_list.append(x_point_to_plot)
        entropy_vals_list.append(entropy_val)
    full_state_list = list(res_frame[' Entropy State'])
    freq_dict = create_frequency_dict(full_state_list)
    entire_entropy = np.asarray([calculate_scaled_entropy(freq_dict)] * len(rmsd_list))
    entire_ts_list = (np.arange(len(rmsd_list)) * 0.0002)
    plt.plot(x_points_list, entropy_vals_list, color='red', marker='o', markerfacecolor='blue', markersize=3) 
    plt.plot(entire_ts_list, entire_entropy, color="green", label="Scaled Shannon entropy")
    plt.grid(True)
    plt.xlabel("Time (ns)")
    plt.ylabel("Entropy value")
    plt.legend()
    plt.savefig(f"/home/annika/md_sims/official_extraction/rolling_entropy_20ns/entropy_scaled_20ns_{res_num}_{step_time}_{window_interval}_frames.png")

# %%
plot_scaled_rolling_entropy(10, 0.02, 12)

# %%
plot_scaled_rolling_entropy(7.5, 0.02, 35)

# %%
plot_scaled_rolling_entropy(1, 0.2, 35)

# %%
def plot_rolling_gyration(window_interval, step_time, res_num):
    # window interval - ex. 1 ns
    # step_time - collect every 0.1 ns at 1 ns time intervals.
    global rmsd_list
    x_points_list = []
    print(len(x_points_list))
    gyration_list = []
    res_num = res_num + 0.0
    res_frame = time_dep_frame[time_dep_frame['# Residue'] == res_num]
    interval = int(len(rmsd_list)/(len(rmsd_list)* 0.0002) * window_interval)
    step_time_interval = int(len(rmsd_list)/(len(rmsd_list)* 0.0002) * step_time)
    plt.figure()
    plt.title(f"Rolling radius of gyration collected after every {window_interval} ns interval for 20 ns")
    for i in range(0, len(rmsd_list) - interval, step_time_interval):
        gyration_vals = list(res_frame[' Radius of Gyration'][i: i + interval])
        mean_gyration_vals = np.mean(gyration_vals)
        temp_ts_list = (np.arange(len(rmsd_list)) * 0.0002)[i: i + interval]
        x_point_to_plot = (temp_ts_list[0] + temp_ts_list[-1]) / 2
        x_points_list.append(x_point_to_plot)
        gyration_list.append(mean_gyration_vals)
    full_state_list = np.asarray(list(res_frame[' Radius of Gyration']))
    mean_gyration_all = [np.mean(full_state_list)] * len(rmsd_list)
    entire_ts_list = (np.arange(len(rmsd_list)) * 0.0002)
    plt.plot(x_points_list, gyration_list, color='red', marker='o', markerfacecolor='blue', markersize=3) 
    plt.plot(entire_ts_list, mean_gyration_all, color="green")
    plt.grid(True)
    plt.xlabel("Time (ns)")
    plt.ylabel("Radius of Gyration (nm)")
    plt.savefig(f"/home/annika/md_sims/official_extraction/rolling_gyration_20ns/rad_gyration_scaled_20ns_{res_num}_{step_time}_{window_interval}_frames.png")
    

# %%
plot_rolling_gyration(10, 0.02, 11)

# %%
res_frame = time_dep_frame[time_dep_frame['# Residue'] == 11.0]
full_state_list = np.asarray(list(res_frame[' Radius of Gyration']))
plt.figure()
plt.title(f"Radius of Gyration 20 ns")
plt.plot(timesteps, full_state_list, label="Radius of Gyration (nm)")
plt.yticks(np.arange(min(full_state_list), max(full_state_list)+1, 0.4), fontsize=8)
plt.xlabel("Time (ns)")
plt.ylabel("Entropy value")
plt.legend()
plt.grid(True)
plt.savefig(f"/home/annika/md_sims/official_extraction/convergence_plots_20ns/rad_gyration.png")

# %%
def plot_rolling_rmsd(window_interval, step_time, res_num):
    # window interval - ex. 1 ns
    # step_time - collect every 0.1 ns at 1 ns time intervals.
    global rmsd_list
    x_points_list = []
    rmsd_values_list = []
    interval = int(len(rmsd_list)/(len(rmsd_list)* 0.0002) * window_interval)
    step_time_interval = int(len(rmsd_list)/(len(rmsd_list)* 0.0002) * step_time)
    plt.figure()
    plt.title(f"Rolling RMSD collected after every {window_interval} ns interval for 20 ns")
    for i in range(0, len(rmsd_list) - interval, step_time_interval):
        rmsd_vals = rmsd_list[i: i + interval]
        mean_gyration_vals = np.mean(rmsd_vals)
        temp_ts_list = (np.arange(len(rmsd_list)) * 0.0002)[i: i + interval]
        x_point_to_plot = (temp_ts_list[0] + temp_ts_list[-1]) / 2
        x_points_list.append(x_point_to_plot)
        rmsd_values_list.append(mean_gyration_vals)
    mean_rmsd = [np.mean(rmsd_list)] * len(rmsd_list)
    entire_ts_list = (np.arange(len(rmsd_list)) * 0.0002)
    plt.plot(x_points_list, rmsd_values_list, color='red', marker='o', markerfacecolor='blue', markersize=3) 
    plt.plot(entire_ts_list, mean_rmsd, color="green")
    plt.grid(True)
    plt.xlabel("Time (ns)")
    plt.ylabel("RMSD")
    plt.savefig(f"/home/annika/md_sims/official_extraction/rolling_rmsd/rmsd_scaled_20ns_{res_num}_{step_time}_{window_interval}_frames.png")
    

# %%
plot_rolling_rmsd(10, 0.02, 1)

# %%
# Perform convergence test (same code as before)
def convergence_test(values, threshold=0.1, window_size=10):
    for i in range(window_size, len(values)):
        window_mean = np.mean(values[i - window_size : i])
        if np.abs(window_mean - values[i]) > threshold:
            return False
    return True

# %%
# Define convergence parameters
convergence_threshold = 0.1
convergence_window_size = 10

# %%
# Perform the convergence test
is_converged = convergence_test(rmsd_values, convergence_threshold, convergence_window_size)

if is_converged:
    print("Convergence achieved.")
else:
    print("Convergence not achieved.")


