import os
import pickle as pkl

import numpy as np

from mdance import data
from mdance.tools.bts import calculate_medoid, extended_comparison


# System parameters
input_traj_numpy = data.sim_traj_numpy              # Trajectory as a .npy array (frames x features)
N_atoms = 50                                        # Number of atoms in the system
sieve = 1                                           # Use every sieve-th frame (1 = all frames)

# HELM parameters
n_clusters = 10                                     # Number of clusters to assign
pre_cluster_labels = data.labels_60                 # Labels of the initial clusters (e.g. from NANI)
pickle_file = 'outputs/intra-helm.pkl'              # HELM output from run_helm.py
metric = 'MSD'                                      # Distance metric for clustering
n_structures = 11                                   # Frames saved per cluster, closest to its medoid
output_dir = 'outputs'                              # Folder for output files


if __name__ == '__main__':
    os.makedirs(output_dir, exist_ok=True)

    # Load data and cluster labels
    traj_numpy = np.load(input_traj_numpy)[::sieve]
    total_frames = len(traj_numpy)
    with open(pickle_file, 'rb') as f:
        cluster_dict = pkl.load(f)
    all_indices = [i[0] for i in cluster_dict[n_clusters]]; print(f'cluster indices: {all_indices}')
    pre_frames, pre_labels = np.loadtxt(pre_cluster_labels, unpack=True, dtype=int, delimiter=',')
    # Keep frames that are in the sieved trajectory and skip unassigned frames (-1)
    keep = (pre_frames % sieve == 0) & (pre_labels >= 0)
    pre_frames, pre_labels = pre_frames[keep] // sieve, pre_labels[keep]

    # Load cluster info based on pre_labels
    pre_clusters = []
    unique_labels = np.unique(pre_labels)
    for i, label in enumerate(unique_labels):
        pre_clusters.append(traj_numpy[pre_frames[pre_labels == label]])

    # Sort clusters by population
    cluster_indices = cluster_dict[n_clusters]
    unsorted_indices = []
    for i, index in enumerate(cluster_indices):
        population = index[2]   # Nik is number of frames in cluster
        unsorted_indices.append((i, population))
    sorted_indices = sorted(unsorted_indices, key=lambda x: x[1], reverse=True)

    # Save cluster data and medoids
    individual_frac = []
    frame_indices_all = {}
    best_frame_indices = []
    for i, index in enumerate(sorted_indices):
        frame_indices_all[i] = []
        cluster_dict[i] = cluster_indices[index[0]]
        pre_clus_indices = cluster_dict[i][0]
        individual_frac.append('%.6f' % (cluster_dict[i][2] / total_frames))

        # Find the corresponding frame indices in the original data
        for j, label in enumerate(pre_labels):
            for index in pre_clus_indices:
                if label == unique_labels[index]:
                    frame_indices_all[i].append(pre_frames[j])

    for key in frame_indices_all:
        # Concatenate data that corresponds to the cluster_dict[i] (pre_clus_indices)
        frames = np.stack([traj_numpy[j] for j in frame_indices_all[key]])
        if len(frames) > 1:
            # Calculate medoid of each cluster.
            medoid_index = calculate_medoid(frames, metric=metric, N_atoms=N_atoms)

            # Find the closest frame to the medoid in the pre_clusters
            medoid = frames[medoid_index]
            msd_to_medoid = []
            for j, frame in enumerate(frames):
                msd_to_medoid.append((j, extended_comparison(np.array([frame, medoid]), data_type='full',
                                                                metric=metric, N_atoms=N_atoms)))
            msd_to_medoid = np.array(msd_to_medoid)
            sorted = np.argsort(msd_to_medoid[:, 1])[:n_structures]
            best = np.array(frame_indices_all[key])[sorted]
            best_frame_indices.append(best)
        else:
            best_frame_indices.append(np.array(frame_indices_all[key]))
    params = f'HELM, n_clusters: {n_clusters}, pickle_file: {pickle_file}, metric: {metric}, sieve: {sieve}'

    # Save cluster labels (-1 for frames not in any cluster)
    labels = np.full(total_frames, -1, dtype=int)
    for cluster, frames in frame_indices_all.items():
        labels[frames] = cluster
    with open(f'{output_dir}/helm_labels_{n_clusters}.csv', 'w') as f:
        f.write(f'# {params}\n')
        f.write('# Frame Index, Cluster Index\n')
        for i, label in enumerate(labels):
            f.write(f'{i * sieve},{label}\n')

    # Save best frame indices for each cluster
    with open(f'{output_dir}/helm_best_frames_{n_clusters}.csv', 'w') as f:
        f.write(f'# {params}, n_structures: {n_structures}\n')
        f.write('# Frame Index, Cluster Index\n')
        for i, cluster in enumerate(best_frame_indices):
            for frame in cluster:
                f.write(f'{frame * sieve},{i}\n')

    # Population of top 10 clusters
    top_10_frac = sum(float(i) for i in individual_frac[:10])
    with open(f'{output_dir}/helm_summary_{n_clusters}.csv', 'w') as f:
        f.write(f'# {params}, top 10 frac: {top_10_frac}\n')
        f.write('# Cluster Index, Fraction of total frames\n')
        for i, frac in enumerate(individual_frac):
            f.write(f'{i},{frac}\n')
