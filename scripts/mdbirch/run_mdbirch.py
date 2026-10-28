import os

import numpy as np

from mdance.cluster import mdbirch
from mdance import data


# System parameters
input_traj_numpy = data.sim_traj_numpy              # Trajectory as a .npy array (frames x features)
sieve = 1                                           # Use every sieve-th frame (1 = all frames)

# mdBIRCH parameters
threshold = 3.06                                    # Subcluster radius; larger values give fewer clusters
output_dir = 'outputs'                              # Folder for output files


if __name__ == '__main__':
    os.makedirs(output_dir, exist_ok=True)

    # Load trajectory data
    traj_numpy = np.load(input_traj_numpy)[::sieve]

    # Set merge criterion
    D = traj_numpy.shape[1]
    mdbirch.set_merge('radius', features=D)

    # Initialize and fit mdBirch model
    model = mdbirch.mdBirch(threshold=threshold)
    model.fit(traj_numpy)

    clusters = model.get_cluster_mol_ids()
    # Renumber the clusters by size, largest first
    clusters = sorted(clusters, key=len, reverse=True)
    centroids = model.get_centroids()

    # Create frame-to-cluster label assignment
    n_frames = len(traj_numpy)
    frame_labels = np.full(n_frames, -1, dtype=int)

    for cluster_id, frame_indices in enumerate(clusters):
        for frame_idx in frame_indices:
            frame_labels[frame_idx] = cluster_id

    print(f"Number of clusters: {len(clusters)}")

    # Create CSV output with frame,label
    frame_numbers = np.arange(n_frames) * sieve
    csv_data = np.column_stack((frame_numbers, frame_labels))

    # Save as CSV with header
    output_csv = f'{output_dir}/mdbirch_labels_{threshold}.csv'
    params = f'mdBIRCH, threshold: {threshold}, n_clusters: {len(clusters)}, sieve: {sieve}'
    np.savetxt(output_csv, csv_data, delimiter=',', header=f'{params}\nFrame Index, Cluster Index', fmt='%d')
