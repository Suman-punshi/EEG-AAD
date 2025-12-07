import os
import numpy as np
from tqdm import tqdm
import h5py

def getData_test(data_path, sub_ids):
    """
    Load preprocessed .npy EEG files for a list of subject IDs.

    Parameters
    ----------
    data_path : str
        Directory containing files named like 'S{subject_id}.npy'.
    sub_ids : list of int
        Subject IDs to load (e.g., [31, 32, ...]).

    Returns
    -------
    alldata : list of np.ndarray
        List where each element is the loaded data for one subject,
        shaped (N, 32, 128) after transposition:
            - N: number of trials / segments for that subject
            - 32: EEG channels (electrodes)
            - 128: time points per segment
    """
    alldata = []
    # iterate through subject ids with a progress bar
    for id in tqdm(sub_ids):
        # Build the filename 'S{id}.npy' and load it
        onedata = np.load(os.path.join(data_path, f'S{id}.npy'))
        onedata = onedata.transpose(0, 2, 1)   # (N, 32, 128)
        # Append this subject's data to the list
        alldata.append(onedata)
    return alldata


def data_preprocess_test(sub_ids, seq_alldata, chunk_size=512,
                         save_path='./h5_files_task1', data_split="test"):
    """
    Save per-subject EEG data to individual HDF5 files in chunks.

    For each subject in seq_alldata (a list aligned with sub_ids), this:
      - Creates an HDF5 file named "S{subject_id}_{data_split}_data.h5".
      - Writes two extensible datasets: 'data' and 'sub_id'.
          * 'data' stores EEG segments with shape (num_samples, 32, 128)
          * 'sub_id' stores the subject ID for each segment shape (num_samples, 1)
      - Data is written in chunks (controlled by chunk_size) to avoid high memory usage.

    Parameters
    ----------
    sub_ids : list of int
        Subject IDs corresponding to entries in seq_alldata.
    seq_alldata : list of np.ndarray
        List of per-subject arrays. Each array must have shape (N, 32, 128).
    chunk_size : int, optional
        How many samples to write at once into the HDF5 file. Default 512.
    save_path : str, optional
        Directory where the h5 files will be written. Default './h5_files_task1'.
    data_split : str, optional
        Label to include in filenames (e.g., "test", "val", "train").

    Returns
    -------
    save_path : str
        Path to the directory where files were saved (useful for chaining).
    """
    # Ensure output directory exists
    os.makedirs(save_path, exist_ok=True)

    # We want a visible progress bar for subjects
    for subject_idx, data in tqdm(enumerate(seq_alldata)):
        # Map list index to corresponding subject id
        subject_id = sub_ids[subject_idx]
        num_samples = data.shape[0]  # number of segments for this subject

        # Compose HDF5 filename
        h5_path = os.path.join(save_path, f"S{subject_id}_{data_split}_data.h5")

        # Open an HDF5 file for writing. Using context manager ensures proper close().
        with h5py.File(h5_path, 'w') as f:
            # dset and sset will hold references to datasets once created.
            dset = None
            sset = None
            total_written = 0

            # Iterate through the subject's samples in chunks to reduce memory usage.
            # start ranges: 0, chunk_size, 2*chunk_size, ...
            for start in range(0, num_samples, chunk_size):
                end = min(start + chunk_size, num_samples)

                # Slice the numpy array to get the chunk.
                chunk_data = data[start:end]      # shape: (chunk, 32, 128)
                # np.squeeze ensures there are no accidental singleton dims;
                # should be a no-op if chunk_data already has shape (k, 32, 128).
                chunk_data = np.squeeze(chunk_data)

                # Create subject ID column for these rows: shape (chunk, 1)
                chunk_sub_ids = np.full((chunk_data.shape[0], 1),
                                        subject_id, dtype=np.int32)

                # Create datasets on first iteration only
                if dset is None:
                    # Determine the per-sample shape (channels, timesteps) e.g., (32,128)
                    data_shape_suffix = chunk_data.shape[1:]  # (32, 128)

                    # Create an extensible dataset for 'data':
                    #   - initial shape (0, 32, 128): no rows yet
                    #   - maxshape (None, 32, 128): first axis extensible
                    #   - chunks=(chunk_size, 32, 128): optimal storage chunking
                    #   - dtype float32: saves disk space and is standard for NN input
                    #   - compression gzip with moderate compression level (4)
                    f.create_dataset(
                        'data',
                        shape=(0,) + data_shape_suffix,
                        maxshape=(None,) + data_shape_suffix,
                        chunks=(chunk_size,) + data_shape_suffix,
                        dtype=np.float32,
                        compression='gzip', compression_opts=4
                    )

                    # Create an extensible integer dataset for subject IDs: (0,1) initially
                    f.create_dataset(
                        'sub_id',
                        shape=(0, 1),
                        maxshape=(None, 1),
                        chunks=(chunk_size, 1),
                        dtype=np.int32,
                        compression='gzip', compression_opts=4
                    )

                    # Keep handles to the created datasets for later writes
                    dset = f['data']
                    sset = f['sub_id']

                # Resize datasets to append the new chunk
                old_size = dset.shape[0]
                new_size = old_size + chunk_data.shape[0]

                # Resize along the first axis; other axes remain fixed
                dset.resize((new_size,) + dset.shape[1:])
                sset.resize((new_size, 1))

                # Write chunk into the newly allocated slice
                dset[old_size:new_size] = chunk_data
                sset[old_size:new_size] = chunk_sub_ids

                total_written += chunk_data.shape[0]

                # Informational print for debugging / monitoring I/O progress per subject
                print(f"Subject S{subject_id}: wrote {start}:{end} (total={total_written})")

        # File closed at the end of 'with' block; confirm save path
        print(f"Saved: {h5_path}")

    return save_path


# A list of subject IDs to load and convert
sub_ids = [31, 32, 33, 34, 35, 36, 37, 38, 39, 40]

# Load numpy files into a list of arrays (one array per subject)
alldata = getData_test(
    "D:/EEG_AAD_Project/Testset_audio_visual/preprocessed/data",
    sub_ids
)

# Convert and save each subject's data to an HDF5 file in './h5_files'
data_preprocess_test(sub_ids, alldata, save_path='./h5_files')
