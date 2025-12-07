import os
import numpy as np
from tqdm import tqdm
import h5py


def getData_val_train(data_path, label_path, sub_ids):
    """
    Load EEG data and corresponding labels for a list of subject IDs.

    Parameters
    ----------
    data_path : str
        Path to directory containing EEG numpy files, e.g., 'S1.npy'.
    label_path : str
        Path to directory containing label numpy files, e.g., 'S1.npy'.
    sub_ids : list of int
        List of subject IDs to load (e.g., [1,2,3,...]).

    Returns
    -------
    alldata : list of np.ndarray
        EEG data per subject, each array shaped (N, 32, 128) after transpose.
    alllabel : list of np.ndarray
        Corresponding labels per subject.
    """
    alldata = []
    alllabel = [] 

    for id in tqdm(sub_ids):
        # f"{data_path}/S{id}.npy" → Example: "EEG/pre/data/S5.npy"
        onedata = np.load(f"{data_path}/S{id}.npy")

        # f"{label_path}/S{id}.npy" → Example: "EEG/pre/label/S5.npy"
        onelabel = np.load(f"{label_path}/S{id}.npy")

        # Transpose EEG to (N, 32, 128)
        onedata = onedata.transpose(0, 2, 1)

        alldata.append(onedata)
        alllabel.append(onelabel)
    
    return alldata, alllabel



def data_preprocess_val_train(sub_ids, split_ids, seq_alldata, alllabel,
                              chunk_size=512, save_path='./h5_files_task1',
                              data_split="train"):
    """
    Preprocess EEG data and labels for train or validation splits, 
    concatenate across subjects, and save into a single HDF5 file.
    Also includes a 'sub_id' field per sample.

    Parameters
    ----------
    sub_ids : list of int
        All subject IDs.
    split_ids : list of int
        Subject IDs for this split (train or val).
    seq_alldata : list of np.ndarray
        EEG data per subject.
    alllabel : list of np.ndarray
        Labels per subject.
    chunk_size : int, optional
        Number of samples to write per chunk, default=512.
    save_path : str, optional
        Directory to save HDF5 files, default='./h5_files_task1'.
    data_split : str, optional
        Split name for filename, e.g., 'train' or 'val'.

    Returns
    -------
    h5_path : str
        Path to the saved HDF5 file.
    """
    os.makedirs(save_path, exist_ok=True)

    # Filter data, labels, and subject IDs for this split
    data = [seq for idx, seq in enumerate(seq_alldata) if sub_ids[idx] in split_ids]
    label = [lbl for idx, lbl in enumerate(alllabel) if sub_ids[idx] in split_ids]
    subids = [sid for sid in sub_ids if sid in split_ids]

    # Save path 
    h5_path = os.path.join(save_path, f"{data_split}_data.h5")

    with h5py.File(h5_path, 'w') as f:
        dset = None  # EEG data
        lset = None  # Labels
        sset = None  # Subject IDs
        total_written = 0

        for idx, (data_block, label_block) in tqdm(enumerate(zip(data, label))):
            sid = subids[idx]
            num_samples = data_block.shape[0]

            # Write in chunks
            for start in range(0, num_samples, chunk_size):
                end = min(start + chunk_size, num_samples)
                chunk_data = data_block[start:end]
                chunk_label = label_block[start:end]

                # Flatten label, squeeze EEG, reshape labels
                chunk_label = chunk_label.flatten().reshape(-1, 1)
                chunk_data = np.squeeze(chunk_data)

                # Subject ID array for this chunk
                chunk_subid = np.full((chunk_data.shape[0], 1), sid, dtype=np.int32)

                # Initialize datasets on first chunk
                if dset is None:
                    data_shape_suffix = chunk_data.shape[1:]  # e.g., (32,128)
                    f.create_dataset(
                        'data',
                        shape=(0,) + data_shape_suffix,
                        maxshape=(None,) + data_shape_suffix,
                        chunks=(chunk_size,) + data_shape_suffix,
                        dtype=np.float32,
                        compression='gzip', compression_opts=4
                    )
                    f.create_dataset(
                        'label',
                        shape=(0, 1),
                        maxshape=(None, 1),
                        chunks=(chunk_size, 1),
                        dtype=np.int32,
                        compression='gzip', compression_opts=4
                    )
                    f.create_dataset(
                        'sub_id',
                        shape=(0, 1),
                        maxshape=(None, 1),
                        chunks=(chunk_size, 1),
                        dtype=np.int32,
                        compression='gzip', compression_opts=4
                    )
                    dset = f['data']
                    lset = f['label']
                    sset = f['sub_id']

                # Resize datasets and append new chunk
                old_size = dset.shape[0]
                new_size = old_size + chunk_data.shape[0]
                dset.resize((new_size,) + dset.shape[1:])
                lset.resize((new_size, 1))
                sset.resize((new_size, 1))

                dset[old_size:new_size] = chunk_data
                lset[old_size:new_size] = chunk_label
                sset[old_size:new_size] = chunk_subid  # Write subject ID

                total_written += chunk_data.shape[0]
                print(f"  Wrote chunk {start}:{end} (shape {chunk_data.shape}) total={total_written}")

    return h5_path



sub_ids = list(range(1, 31))
val_ids = [1,2,3,6]
train_ids = [sub_id for sub_id in sub_ids if sub_id not in val_ids]

alldata, all_label = getData_val_train(
    "EEG-AAD_audio_visual/preprocessed/data",
    "EEG-AAD_audio_visual/preprocessed/label",
    sub_ids
)

# Save train HDF5
data_preprocess_val_train(sub_ids, train_ids, alldata, all_label, save_path='./h5_files', data_split="train")

# Save val HDF5
data_preprocess_val_train(sub_ids, val_ids, alldata, all_label, save_path='./h5_files', data_split="val")
