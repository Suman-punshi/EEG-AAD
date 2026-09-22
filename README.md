# EEG-Based Auditory Attention Detection (AAD)

This repository contains the code, preprocessing scripts, and trained models for EEG-based Auditory Attention Detection (AAD) for the ICASSP 2026 Grand Challenge.

## Data Preprocessing
### Train and Validation Data
1. Download the train/validation EEG and label data and move it into the project directory.
2. Run the preprocessing script:

```bash
python data_process_val_train.py
```

### Test data
1. Download the test EEG data and move it into the project directory.
2. Run the test preprocessing script:

```bash
python data_process.py
```

## Notebooks
1. **eeg_aad_track1_dann.ipynb** – Trains the model with Domain-Adversarial Neural Network (DANN) module for cross-subject generalization.
2. **eeg_aad_track1_lstm.ipynb** – Trains the model without the DANN module, using only LSTM for temporal dependencies.
3. **eeg-aad-mm-aad-data-seed42-100.ipynb** – Trains the model with Domain-Adversarial Neural Network (DANN) module for cross-subject generalization and does this for two seeds.
4. **eeg-aad-track1-test.ipynb, eed-aad-model-testing** – Computes subject-wise standard deviation and runs inference on the test set using the trained DANN model.
5. **extract-activations.ipynb** - Extracts the activation of the model with gradient reversal layer from temporal, spatial and lstm layer**
6. **sae-training.ipynb** - Trains the Sparse autoencoder model with the activations from temporal, spatial and lstm layer
7. **sparsefeatureanalysis.ipynb** - Analyze the sparse features from trained sparse autoencoders.

## Model Architecture
![](AAD_arch.png)



