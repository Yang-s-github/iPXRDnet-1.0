# iPXRDnet

This repository is the official implementation of iPXRDnet.


# Requirements
```
gpuinfo==1.0.0a7
matplotlib==3.5.3
numpy==1.21.5
pandas==1.3.5
rdkit==2024.3.1
scikit_learn==1.0.2
torch==1.12.1.post200
torch_scatter==2.0.9
tqdm==4.66.1
transformers==4.24.0
```

# Dataset

We provide datasets and model checkpoint under https://zenodo.org/doi/10.5281/zenodo.12602593 .



# Script functionality


```
iPXRDnet_hMOF-130T: Python script for training adsorption predictions in the hMOF-130T database
iPXRDnet_hmof-300T: Python script for training adsorption predictions in the hmof-300T database
iPXRDnet_Sa: Python script for training separation selectivity predictions
iPXRDnet_SD: Python script for training self-diffusion coefficient predictions
iPXRDnet_MOD: Python script for training bulk modulus and shear modulus predictions
iPXRDnet_exAPMOF: Python script for training experimental adsorption isotherms of anion column MOFs using PXRD and material ligands
iPXRDnet_exAPMOF-ALM: Python script for training experimental adsorption isotherms of anion column MOFs using material ligands only
iPXRDnet_exAPMOF-PXRD: Python script for training experimental adsorption isotherms of anion column MOFs using PXRD only
iPXRDnet_exAPMOF-ISO: Python script for training experimental adsorption isotherms of anion column MOFs
CMPNN_pretrain: Python script for pre-training CMPNN
```
