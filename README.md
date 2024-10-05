# Towards aerodynamic surrogate modeling based on &beta;-variational autoencoders
![scheme](https://github.com/user-attachments/assets/fa4d8571-3f99-4cc3-93be-37e6aaf86aca)

## Introduction

This repository provides a streamlined pipeline for implementing &beta;-variational autoencoders (&beta;-VAEs) on a given dataset. The &beta;-VAE model identifies a low-dimensional latent space that represents the input features (pressure coefficients) based on the dataset's labels (Mach number and angle of attack). A Gaussian Process Regression (GPR) model is then applied to link the labels to the latent space, enabling feature prediction for unseen labels. The use of Principal Component Analysis (PCA) prior to &beta;-VAE training is also explored, offering significant reductions in time and computational resources.

A further explanation of the implementation and the results can be found at ["Towards aerodynamic surrogate modeling based on &beta;-variational autoencoders", Víctor Francés-Belda, Alberto Solera-Rico, Javier Nieto-Centenero, Esther Andrés, Carlos Sanmiguel Vila, and Rodrigo Castellanos](https://www.arxiv.org/abs/2408.04969).



## Structure

Project structure: 

* ```src```
  * ```data.py```: file to process the data. It must be adapted to each database.
  * ```model.py```: models to be used (&beta;-VAE with and without PCA pre-processing and MLP nets).
  * ```vaes_training.py```, ```vaes_fine_tuning.py```, ```mlp_training.py```: training routines for &beta;-VAE models, 
fine tuned models, and MLP models, respectively.
  * ```auxiliar.py```: complementary functions to be used during training.


* ```train_VAE.py```, ```train_VAE_prePCA.py```: scripts to launch the training process for the &beta;-VAE models.


* ```fine_tuning_VAE.py```, ```fine_tuning_VAE_prePCA.py```: files to conduct the fine tuning process. The ```.py``` 
files above should be executed first.


* ```train_MPL.py```, ```train_MPL_prePCA.py```: training scripts for the MLP models.

The file ```requirements.txt``` contains all the necessary dependencies of the project. It can be installed using 
```pip install -r ./requirements.txt``` from the project's path.

## Input data

The file ```./src/data.py``` expects a pickle file as input. It should contain a dictionary with the keys and values: 

* ```Cp_train_filtered```: numpy array with shape $(j, q)$, where $j$ is the number of snapshots or cases for training, and $q$ is the number of features of each case.

* ```Cp_test_filtered```: numpy array with shape $(k, q)$, where $k$ is the number of snapshots or cases for testing, and $q$ is the number of features of each case.

* ```Cp_mean```: mean $C_p$ value as a numpy float. Only training cases are considered.

* ```Cp_std```: standard deviation of the $C_p$  as a numpy float. Only training cases are considered.

* ```MA_train```: numpy array with shape $(j, 2)$, where $j$ is the number of snapshots for training, and $2$ refers to Mach (first column) and angle of attack (second column) couples. It should be ordered as ```Cp_train_filtered``` is.

* ```MA_test```: numpy array with shape $(k, 2)$, where $j$ is the number of snapshots for testing, and $2$ refers to Mach (first column) and angle of attack (second column) couples. It should be ordered as ```Cp_test_filtered``` is. 

* ```MA_mean```: mean value of value ```MA_train``` as a numpy float.

* ```Cp_std```: standard deviation of ```MA_train``` as a numpy float.
