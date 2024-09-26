# Towards aerodynamic surrogate modeling based on &beta;-variational autoencoders

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