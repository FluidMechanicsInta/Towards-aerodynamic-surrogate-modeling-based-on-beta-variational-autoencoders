import torch
import time
from torch.utils.tensorboard import SummaryWriter
import torch.optim.lr_scheduler as lr_scheduler
import os

import src.data as lib_data
import src.model as lib_model
import src.vaes_training as lib_training

torch.manual_seed(seed=42)

device = "cuda" if torch.cuda.is_available() else "cpu"

# Training parameters
batch_size = 64
lr = 2e-4
lr_end = 1e-5
epochs = 1
beta_sweep = [0.1]
latent_dim_sweep = [2]

# Working directories
general_outputs_folder = "./outputs/training_VAE"
os.makedirs(name=general_outputs_folder, exist_ok=True)

logs_folder = f"./{general_outputs_folder}/01_logs_VAE"
os.makedirs(name=logs_folder, exist_ok=True)

checkpoints_folder = f"./{general_outputs_folder}/02_checkpoints_VAE"
os.makedirs(name=checkpoints_folder, exist_ok=True)

# Data
(MA_train, MA_test, Cp_train, Cp_test,
 MA_mean, MA_std, Cp_mean, Cp_std) = lib_data.load_filtered_dataset(path="./00_data/filtered_dataset.pkl")

Cp_train_scaled = (Cp_train - Cp_mean) / Cp_std
Cp_test_scaled = (Cp_test - Cp_mean) / Cp_std

n_train = Cp_train_scaled.shape[0]
n_test = Cp_test_scaled.shape[0]
print(f"N. train: {n_train:d}, N. test: {n_test:d}\n")

dataset_train = torch.utils.data.DataLoader(dataset=torch.from_numpy(Cp_train_scaled).to(device),
                                            batch_size=batch_size, shuffle=True, num_workers=0)

dataset_test = torch.utils.data.DataLoader(dataset=torch.from_numpy(Cp_test_scaled).to(device),
                                           batch_size=batch_size, shuffle=False, num_workers=0)

for beta in beta_sweep:

    for latent_dim in latent_dim_sweep:

        betaSch = lib_training.betaScheduler(start_value=beta, end_value=beta, warmup=20)

        # Get model
        model = lib_model.VAE(data_size=Cp_train.shape[1], latent_dim=latent_dim).to(device)
        encoder_params = list(model.encoder.parameters())
        decoder_params = list(model.decoder.parameters())

        # Get optimizer
        optimizer = torch.optim.Adam(
            params=[{"params": encoder_params, "weight_decay": 0},
                    {"params": decoder_params, "weight_decay": 0}],
            lr=lr, weight_decay=0)

        scheduler = lr_scheduler.OneCycleLR(optimizer, max_lr=lr, total_steps=epochs, div_factor=2,
                                            final_div_factor=lr / lr_end, pct_start=0.2)

        # Training loop
        str_date = time.strftime("%Y%m%d_%H_%M")

        model_name = (f"{str_date}_net0_beta{beta}_dim{latent_dim}_lr{lr}OneCycleLR{lr_end}_bs{batch_size}_"
                      f"epochs{epochs}")
        print(f"Model: {model_name}")

        logger = SummaryWriter(log_dir=f"{logs_folder}/{model_name}")

        for epoch in range(1, epochs + 1):

            model.train()
            beta = betaSch.get_beta(epoch, prints=False)
            loss, MSE, KLD, elapsed = lib_training.train_epoch(model=model, data=dataset_train, optimizer=optimizer,
                                                               beta=beta, device=device)

            model.eval()
            loss_test, MSE_test, KLD_test, elapsed_test = lib_training.test_epoch(model=model, data=dataset_test,
                                                                                  beta=beta, device=device)

            scheduler.step()

            lib_training.print_progress(epoch=epoch, epochs=epochs, loss=loss, loss_test=loss_test, mse=MSE, kld=KLD,
                                        elapsed=elapsed, elapsed_test=elapsed_test)

            logger.add_scalar(tag="General loss/Total", scalar_value=loss, global_step=epoch)
            logger.add_scalar(tag="General loss/MSE", scalar_value=MSE, global_step=epoch)
            logger.add_scalar(tag="General loss/KLD", scalar_value=KLD, global_step=epoch)
            logger.add_scalar(tag="General loss/Total_test", scalar_value=loss_test, global_step=epoch)
            logger.add_scalar(tag="General loss/MSE_test", scalar_value=MSE_test, global_step=epoch)
            logger.add_scalar(tag="General loss/KLD_test", scalar_value=KLD_test, global_step=epoch)
            logger.add_scalar(tag="Optimizer/LR", scalar_value=scheduler.get_last_lr()[0], global_step=epoch)

        state_checkpoint = {"state_dict": model.state_dict(), "optimizer_dict": optimizer.state_dict()}
        checkpoint_file = f"{checkpoints_folder}/{model_name}_epoch_final.pth.tar"
        lib_model.save_checkpoint(state=state_checkpoint, path_name=checkpoint_file)

        print(f"End model: {model_name}\n")

print("\nEND")
