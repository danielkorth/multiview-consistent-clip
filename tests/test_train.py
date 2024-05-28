import os
from pathlib import Path
import torch
import numpy as np

import pytest
from hydra.core.hydra_config import HydraConfig
from omegaconf import DictConfig, open_dict
import pytorch_lightning as pl

from src.train_demo import train
from tests.helpers.run_if import RunIf

from torchmetrics.functional.pairwise import pairwise_cosine_similarity
import torch.nn.functional as F

from src.models.components.losses import LossContrastive


import torch

def nearest_power_of_2(x):
    return 1 << (x - 1).bit_length()

def test_layers_size():

    # for encoder
    input_size = 512
    output_size = 128
    n_hidden_layers = 2



    linear_sizes = [
        int(input_size - i * (input_size - output_size) / (n_hidden_layers + 1))
        for i in range(n_hidden_layers + 2)
    ]

    print(linear_sizes)
    
    # Adjust sizes to the nearest power of 2
    layer_sizes = [1 << (size - 1).bit_length() for size in linear_sizes]

    print(layer_sizes)

    ## for decoder

    input_size = 128
    output_size = 512
    n_hidden_layers = 2

    linear_sizes = [
        int(input_size - i * (input_size - output_size) / (n_hidden_layers + 1))
        for i in range(n_hidden_layers + 1, -1, -1)
    ]

    print(linear_sizes)
    
    # Adjust sizes to the nearest power of 2
    layer_sizes = [1 << (size - 1).bit_length() for size in linear_sizes]

    print(layer_sizes)

def sum_all_diagonal_matrix(mat: torch.tensor): 
    n,_ = mat.shape
    zero_mat = torch.zeros((n, n)) # Zero matrix used for padding
    mat_padded =  torch.cat((zero_mat, mat, zero_mat), 1) # pads the matrix on left and right
    mat_strided = mat_padded.as_strided((n, 2*n), (3*n + 1, 1)) # Change the strides
    # print(mat_strided)
    sum_diags = torch.sum(mat_strided, 0) # Sums the resulting matrix's columns
    return sum_diags[1:]

def test_sum_all_diagonal_matrix():

    dim = 7
    X = torch.arange(dim*dim).reshape(dim, dim)
    # print(X)
    # tensor([[0, 1, 2],
    #        [3, 4, 5],
    #        [6, 7, 8]]) 
    print(f'num diags  {len(sum_all_diagonal_matrix(X))}')
    # tensor([ 6., 10., 12.,  6.,  2.])

def new_loss(text_embeddings, predicted_img_embeddings):

    batch_size, datapoint_size, embedding_size = predicted_img_embeddings.shape

    al_embeddings = torch.cat((text_embeddings.unsqueeze(1), predicted_img_embeddings), dim=1) #shape (batch size, data points size + 1, embedding size)
    permuted_embeddings = al_embeddings.permute(1, 0, 2) #shape (data points size + 1, batch size, embedding size)
    permuted_embeddings = permuted_embeddings.reshape(-1, embedding_size) #shape ((data points size + 1) * batch size, embedding size)

    sim = pairwise_cosine_similarity(permuted_embeddings)

    diag_sums = sum_all_diagonal_matrix(sim)

    dim = batch_size + batch_size * datapoint_size #dim of sim matrix
    similarity_indexes = np.arange(batch_size, dim, batch_size) #index of diagonals with sim score
    sim_mask = torch.zeros(dim)
    sim_mask[similarity_indexes] = 1
    dissim_mask = 1 - sim_mask	

    shifted_sim_mask = torch.cat((torch.zeros(dim-1), sim_mask))
    shifted_dissim_mask = torch.cat((torch.zeros(dim-1), dissim_mask))

    similarity_scores = diag_sums * shifted_sim_mask
    similarity_score_summed = torch.sum(similarity_scores)

    dissimilarity_scores = diag_sums * shifted_dissim_mask
    dissimilarity_score_summed = torch.sum(dissimilarity_scores)

    return similarity_score_summed, dissimilarity_score_summed

def test_new_loss():
     # --- input ----
    text_embeddings = torch.stack([
        torch.tensor([1,2,3,4]),
        torch.ones(4) * 2
    ])

    # Create image embedding tensor with shape (batch_size, datapoint_size, embedding_size)
    predicted_img_embeddings = torch.stack([
        torch.stack([
            torch.tensor([1,2,3,4]),
            torch.tensor([1,2,3,4]),
            torch.tensor([1,2,3,4]),
        ]),
        torch.stack([
            torch.ones(4) * 100,
            torch.ones(4) * 200,
            torch.ones(4) * 300,
        ])
    ])

    loss_sim, loss_dissim = new_loss(text_embeddings, predicted_img_embeddings)



def test_loss():

    # --- input ----
    text_embeddings = torch.stack([
        torch.tensor([1,2,3,4]),
        torch.ones(4) * 2
    ])

    # Create image embedding tensor with shape (batch_size, datapoint_size, embedding_size)
    predicted_img_embeddings = torch.stack([
        torch.stack([
            torch.tensor([1,2,3,4]),
            torch.tensor([1,2,3,4]),
            torch.tensor([1,2,3,4]),
        ]),
        torch.stack([
            torch.ones(4) * 100,
            torch.ones(4) * 200,
            torch.ones(4) * 300,
        ])
    ])

    # -------
    # -------- Compute the similarity and dissimilarity losses using loops
    batch_size, datapoint_size, embedding_size = predicted_img_embeddings.shape
    al_embeddings = torch.cat((text_embeddings.unsqueeze(1), predicted_img_embeddings), dim=1) #shape (batch size, data points size + 1, embedding size)


    permuted_embeddings = al_embeddings.permute(1, 0, 2) #shape (data points size + 1, batch size, embedding size)
    permuted_embeddings_flat = permuted_embeddings.reshape(-1, embedding_size) #shape ((data points size + 1) * batch size, embedding size)

    sim = pairwise_cosine_similarity(permuted_embeddings_flat)

    # print(f'al embeddings shape {al_embeddings.shape} \n {al_embeddings}')
    # print(f' permuted embeddings shape {permuted_embeddings.shape}\n {permuted_embeddings}')
    # print(f' permuted embeddings flat shape {permuted_embeddings_flat.shape}\n {permuted_embeddings_flat}')
    # print(f'sim mat \n {sim}')
    
    loss_sim = 0
    loss_dissim = 0

    print(f'batch size {batch_size}, datapoint size {datapoint_size}')

    sim_indexes = np.arange(batch_size, datapoint_size*batch_size + batch_size, batch_size)

    for i in range(1, datapoint_size*batch_size + batch_size):
        if i in sim_indexes:
            loss_sim += sim.diagonal(i).sum()
        else:
            loss_dissim += sim.diagonal(i).sum()

    print(f'loss sim {loss_sim}')
    print(f'loss dissim {loss_dissim}')

    ## -------- Compute the similarity and dissimilarity losses without using loops

    loss_sim, loss_dissim = new_loss(text_embeddings, predicted_img_embeddings)

    print(f'loss sim {loss_sim}')
    print(f'loss dissim {loss_dissim}')

    ## -------- Compute loss and all with loss_contrastive

    loss, sim_score, dissim_score= loss_contrastive(text_embeddings, predicted_img_embeddings)
    print(f'loss sim {sim_score}')
    print(f'loss dissim {dissim_score}')


def test_train_fast_dev_run(cfg_train: DictConfig) -> None:
    """Run for 1 train, val and test step.

    :param cfg_train: A DictConfig containing a valid training configuration.
    """
    HydraConfig().set_config(cfg_train)
    with open_dict(cfg_train):
        cfg_train.trainer.fast_dev_run = True
        cfg_train.trainer.accelerator = "cpu"
    train(cfg_train)


@RunIf(min_gpus=1)
def test_train_fast_dev_run_gpu(cfg_train: DictConfig) -> None:
    """Run for 1 train, val and test step on GPU.

    :param cfg_train: A DictConfig containing a valid training configuration.
    """
    HydraConfig().set_config(cfg_train)
    with open_dict(cfg_train):
        cfg_train.trainer.fast_dev_run = True
        cfg_train.trainer.accelerator = "gpu"
    train(cfg_train)


@RunIf(min_gpus=1)
@pytest.mark.slow
def test_train_epoch_gpu_amp(cfg_train: DictConfig) -> None:
    """Train 1 epoch on GPU with mixed-precision.

    :param cfg_train: A DictConfig containing a valid training configuration.
    """
    HydraConfig().set_config(cfg_train)
    with open_dict(cfg_train):
        cfg_train.trainer.max_epochs = 1
        cfg_train.trainer.accelerator = "gpu"
        cfg_train.trainer.precision = 16
    train(cfg_train)


@pytest.mark.slow
def test_train_epoch_double_val_loop(cfg_train: DictConfig) -> None:
    """Train 1 epoch with validation loop twice per epoch.

    :param cfg_train: A DictConfig containing a valid training configuration.
    """
    HydraConfig().set_config(cfg_train)
    with open_dict(cfg_train):
        cfg_train.trainer.max_epochs = 1
        cfg_train.trainer.val_check_interval = 0.5
    train(cfg_train)


@pytest.mark.slow
def test_train_ddp_sim(cfg_train: DictConfig) -> None:
    """Simulate DDP (Distributed Data Parallel) on 2 CPU processes.

    :param cfg_train: A DictConfig containing a valid training configuration.
    """
    HydraConfig().set_config(cfg_train)
    with open_dict(cfg_train):
        cfg_train.trainer.max_epochs = 2
        cfg_train.trainer.accelerator = "cpu"
        cfg_train.trainer.devices = 2
        cfg_train.trainer.strategy = "ddp_spawn"
    train(cfg_train)


@pytest.mark.slow
def test_train_resume(tmp_path: Path, cfg_train: DictConfig) -> None:
    """Run 1 epoch, finish, and resume for another epoch.

    :param tmp_path: The temporary logging path.
    :param cfg_train: A DictConfig containing a valid training configuration.
    """
    with open_dict(cfg_train):
        cfg_train.trainer.max_epochs = 1

    HydraConfig().set_config(cfg_train)
    metric_dict_1, _ = train(cfg_train)

    files = os.listdir(tmp_path / "checkpoints")
    assert "last.ckpt" in files
    assert "epoch_000.ckpt" in files

    with open_dict(cfg_train):
        cfg_train.ckpt_path = str(tmp_path / "checkpoints" / "last.ckpt")
        cfg_train.trainer.max_epochs = 2

    metric_dict_2, _ = train(cfg_train)

    files = os.listdir(tmp_path / "checkpoints")
    assert "epoch_001.ckpt" in files
    assert "epoch_002.ckpt" not in files

    assert metric_dict_1["train/acc"] < metric_dict_2["train/acc"]
    assert metric_dict_1["val/acc"] < metric_dict_2["val/acc"]
