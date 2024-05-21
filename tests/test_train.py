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
    batch_size, datapoint_size, embedding_size = predicted_img_embeddings.shape

    al_embeddings = torch.cat((text_embeddings.unsqueeze(1), predicted_img_embeddings), dim=1) #shape (batch size, data points size + 1, embedding size)


    print(f'al embeddings shape {al_embeddings.shape} \n {al_embeddings}')
    permuted_embeddings = al_embeddings.permute(1, 0, 2) #shape (data points size + 1, batch size, embedding size)
    print(f' permuted embeddings shape {permuted_embeddings.shape}\n {permuted_embeddings}')
    permuted_embeddings_flat = permuted_embeddings.reshape(-1, embedding_size) #shape ((data points size + 1) * batch size, embedding size)

    print(f' permuted embeddings flat shape {permuted_embeddings_flat.shape}\n {permuted_embeddings_flat}')
    sim = pairwise_cosine_similarity(permuted_embeddings_flat)
    print(f'sim mat \n {sim}')
    
    # -------- Compute the similarity and dissimilarity losses using loops

    loss_sim = 0
    loss_dissim = 0

    print(f'batch size {batch_size}, datapoint size {datapoint_size}')

    sim_indexes = np.arange(batch_size, datapoint_size*batch_size + batch_size, batch_size)
    print(f'sim indexes {sim_indexes}')
    for i in sim_indexes:
        print(f'sim diag {sim.diagonal(i)}')

    for i in range(1, datapoint_size*batch_size + batch_size):
        if i in sim_indexes:
            loss_sim += sim.diagonal(i).sum()
        else:
            loss_dissim += sim.diagonal(i).sum()

    print(f'loss sim {loss_sim}')
    print(f'loss dissim {loss_dissim}')


    ## -------- Compute the similarity and dissimilarity losses without using loops

    dim = batch_size + batch_size * datapoint_size
    num_diagonls = 2*dim -1

    w = torch.eye(dim).unsqueeze(0)
    print(f'w shape {w.shape}, \n {w}')
    # from after conv2d result, extract inner inner dim, then take the middle column
    result = F.conv2d(sim, w, padding=num_diagonls//2)#[0][0][:, num_diagonls//2]

    print(f'result shape {result.shape}, \n {result}')


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
