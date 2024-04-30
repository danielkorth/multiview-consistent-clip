from pathlib import Path

import pytest
import torch

from src.data.mnist_datamodule import MNISTDataModule

from src.data.dataset.rendered_images import RenderedImagesDataset
from src.data.dataset.lvm_embeddings import LVMEmbeddingsDataset
from src.data.datamodule.lvm_embeddings import LVMEmbeddingsDataModule


@pytest.mark.parametrize("batch_size", [32, 128])
def test_mnist_datamodule(batch_size: int) -> None:
    """Tests `MNISTDataModule` to verify that it can be downloaded correctly, that the necessary
    attributes were created (e.g., the dataloader objects), and that dtypes and batch sizes
    correctly match.

    :param batch_size: Batch size of the data to be loaded by the dataloader.
    """
    data_dir = "data/"

    dm = MNISTDataModule(data_dir=data_dir, batch_size=batch_size)
    dm.prepare_data()

    assert not dm.data_train and not dm.data_val and not dm.data_test
    assert Path(data_dir, "MNIST").exists()
    assert Path(data_dir, "MNIST", "raw").exists()

    dm.setup()
    assert dm.data_train and dm.data_val and dm.data_test
    assert dm.train_dataloader() and dm.val_dataloader() and dm.test_dataloader()

    num_datapoints = len(dm.data_train) + len(dm.data_val) + len(dm.data_test)
    assert num_datapoints == 70_000

    batch = next(iter(dm.train_dataloader()))
    x, y = batch
    assert len(x) == batch_size
    assert len(y) == batch_size
    assert x.dtype == torch.float32
    assert y.dtype == torch.int64


def test_rendered_images_dataset():
    path_to_rendered_images = "C:\data\objaverse"
    dataset = RenderedImagesDataset(path_to_rendered_images)
    dataset.load_relative_image_paths()
    
    image, hash = dataset.__getitem__(0)
    print(image.shape)
    print(hash)

def test_lmv_embeddings_dataset():
    path_to_data = "C:\\Users\\Hannah\\sw\\multiview-robust-clip\\data\\objaverse"
    path_to_hashes = "C:\\Users\\Hannah\\sw\\multiview-robust-clip\\data\\objaverse\\training_hashes.csv"

    dataset = LVMEmbeddingsDataset(path_to_data, path_to_hashes)

    data_dict = dataset.__getitem__(3)
    print(data_dict["embedding"])
    print(data_dict["text_prompt"])

def test_lmv_embeddings_datamodule():
    path_to_data = "C:\\Users\\Hannah\\sw\\multiview-robust-clip\\data\\objaverse"

    # TODO: provide 3 different paths to hashe files
    path_to_hashes = "C:\\Users\\Hannah\\sw\\multiview-robust-clip\\data\\objaverse\\training_hashes.csv"

    datamodule = LVMEmbeddingsDataModule(path_to_data, path_to_hashes, path_to_hashes, path_to_hashes)
    datamodule.setup()