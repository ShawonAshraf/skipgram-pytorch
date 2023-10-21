import argparse
import os
import shutil
from typing import Any

import numpy as np
import torch
import torch.optim as optim
from accelerate import Accelerator
from dataset import SkipgramDataset
from loguru import logger
from model import NegativeSamplingLoss, SkipgramModel
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

torch.manual_seed(2023)
np.random.seed(2023)


argparser = argparse.ArgumentParser()

# ============ dataset args ==========
argparser.add_argument(
    "--n_instances",
    type=int,
    required=True,
    help="number of instances to load from the training dataset",
)
argparser.add_argument(
    "--window_size", type=int, required=True, help="window size for skipgrams"
)
argparser.add_argument(
    "--min_freq",
    type=int,
    required=True,
    help="minimum frequency for a word to be allowed",
)


# ================= dataloader args ==================
argparser.add_argument(
    "--batch_size", type=int, required=True, help="batch size for the dataloader"
)
argparser.add_argument(
    "--num_workers",
    type=int,
    required=True,
    help="number of cpu workers to use for data loading",
)


# =================== trainer =========================
argparser.add_argument("--lr", type=float, required=True, help="initial learning rate")
argparser.add_argument(
    "--epochs", type=int, required=True, help="number of epochs to train for"
)

CONFIG = argparser.parse_args()


# ============= training function ================
# for a single batch
def train_step(
    model: SkipgramModel, batch: Any, criterion: NegativeSamplingLoss, device: Any
) -> torch.Tensor:
    model.train()
    
    target = batch["target"]
    context = batch["context"]

    target_vector, context_vector = model(target, context)
    noise_vector = model.generate_noise(target.size()[0], device)

    loss = criterion(target_vector, context_vector, noise_vector)

    return loss


def train_model(
    model: SkipgramModel,
    epochs: int,
    lr: float,
    train_dataloader: DataLoader,
    accelerator: Accelerator,
) -> tuple:
    optimiser = optim.AdamW(params=model.parameters(), lr=lr)
    criterion = NegativeSamplingLoss()

    model, optimiser, train_dataloader = accelerator.prepare(
        model, optimiser, train_dataloader
    )

    logger.info("Optimiser: Adam, Criterion: NegativeSamplingLoss")
    logger.info(f"Training for {epochs} epochs")

    global_losses = []
    for e in range(epochs):
        epoch_losses = []

        model.train()
        for batch in tqdm(train_dataloader):
            optimiser.zero_grad()

            loss = train_step(model, batch, criterion, accelerator.device)

            epoch_losses.append(loss.item())

            accelerator.backward(loss)
            optimiser.step()

        mean_epoch_loss = torch.tensor(epoch_losses).cpu().mean(dim=-1).item()
        logger.info(
            f"Epoch: [{e + 1}]/[{epochs}] ::: Mean Loss/Train : {mean_epoch_loss}"
        )
        global_losses.append(mean_epoch_loss)

    return model, global_losses


def save_model(model: SkipgramModel, overwrite: bool) -> None:
    if not os.path.exists("../saved_models"):
        logger.info("Creating directory to save models")
        os.makedirs("../saved_models", exist_ok=False)
    else:
        if overwrite:
            logger.warning(
                "Overwrite is True, removing directory and creating a new one"
            )
            shutil.rmtree("../saved_models")
            os.makedirs("../saved_models", exist_ok=False)
        else:
            logger.error("Directory already exists, so, exiting")
            return

    state_dict = model.state_dict()
    torch.save(state_dict, "../saved_models/saved.ckpt")
    logger.success(f"Saved model at {os.getcwd()}/saved_models/saved.ckpt")


if __name__ == "__main__":
    # create dataset
    logger.info("Creating training dataset .... ")
    train_set = SkipgramDataset(
        n_instances=CONFIG.n_instances,
        window_size=CONFIG.window_size,
        min_word_freq=CONFIG.min_freq,
    )

    # dataloader
    logger.info(f"Dataloader Batch Size :: {CONFIG.batch_size}")
    logger.info(f"Using {CONFIG.num_workers} CPU workers")
    train_loader = DataLoader(
        train_set,
        shuffle=True,
        batch_size=CONFIG.batch_size,
        pin_memory=True,
        num_workers=CONFIG.num_workers,
    )

    # model init
    logger.info("Model init")
    model = SkipgramModel(
        vocab_size=train_set.vocab_size,
        vocabulary=train_set.vocabulary,
        word_to_idx=train_set.word_to_idx,
        embedding_dim=300,
        noise_distribution=train_set.get_noise_distribution(),
    )

    # accelerator
    accelerator = Accelerator()

    trained_model, _ = train_model(
        model, CONFIG.epochs, CONFIG.lr, train_loader, accelerator
    )

    save_model(trained_model, True)
