# skipgram-pytorch

A Pytorch implementation of the skipgram model from [Mikolov et al. 2013](https://proceedings.neurips.cc/paper_files/paper/2013/file/9aa42b31882ec039965f3c4923ce901b-Paper.pdf) on the train split of the [multi_news](https://huggingface.co/datasets/multi_news) dataset.

## Pre-requisites

1. Make sure to have Anaconda / Miniconda installed
2. WSL or a Linux distro installed on your computer, since the conda env for this project was created in a Linux environment (Ubuntu 22.04 to be exact.)
3. If you "don't" plan to use the entirety of the `multi_news` dataset and would rather use a small subset,
you can train the entire model on a CPU. Otherwise, any GPU with 8GB of VRAM will do fine. (Or, you can use Google Colab.)

## Setup

```bash
git clone https://github.com/ShawonAshraf/skipgram-pytorch.git
# or using gh-cli 
gh repo clone ShawonAshraf/skipgram-pytorch

cd skipgram-pytorch
conda env create -f env.yml
# wait for the environment to setup

# create a directory to save your model checkpoint after training
mkdir saved_models
# alternatively you can use any other directory, more below:
```

## Code Overview

```bash
├── env.yml
├── notebooks
│   └── prototype.ipynb
├── README.md
├── saved_models
└── src
    ├── dataset.py
    ├── model.py
    ├── train.py
    └── visualise.py
```

- The `dataset.py` file contains the necessary preprocessing functions and a `torch.utils.dataset` class, which can be used with a `torch.utils.dataloader` during training
- `model.py` contains the definitions for the model and  negative sampling loss
- `train.py` contains the functions needed to train the model and save trained checkpoints to the disk
- `visualise.py` uses `TSNE` to visualise the word embeddings in a point cloud space

## Usage

### Training

```bash
# run python src/train.py --help to get a description of the args
python src/train.py --help                                                                                                                                  
usage: train.py [-h] --n_instances N_INSTANCES --window_size WINDOW_SIZE --min_freq MIN_FREQ --batch_size BATCH_SIZE --num_workers NUM_WORKERS --lr LR --epochs EPOCHS --path PATH

options:
  -h, --help            show this help message and exit
  --n_instances N_INSTANCES
                        number of instances to load from the training dataset
  --window_size WINDOW_SIZE
                        window size for skipgrams
  --min_freq MIN_FREQ   minimum frequency for a word to be allowed
  --batch_size BATCH_SIZE
                        batch size for the dataloader
  --num_workers NUM_WORKERS
                        number of cpu workers to use for data loading
  --lr LR               initial learning rate
  --epochs EPOCHS       number of epochs to train for
  --path PATH           path where the model will be saved post training


# example 
python src/train.py --n_instances 15000 --window_size 5 --min_freq 5 --batch_size 128 --num_workers 4  --lr 0.003 --epochs 15 --path saved_models/vectors.tar

```

### Visualisation

```bash

# run python src/visualise.py --help to get a description of the args
python src/visualise.py --help                                   
usage: visualise.py [-h] --path PATH --n N

options:
  -h, --help   show this help message and exit
  --path PATH  path to the saved model
  --n N        number of words to visualize


# example
python src/visualise.py --n 300 --path "saved_models/vectors.tar"

```

### Using Colab

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/ShawonAshraf/skipgram-pytorch/blob/main/notebooks/full_notebook.ipynb)

You can upload the `notebooks/full_notebook.ipynb` file to your Google Drive and then open it using Colab. Also, you can use this direct link just above.

## Additional Resources

- [Mikolov, Tomas, Ilya Sutskever, Kai Chen, Greg S Corrado, and Jeff Dean. “Distributed Representations of Words and Phrases and Their Compositionality,” n.d.
](https://proceedings.neurips.cc/paper_files/paper/2013/file/9aa42b31882ec039965f3c4923ce901b-Paper.pdf)

- [Mikolov, Tomas, Kai Chen, Greg Corrado, and Jeffrey Dean. “Efficient Estimation of Word Representations in Vector Space.” arXiv, September 6, 2013. http://arxiv.org/abs/1301.3781.
](https://arxiv.org/abs/1301.3781)

- [Chris McCormick's two part tutorial on Skipgram](https://mccormickml.com/2016/04/19/word2vec-tutorial-the-skip-gram-model/)

- [SLP 3rd Edition Draft Chapter 6](https://web.stanford.edu/~jurafsky/slp3/6.pdf)
