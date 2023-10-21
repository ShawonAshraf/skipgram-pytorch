import argparse
import os

import matplotlib.pyplot as plt
import torch
from model import SkipgramModel
from sklearn.manifold import TSNE

parser = argparse.ArgumentParser()
parser.add_argument("--path", type=str, required=True, help="path to the saved model")
parser.add_argument("--n", type=int, required=True, help="number of words to visualize")
CONFIG = parser.parse_args()


def load_model(model_path: str) -> tuple:
    assert os.path.exists(model_path)

    checkpoint = torch.load(model_path)

    state_dict = checkpoint["model_state_dict"]
    vocabulary = checkpoint["vocabulary"]
    word_to_idx = checkpoint["word_to_idx"]
    noise_distribution = checkpoint["noise_distribution"]
    dimensions = checkpoint["dimensions"]

    model = SkipgramModel(
        vocab_size=len(vocabulary),
        vocabulary=vocabulary,
        word_to_idx=word_to_idx,
        embedding_dim=dimensions,
        noise_distribution=noise_distribution,
    )
    model.load_state_dict(state_dict)

    return model, vocabulary, word_to_idx


def visualize_n_words(n: int, model: SkipgramModel) -> None:
    embeddings_np = model.word_embedding.weight.detach().cpu().numpy()
    vocabulary = model.vocabulary

    tsne = TSNE()
    embeddings_tsne = tsne.fit_transform(embeddings_np[:n, :])

    _, _ = plt.subplots(figsize=(16, 16))
    for idx in range(n):
        plt.scatter(*embeddings_tsne[idx, :], color="steelblue")
        plt.annotate(
            vocabulary[idx],
            (embeddings_tsne[idx, 0], embeddings_tsne[idx, 1]),
            alpha=0.7,
        )

    plt.show()


if __name__ == "__main__":
    model, vocabulary, word_to_idx = load_model(CONFIG.path)
    visualize_n_words(CONFIG.n, model)
