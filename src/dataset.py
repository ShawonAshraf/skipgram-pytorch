import numpy as np
import torch
from datasets import load_dataset
from datasets.arrow_dataset import Dataset as D
from loguru import logger
from torch.utils.data import Dataset
from tqdm.auto import tqdm
from typing import List, Dict, Set


class SkipgramDataset(Dataset):
    def __init__(self,
                 n_instances: int,
                 window_size: int,
                 min_word_freq: int,
                 base_dataset_name: str = "multi_news") -> None:
        
        # https://huggingface.co/datasets/multi_news
        self.base_dataset_name = base_dataset_name
        # how many instances from the dataset should be used
        # sure you can use the whole dataset but
        # that'll mean a lot of data and longer training time
        # if n_instances = -1, the whole dataset will be used
        self.n_instances = n_instances
        self.window_size = window_size
        self.min_word_freq = min_word_freq
        
        
        # load dataset
        logger.info(f"Loading dataset :: {self.base_dataset_name}")
        
        if self.n_instances == -1:
            logger.warning("Using the whole dataset!")
            self.base_dataset = load_dataset(
                self.base_dataset_name)["train"][:]["document"]
        else:
            self.base_dataset = load_dataset(
                self.base_dataset_name)["train"][:self.n_instances]["document"]
        
        logger.info("Preprocessing ....")
        self.word_to_idx, self.word_freqs, self.vocabulary = self.__preprocess()
        
        self.vocab_size = len(self.vocabulary)
        logger.info(f"Vocabulary size :: {len(self.vocabulary)}")
        
        
        logger.info("Subsampling ....")
        self.subsampled_vocab = self.__subsample()
        self.sub_vocab_size = len(self.subsampled_vocab)
        logger.info(f"Subsampled Vocabulary Size :: {self.sub_vocab_size}")
        
        # create a reverse lookup table
        # int to word
        logger.info("Generating idx_to_word ....")
        self.idx_to_word = {self.word_to_idx.get(word): word for word in self.vocabulary}
        
        logger.info("Storing unigram probabilities ....")
        self.total_words = sum(self.word_freqs.values())
        
        
        # populate
        logger.info("Creating target context pairs ....")
        self.pairs = self.__create_target_context_pairs()
        logger.success(f"Done! Created {len(self.pairs)} instances.")
        
        
    # utils
    def __clean_and_tokenise(self, text: str) -> List[str]:
        # lowercase
        doc = text.lower()
        # replace space + newline with empty str
        doc = doc.replace(" \n", "")
        # split
        doc = doc.split()
        return doc
    
    def __preprocess(self) -> tuple:
        # word to idx mapping
        word_to_idx = dict()

        # frequencies
        word_freqs = dict()


        for _, ds in tqdm(enumerate(self.base_dataset), total=len(self.base_dataset), desc="update_freq"):
            doc = self.__clean_and_tokenise(ds)

            # update freq
            for word in doc:
                if not word in word_freqs.keys():
                    word_freqs[word] = 1
                else:
                    word_freqs[word] += 1


        # filter words which have a frequency of less than min_freq
        # as the final words list / vocabulary
        vocabulary = [w for w in word_freqs.keys() if word_freqs[w] >= self.min_word_freq]
            
        # sort vocabulary in the descending order of frequency
        vocabulary = sorted(vocabulary, key=word_freqs.get, reverse=True) # type: ignore
        
        # populate word_to_idx
        for idx, word in tqdm(enumerate(vocabulary), total=len(vocabulary), 
                          desc="word_to_idx"):
            word_to_idx[word] = idx
        

        return word_to_idx, word_freqs, vocabulary
    
    # select a subsampled list to be used as vocabulary from the main vocab list
    def __subsample(self) -> List[str]:
        T = 1e-5 # follwing the original threshold from the paper
        
        # probability for a single word using its frequency
        def subsample_probability(frequency):
            return 1 - np.sqrt(T / frequency)
        
        # freq(word) / vocab_size : what is the representation of the word in the total corpus
        # more frequent words such as the, to don't really contribute much to the meaning of another vector
        # and hence can be dropped for the sake of learning good word vectors
        # https://mccormickml.com/2017/01/11/word2vec-tutorial-part-2-negative-sampling/
        scaled_frequencies = { word: self.word_freqs[word] / self.vocab_size for word in self.vocabulary } 
        
        
        # get the probability of a word to drop
        drop_probability = { word: subsample_probability(scaled_frequencies[word]) for word in self.vocabulary}
        
        # if probability of being dropped is drop_probability then for remaining, 1 - drop_probability
        subsampled_vocab = [word for word in self.vocabulary if np.random.rand() < (1 - drop_probability[word])]
        
        # update original vocabulary with the subsampled one
        return subsampled_vocab
    
    
    # the target word index is the index of the word in the list
    def __get_context_words(self, tokens: List[str], target_word_index: int) -> List[str]:
        # index range to look for
        # we look among window_size words left and right of the targer word in a sentence
        index_range = np.random.randint(1, self.window_size)
        
        # we look through all the tokens in a sentence and pick context words
        # excluding the target word
        
        start = max(0, target_word_index - index_range)
        finish = min(target_word_index + index_range, len(tokens) - 1)
        
        context_words = tokens[start: target_word_index] + tokens[target_word_index + 1 : finish + 1]
        # discard words which failed to meet min freq (in preprocessing)
        # and are not part of the vocab
        context_words = [cw for cw in context_words if self.word_to_idx.get(cw) is not None]
        
        return context_words
    
    def __create_target_context_pairs(self) -> Dict[str, Set]:
        pairs = dict()
        
        for _, sentence in tqdm(enumerate(self.base_dataset), 
                                  desc="create_pairs",
                                  total=len(self.base_dataset)):
            
            tokens = self.__clean_and_tokenise(sentence)
            for index, token in enumerate(tokens):
                context_words = self.__get_context_words(tokens, index)
                
                if token not in pairs.keys():
                    pairs[token] = set(context_words)
                else:
                    pairs[token].update(context_words)
            
        
        return pairs
    
    
    # dataset methods
    def __len__(self) -> int:
        return self.sub_vocab_size
    
    
    
    def __getitem__(self, index: int) -> Dict:
        target_word = self.vocabulary[index]
        all_context_words = list(self.pairs[target_word])
        
        # randomly pick one
        random_idx = np.random.randint(0, len(all_context_words))
        context_word = all_context_words[random_idx]
        
        
        target = self.word_to_idx[target_word]
        context = self.word_to_idx[context_word]
        
        
        target = torch.tensor(target, dtype=torch.long)
        context = torch.tensor(context, dtype=torch.long)
        
        return {
            "target": target,
            "context": context,
        }

    # for negative sampling
    def get_noise_distribution(self) -> torch.Tensor:        
        frequencies = torch.tensor([
            self.word_freqs[word] for word in self.vocabulary])

        # sort in the descending order
        frequencies, _ = torch.sort(frequencies, dim=-1, descending=True, stable=True)
        
        # unigram_probabilities
        unigram_probabilities = frequencies / frequencies.sum(dim=-1)
        
        unigram_distribution = unigram_probabilities / unigram_probabilities.sum(dim=-1)
        noise_distribution = torch.pow(unigram_distribution, 0.75) /  torch.pow(unigram_distribution, 0.75).sum(dim=-1)
        
        return noise_distribution
    
    
if __name__ == "__main__":
    ds = SkipgramDataset(100, 5, 5)
