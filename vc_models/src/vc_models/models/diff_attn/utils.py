from functools import lru_cache
from typing import TypeVar
import spacy
import torch


__all__ = ['auto_autocast', 'compute_token_merge_indices', 'cached_nlp']

T = TypeVar('T')


def auto_autocast(*args, **kwargs):
    if not torch.cuda.is_available():
        kwargs['enabled'] = False

    return torch.cuda.amp.autocast(*args, **kwargs)

def compute_input_word_token_indices_from_sentence_tokens(tokenizer, prompts: str, words: str, word_idx, offset_idx: int = 0):
    # prompts is a batch of sentemnces, words is a batch of search words
    # tokenize the prompts and words - each word can map to multiple tokens
    # compute the token indices of the batch of words in the batch of prompts
    tokens = [tokenizer.tokenize(prompt.lower()) for prompt in prompts]
    search_tokens = [tokenizer.tokenize(word.lower()) for word in words]
    start_indices = []
    for i in range(len(tokens)):
        start_indices.append([x + offset_idx for x in range(len(tokens[i])) if tokens[i][x:x + len(search_tokens[i])] == search_tokens[i]])
    return start_indices, None      # none is there just to match the func signature of compute_token_merge_indices

def compute_token_merge_indices(tokenizer, prompt: str, word: str, word_idx: int = None, offset_idx: int = 0):
    merge_idxs = []
    # tokens = tokenizer([prompt_elt.lower() for prompt_elt in prompt]).input_ids
    tokens = [tokenizer.tokenize(prompt_elt.lower()) for prompt_elt in prompt]
    # tokens = tokenizer.tokenize(prompt.lower())
    if word_idx is None:
        word = word.lower()
        search_tokens = [tokenizer.tokenize(word_elt.lower()) for word_elt in word]
        # search_tokens = tokenizer.tokenize(word)
        start_indices = [x + offset_idx for x in range(len(tokens)) if tokens[x:x + len(search_tokens)] == search_tokens]
        for indice in start_indices:
            merge_idxs += [i + indice for i in range(0, len(search_tokens))]
        if not merge_idxs:
            raise ValueError(f'Search word {word} not found in prompt!')
    else:
        merge_idxs.append(word_idx)

    return [x + 1 for x in merge_idxs], word_idx  # Offset by 1.  ## but didn't we already indedx the maps as maps[1:-1] so is this valid??


nlp = None


@lru_cache(maxsize=100000)
def cached_nlp(prompt: str, type='en_core_web_md'):
    global nlp

    if nlp is None:
        try:
            nlp = spacy.load(type)
        except OSError:
            import os
            os.system(f'python -m spacy download {type}')
            nlp = spacy.load(type)

    return nlp(prompt)
