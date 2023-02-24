import collections
import itertools

import numpy as np
import torch as tc
    

def calulate_position_embed(lenth, dim):
    if lenth == 0:
        return np.array([])
    base = np.vstack([pos / np.power(10000, 2.0 * np.arange(dim) / dim)\
                      for pos in range(lenth)])
    base[:, 0::2] = np.sin(base[:, 0::2])
    base[:, 1::2] = np.cos(base[:, 1::2])
    return base


def softmax(x):
    power = np.exp(x)
    return power / power.sum(axis=1, keepdims=True)


def logkernel(x, y):
    return np.log2(1.0 + (x - y) ** 2)


def rbfkernel(x, y, lamda):
    return np.exp(- lamda * (x - y) ** 2)


class NonParametricPairwiseAttention:
    def __init__(self, word2vec_path, word2freq_path, alpha=None, noisy=0, lamda=1.0, device="cuda"):
        self.glove_path = word2vec_path
        self.freq_path = word2freq_path
        self._noisy, self._alpha = None, None
        self.noisy = noisy
        self.alpha = alpha
        self.lamda = lamda
        self._prepared = False
        self._estimated = False
        self.device = device

    @property
    def alpha(self):
        return self._alpha

    @alpha.setter
    def alpha(self, new_alpha):
        assert isinstance(new_alpha, float) or new_alpha is None
        assert new_alpha is None or 0.0 <= new_alpha <= 1.0 
        self._alpha = new_alpha

    @property
    def noisy(self):
        return self._noisy

    @noisy.setter
    def noisy(self, new_noisy):
        if isinstance(new_noisy, float):
            new_noisy = int(new_noisy)
        assert isinstance(new_noisy, int)
        assert 0 <= new_noisy
        if new_noisy != self._noisy:
            self._estimated = False
            self._noisy = new_noisy

    @property
    def lamda(self):
        return self._lamda

    @lamda.setter
    def lamda(self, new_lamda):
        self._lamda = new_lamda

    @property
    def is_adapted(self):
        return self._prepared

    def fit(self, sentences):
        assert isinstance(sentences, (list, tuple))
        assert self.is_adapted, "please call `adapt()` before calling `fit()`"
        embedding = np.array(tuple(map(self._forward, sentences)))
        self._estimate(embedding)

    def transform(self, sentences): 
        assert isinstance(sentences, (list, tuple))
        assert self.is_adapted, "please call `adapt()` at first"
        embedding = np.array(tuple(map(self._forward, sentences)))
        return self._remove_nosiy(embedding)

    def fit_transform(self, sentences):
        assert isinstance(sentences, (list, tuple))
        assert self.is_adapted, "please call `adapt()` before calling `fit()`"
        embedding = np.array(tuple(map(self._forward, sentences)))
        self._estimate(embedding)
        return self._remove_nosiy(embedding)

    def adapt(self, corpus, min_freq=1):
        counter = collections.Counter(itertools.chain(*corpus))
        self._prepared = False
        self.low2idx, self.word2idx, self.idx2word, self.idx2vec, self.idx2freq = {}, {}, [], [], []
        with open(self.glove_path, 'r', encoding="utf-8") as f:
            for line in f:
                word, vec = line.split(" ", 1)
                if counter.get(word, 0.0) >= min_freq:
                    self._update_vocab(word, np.fromstring(vec, sep=" "), 0.0)

        total = 0.0
        with open(self.freq_path, 'r', encoding="utf8") as f:
            for line in f:
                word, freq = line.strip().split()
                lower, freq = word.lower(), float(freq)
                total += float(freq)
                if lower in self.low2idx:
                    self.idx2freq[self.low2idx[lower]] += freq
                    
        self.idx2freq = [_ / total for _ in self.idx2freq]
        dim = len(self.idx2vec[0])
        self.idx2pos = calulate_position_embed(max(map(len, corpus)), dim)
        self.scaler = np.sqrt(dim)
        self._prepared = True
        return self

    def _update_vocab(self, word, embed, freq):
        idx, lower = len(self.word2idx), word.lower()
        self.idx2word.append(idx)
        self.idx2vec.append(embed)
        self.idx2freq.append(freq)
        self.word2idx[word] = idx
        self.low2idx[lower] = self.word2idx.get(lower, idx)

    def _token2vector(self, sentence):
        return np.vstack([self.idx2vec[self.word2idx.get(token, 0)] for token in sentence])

    def _token2frequency(self, sent):
        if self.alpha == 0:
            return np.ones((len(sent),)) 
        freq = np.array([self.idx2freq[self.low2idx[_.lower()]] for _ in sent])
        return self.alpha / (self.alpha / 2.0 + freq) 

    def _positional_embedding(self, embed):
        return embed + self.idx2pos[:embed.shape[0]]

    def _forward(self, tokens):
        tokens = [_ for _ in tokens if _ in self.word2idx]
        if len(tokens) == 0:
            tokens = ["."]
        emb = self._token2vector(tokens)                # O(n)
        pos = self._positional_embedding(emb)           # O(n)
        weight = self._token2frequency(tokens)          # O(n)
        att = softmax(pos.dot(pos.T) / self.scaler)
        tmp = [w.dot(logkernel(pos, v)) for w, v in zip(att, pos)]       # O(n^2d)
        return weight.dot(np.hstack([tmp, emb])) / len(tokens)             # O(n)

    def _estimate(self, embed):
        self._estimated = False
        if self.noisy > 0:
            self._average, self._stderr = embed.mean(axis=0), embed.std(axis=0)
            normalized = (embed - self._average) / self._stderr
            with tc.no_grad():
                embed = tc.Tensor(normalized).to(self.device)
                self.sv = tc.svd(embed, some=True)[-1].to("cpu").numpy()[-self.noisy:]
        self._estimated = True
    
    def _remove_nosiy(self, embed):
        if self.noisy > 0:
            assert self._estimated, "please call `fit()` or `fit_transform` at first"
            embed = (embed - self._average) / self._stderr
            embed -= np.dot(np.dot(embed, self.sv.T), self.sv) # O(k^2d + d)
            embed = embed * self._stderr + self._average
        return embed

    
        
