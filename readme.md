## Non-Parametric Pairwise Attention Random Walk Model (NoPPA) 

#### 0. Setup

```shell
cd ./data/downstream/
./get_transfer_data.bash
wget https://nlp.stanford.edu/data/glove.840B.300d.zip
unzip glove.840B.300d.zip && rm -rf glove.840B.300d.zip
cd ../../
```

#### 1. Install

```shell
pip install numpy
pip install torch
pip install scikit-learn
pip install bayesian-optimization
```

#### 2. Reproduce Experiments

```shell
python -u bayessian_search.py 0-8 > log\results.log
```

#### 3. Use On Your Data

```python
# you need to prepare word frequency and pre-trained word embedding
PATH_TO_DATA = './data'
PATH_TO_FREQ = PATH_TO_DATA + r'/enwiki_vocab_min200.txt'
PATH_TO_VEC = PATH_TO_DATA + r'/glove.840B.300d.txt'


# text you need to encode
data = ["this is a sentence.", 
        "this is another sentence."]

from src.model import NonParametricPairwiseAttention
encoder = NonParametricPairwiseAttention(PATH_TO_VEC, PATH_TO_FREQ)
encoder.adapt(data)
embeds = encoder.fit_transform(data)
```

#### 3. Running Time

Although the generating sentence embedding from NoPPA doesn't require GPU and is pretty fast, SentEval framework still needs GPU to speed up the evaluation of sentence embedding on downstream tasks. By using a single GTX3090 GPU, running through all 8 datasets needs almost 2.5 hours for one round. The Bayesian search will totally run 40 times, which means almost 4 days in total. 
