import sys
import logging
import time
import functools

import numpy as np
import senteval
import bayes_opt 

from src import reset_seed
from src.model import NonParametricPairwiseAttention


# Set PATHs
PATH_TO_DATA = './data'
PATH_TO_FREQ = PATH_TO_DATA + r'/enwiki_vocab_min200.txt'
PATH_TO_VEC = PATH_TO_DATA + r'/glove.840B.300d.txt'


def print(msg, level="INFO"):
    message = "[%s] %s: %s\n" % (time.asctime(), level, msg)
    sys.stdout.write(message)
    sys.stdout.flush()


# SentEval prepare and batcher
def prepare(params, src):
    if params["model"].is_adapted is False:
        unique, selected = set(), []
        for row in src:
            tmp = "".join(row).lower()
            if tmp not in unique:
                unique.add(tmp)
                selected.append(row)
        params["model"].adapt(selected)

    params["sent_embed"] = {}
    for text, embed in zip(src, params["model"].fit_transform(src)):
        params["sent_embed"][" ".join(text)] = embed


def batcher(params, batch):
    return np.vstack([params["sent_embed"][" ".join(sent)] for sent in batch])


def run(framework, model, task, noisy, alpha):
    model.noisy = noisy
    model.alpha = alpha

    out = {"task": task, "noisy": noisy, "alpha": alpha,}
    rslt = framework.eval([task])[task]    
    rslt = rslt.get("all", rslt)
    for key, val in rslt.items():
        if not isinstance(val, (tuple, list)):
            out[key] = val
    print(str(out).replace("\n", ""))
    return out["acc"]


# Set params for SentEval
encoder = NonParametricPairwiseAttention(PATH_TO_VEC, PATH_TO_FREQ)
params =  {"task_path": PATH_TO_DATA,
           "usepytorch": True,
           "kfold": 10,
           "model": encoder,
           "classifier": {"optim": "adam",
                          "batch_size": 64,
                          "nhid": 50,
                          "tenacity": 5,
                          "epoch_size": 4}}

# Set up logger
logging.basicConfig(format='%(asctime)s : %(message)s', level=logging.DEBUG)

if __name__ == "__main__":
    tasks = sys.argv[1].split("-")
    tasks = ['SICKEntailment', # 1 min
             'MRPC',           # 2 min
             'TREC',           # 3 min
             'SST2',           # 4 min
             'CR',             # 14 min
              'MR',             # 36 min
              'MPQA',           # 32 min
              'SUBJ',           # 37 min
              ][int(tasks[0]):int(tasks[1])]
    print("Running Gridsearch for tasks: %s" % tasks)
    for task in tasks:
        print("Evaluating: %s" % task)
        encoder._prepared = False
        se = senteval.engine.SE(params, batcher, prepare)
        run_one_dataset = functools.partial(run, framework=se, model=encoder, task=task)
        optimizer = bayes_opt.BayesianOptimization(f=run_one_dataset,
                                                   pbounds={"noisy": (1, 24),
                                                            "alpha": (0.01, 0.15),
                                                            },
                                                   random_state=reset_seed.seed)
        optimizer.maximize(n_iter=40)
        print("%s Results: %s" % (task, optimizer.max))
