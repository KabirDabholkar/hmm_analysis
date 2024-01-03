from omegaconf import DictConfig, OmegaConf
import hydra
import os
import pandas as pd
import pickle as pkl
import similarity
import numpy as np
# from hydra.utils import instantiate
from config_utils import instantiate
from prepare_model import CoHMM,split_model_emission,fuse_model
from copy import deepcopy
from hmmlearn.hmm import GaussianHMM
from utils import flatten_with_lengths,HMM_Dataset, normalise, setattrs, setattrs_kwargs
from metrics import bernoulli_bits_per_spike
from hmm_adapter import CoHMM3d as CoHMM

CONFIG_PATH = "configs"
# CONFIG_NAME = "config"
CONFIG_NAME = "config_cohmm"

@hydra.main(version_base='1.3', config_path=CONFIG_PATH, config_name=CONFIG_NAME)
def main(cfg):
    instantiate(cfg.numpy_seed)
    cfg.teacher.n_components = 2
    # cfg.teacher.object._target_ = 'hmm_adapter.GaussianHMM3d'

    teacher2 = instantiate(cfg.teacher)
    teacher = GaussianHMM(n_components=cfg.teacher.n_components)
    teacher.n_features = teacher2.n_features
    teacher.covariance_type = 'spherical'
    teacher.means_ = np.random.normal(size=(teacher.n_components,teacher.n_features)) * 5e-2
    teacher._covars_ = np.ones((teacher.n_components,teacher.n_features))
    teacher.startprob_ = teacher2.startprob_
    teacher.transmat_ = teacher2.transmat_

    obs,hid = teacher.sample(2)

    proba = teacher.predict_proba(obs)

    print(proba)

    # print(teacher.transmat_)
    # print(obs.shape)
    # print(teacher.lambdas_.shape)
    # B = teacher.lambdas_
    # print(B.shape,obs[0:1].T.shape)
    # Y0 = np.diag(B @ obs[0].astype(float))
    # Y1 = np.diag(B @ obs[1].astype(float))




if __name__ == '__main__':
    resolvers = {
        'eval'      : eval,
        'ind'       : lambda a, i: a[i],
        'listmul'   : lambda l, i: [l] * i,
        'getattr'   : getattr,
        'setattrs'  : setattrs,
        'as_tuple'  :  lambda *args: tuple(args),
    }
    for resolver_name,resolver_val in resolvers.items():
        if not OmegaConf.has_resolver(resolver_name):
            OmegaConf.register_new_resolver(resolver_name,resolver_val)
    main()

