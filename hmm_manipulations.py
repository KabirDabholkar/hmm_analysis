from copy import deepcopy
from prepare_model import CoHMM
import numpy as np
from utils import normalise

def self_kronecker(model,alternate_start_prob_=None):
    new_model = deepcopy(model)
    n_components = model.n_components
    factor = n_components
    n_components *= factor
    new_model.n_components = n_components 
    new_model.lambdas_ = np.concatenate([model.lambdas_] * factor, axis=0)
    for i in range(factor):
        new_model.lambdas_[i::factor, :] = model.lambdas_

    alternate_start_prob_ = model.startprob_ if alternate_start_prob_ is None else alternate_start_prob_
    new_model.startprob_ = normalise(
        np.kron(
            model.startprob_,
            alternate_start_prob_
        )
    )
    new_model.transmat_ = normalise(
        np.kron(
            model.transmat_,
            model.transmat_
        ),
        axis=1
    )
    return new_model

def self_kronecker_with_alternate_startprob(model):
    new_model = self_kronecker(model)
    alternate_start_prob_ = (np.arange(model.n_components) == 0).astype(model.startprob_.dtype)
    new_model.startprob_ = normalise(
        np.kron(
            model.startprob_,
            alternate_start_prob_
            # np.repeat(
            #     (np.arange(model.n_components) == 0).astype(float),
            #     model.n_components
            # )
        )
    )
    return new_model
