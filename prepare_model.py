from hmmlearn.hmm import GaussianHMM, CategoricalHMM
import numpy as np
from sklearn.utils import check_random_state


def prepare_model(model,X_train,lengths,n_features=2):
    model.n_features = n_features
    n_iter = model.n_iter
    mon_n_iter = model.monitor_.n_iter
    model.n_iter = 0
    model.fit(X_train, lengths=lengths)
    model.n_iter = n_iter
    model.monitor_.n_iter = mon_n_iter
    return model



def specify_groundtruth():
    GT = CategoricalHMM(n_components=2, init_params="")
    GT.n_features = 2
    GT.startprob_ = np.array([1 / 2., 1 / 2.])
    GT.transmat_ = np.array([[0.7, 0.3],
                            [0.1, 0.9]])

    GT.emissionprob_ = np.array([[0.3,0.7],
                                 [0.7,0.3]])

    GT.startprob_ = GT.get_stationary_distribution()
    return GT

def sample_hmm(hmm_model,length=40,trials=400,seed_base=0):
    X = [hmm_model.sample(length,random_state=check_random_state(i+seed_base))[0] for i in range(trials)]
    #lengths = [x.shape[0] for x in X]
    X = np.stack(X)
    X = X.flatten()[..., None]
    return X
