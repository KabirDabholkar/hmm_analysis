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

def specify_groundtruth_state(num_hid_states,num_obs_states,eps=0.4,emission_eps=0.4,seed=0,start_prob_dist=None):
    GT = CategoricalHMM(n_components=num_hid_states, init_params="")
    GT.n_features = num_obs_states
    rs = np.random.RandomState(seed)
    GT.startprob_ = np.ones(num_hid_states)/num_hid_states
    GT.transmat_ = np.eye(num_hid_states,k=1)
    GT.transmat_[-1,0] = 1.0
    GT.transmat_ += rs.uniform(0,eps,size=GT.transmat_.shape)
    GT.transmat_ /= GT.transmat_.sum(1,keepdims=True)
    GT.emissionprob_ = np.eye(num_hid_states,num_obs_states)
    GT.emissionprob_ +=  rs.uniform(0,emission_eps,size=GT.emissionprob_.shape)
    GT.emissionprob_ /= GT.emissionprob_.sum(1,keepdims=True)
    if start_prob_dist is None:
        GT.startprob_ = GT.get_stationary_distribution()
    else:
        GT.startprob_ = start_prob_dist #(shape=GT.startprob_.shape)
        GT.startprob_ /= GT.startprob_.sum()
    return GT


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

if __name__ == '__main__':
    GT = specify_groundtruth_state(10,10,
                              start_prob_dist=lambda shape: (np.arange(shape[0])>5).astype(float) )


