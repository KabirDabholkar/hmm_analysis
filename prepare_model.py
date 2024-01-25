from hmmlearn.hmm import GaussianHMM, CategoricalHMM, PoissonHMM
import numpy as np
from collections.abc import Iterable
from sklearn.utils import check_random_state


def fit_model_emission(model,input_obs,target_obs,lengths=None):
    cummu_lengths = np.concatenate([np.zeros((1),dtype=int),np.cumsum(np.array(lengths,dtype=int))])
    if lengths:
        latent_prob = np.concatenate([model.predict_proba(
            input_obs[cummu_lengths[i]:cummu_lengths[i+1]]
        ) for i in range(len(lengths))])
    else:
        latent_prob = model.predict_proba(input_obs)
    print(latent_prob.shape)
    n_feat = model.n_features
    n_comp =  model.n_components
    new_emission_mat = (target_obs==np.arange(n_feat,dtype=int)[None,:])[:,None,:].astype(float) * latent_prob[:,:,None]
    new_emission_mat = new_emission_mat.sum(0)
    new_emission_mat = new_emission_mat
    new_emission_mat /= new_emission_mat.sum(1,keepdims=True)
    model.emissionprob2_ = new_emission_mat
    return model


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


def set_epsilon_chain(hmm):
    hmm.transmat_ = ''


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

def sample_hmm(hmm_model,length=40,trials=400,seed_base=0,use_emmision2=False,flatten=True, return_states=False):
    if use_emmision2:
        temp_emissionprob_ = hmm_model.emissionprob_
        hmm_model.emissionprob_ = hmm_model.emissionprob2_
    X = [hmm_model.sample(length,random_state=check_random_state(i+seed_base)) for i in range(trials)]
    obs,state = zip(*X)
    if use_emmision2:
        hmm_model.emissionprob_ =  temp_emissionprob_
    #lengths = [x.shape[0] for x in X]
    obs = np.stack(obs)
    state = np.stack(state)[...,None]
    if flatten:
        obs = obs.flatten()[..., None]
        state = state.flatten()[...,None]
    if return_states:
        return obs,state
    return obs

def cosample_hmm(hmm_model,length=40,trials=400,seed_base=0):
    state = [hmm_model.sample(length, random_state=check_random_state(i + seed_base))[1] for i in range(trials)]
    state = np.stack(state)
    state = state.flatten()
    print(state.shape,'state cosample shape',length,trials)
    # print(hmm_model._generate_sample_from_state(state)[0].shape
    # print(hmm_model.emissionprob_[state, :].shape)
    cdf = np.cumsum(hmm_model.coemissionprob_[:,state, :],axis=-1)
    rng = np.random.RandomState(seed=seed_base)
    return (cdf > rng.uniform(size=(*cdf.shape[:2],1))).argmax(axis=-1)[...,None]


def split_model_emission(model: PoissonHMM, split_indices: Iterable[int]):
    from copy import deepcopy
    models = [deepcopy(model) for _ in range(len(split_indices)+1)]
    all_lambdas = np.split(model.lambdas_,split_indices,axis=-1)
    for model,lam in zip(models,all_lambdas):
        model.lambdas_ = lam
        model.n_features = model.lambdas_.shape[-1]
    return models

def fuse_model(CoHMM_model):
    from copy import deepcopy
    base_model = deepcopy(CoHMM_model.encoder)
    base_model.lambdas_ = np.concatenate([
        CoHMM_model.encoder.lambdas_,
        CoHMM_model.decoder.lambdas_
    ],axis=1)
    base_model.n_features = CoHMM_model.encoder.n_features+ CoHMM_model.decoder.n_features
    return base_model

def refit_decoder(base_model,train_in,train_out):
    model = deepcopy(base_model)



class CoHMM():
    def __init__(self,encoder,decoder):
        assert encoder.n_components==decoder.n_components
        self.encoder = encoder
        self.decoder = decoder

    def predict(self, X, lengths=None, return_state_proba=False):
        state_proba = self.encoder.predict_proba(X,lengths=lengths)
        rate_pred = self.decoder._generate_rate_from_stateproba(state_proba)
        if return_state_proba:
            return rate_pred,state_proba
        return rate_pred
    def co_fit(self,X_in,X_out,lengths=None):
        # (
        #     stats,
        #     curr_logprob,
        #     sub_X,
        #     lattice,
        #     posteriors,
        #     fwdlattice,
        #     bwdlattice
        # ) = self.encoder._do_estep(X_in, lengths=lengths,return_everything=True)
        #
        # self.decoder._init(X_out, lengths)

        (
            dec_stats,
            dec_curr_logprob,
            dec_sub_X,
            dec_lattice,
            dec_posteriors,
            dec_fwdlattice,
            dec_bwdlattice
        ) = self.decoder._do_estep_modencoder(X_out, X_in=X_in, encoder=self.encoder, lengths=lengths,return_everything=True)

        self.decoder._do_mstep(dec_stats)

        # proba = self.encoder.predict_proba(X_in,lengths = lengths)
        # posteriors = proba
        # X = X_out
        # # print(X[:5].astype(float).shape)
        #
        # obs_zero = posteriors.T @ (~X).astype(float)
        # obs_one = posteriors.T @ (X).astype(float)
        # self.decoder.lambdas_ = obs_one/(obs_zero+obs_one)





class PoissonReadout():
    def __init__(self,n_components):
        self.n_components = n_components

    def predict(self,state_proba):
        return state_proba @ self.lambdas_

    # def fit(self,state_proba,observ):
    #     self.lambdas_ =

def test_cohmm():
    model = CoHMM(
        encoder=PoissonHMM(n_components=2),
        decoder=PoissonHMM(n_components=2,params='l'),
    )
    model.encoder.fit(np.random.poisson(3,(10,20)))
    print(model.encoder.lambdas_.shape)


    model.co_fit(
        np.random.poisson(3,(10,20)),
        np.random.poisson(3,(10,20))
    )

    pred = model.predict(np.random.poisson(3, (10, 20)))
    # print(pred)



if __name__ == '__main__':
    # GT = specify_groundtruth_state(10,10,
    #                           start_prob_dist=lambda shape: (np.arange(shape[0])>5).astype(float) )
    test_cohmm()
    # import matplotlib.pyplot as plt
    # from utils import indicator_func_to_matrix
    # plt.imshow(
    #     # np.expand_dims(np.arange(4),1)==np.expand_dims(np.arange(10)%4,0)
    #     indicator_func_to_matrix(
    #         size=(4,10),
    #         indicator_func=lambda i,j: i == j % 4
    #     )
    # )
    # plt.show()


