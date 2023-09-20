import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from omegaconf import DictConfig, OmegaConf
import hydra
from sklearn.utils import check_random_state
from hmmlearn.hmm import GaussianHMM, CategoricalHMM
from hmmlearn.vhmm import VariationalCategoricalHMM
import pickle as pkl
import os
from prepare_model import specify_groundtruth_state
import matplotlib as mpl
mpl.rcParams['text.usetex'] = True
#mpl.rcParams['text.latex.preamble'] = [r'\usepackage{amsmath}']

CONFIG_PATH = "configs"
CONFIG_NAME = "config"

OmegaConf.register_new_resolver("eval", eval)
OmegaConf.register_new_resolver("ind", lambda a,i: a[i])
OmegaConf.register_new_resolver("listmul", lambda l, i: [l]*i)
OmegaConf.register_new_resolver("getattr", getattr)

def prepare_model(model,X_train,lengths,n_features=2):
    model.n_features = n_features
    n_iter = model.n_iter
    mon_n_iter = model.monitor_.n_iter
    model.n_iter = 0
    model.fit(X_train, lengths=lengths)
    model.n_iter = n_iter
    model.monitor_.n_iter = mon_n_iter
    return model

def train(model_save_path,train_X_lengths,val_X_lengths,model_n_components=5,n_iter=5,tol=1e-2,shift_sampling_window=None):
    X_train,lengths = train_X_lengths
    X_val,lengths_val = val_X_lengths

    model = CategoricalHMM(n_components=model_n_components, init_params="",n_iter=0,tol=tol,verbose=True,shift_sampling_window=shift_sampling_window)
    model.n_features = 2
    model.fit(X_train, lengths=lengths)
    model.n_iter = n_iter
    model.monitor_.n_iter = n_iter
    model.fit(X_train,lengths=lengths,X_val=X_val,lengths_val=lengths_val)

    if not os.path.exists(os.path.dirname(model_save_path)):
        os.makedirs(os.path.dirname(model_save_path))
    with open(model_save_path,'wb') as f:
        pkl.dump(model,f)

    # GT = specify_groundtruth()
    #
    # GT_score = GT.score(X_val, lengths=lengths_val)
    #
    # fig,ax = plt.subplots()
    # ax.plot(np.array(model.monitor_.history)[1:],label='validation score')
    # ax.axhline(GT_score,ls='dashed',label='ground truth score',c='red')
    # fig.tight_layout()
    # fig.savefig('plots/validation_score.png')

    # model_to_plot = GT
    # window_size = 20
    # X_shifts = [X[i:i+window_size] for i in range(0,X.shape[0]-window_size)]
    # hid_shifts = [hid[i:i + window_size] for i in range(0, X.shape[0] - window_size)]
    # hid_predicted = [model_to_plot.predict_proba(x) for x in X_shifts]
    #
    # fig,axs = plt.subplots(2+model_to_plot.n_components,1,sharex=True,figsize=(7,1.2*model_to_plot.n_components))
    #
    # for i,(x,hid,hid_pred) in enumerate(zip(X_shifts,hid_shifts,hid_predicted)):
    #     time_steps = np.arange(i,i+window_size)
    #     ax = axs[0]
    #     ax.plot(time_steps,x,lw=1,c='C0')
    #     ax = axs[1]
    #     ax.plot(time_steps,hid,c='C1',lw=1)
    #
    #     for j in range(model_to_plot.n_components):
    #         ax = axs[j+2]
    #         ax.plot(time_steps,hid_pred[:,j],c=f'C{j+2}',lw=1,alpha=0.7)
    #
    #
    #
    #         ax.set_ylim(0,1)
    #         ax.set_xlim(0,50)
    #
    # #ax.plot(hid_dec[:,0])
    # fig.tight_layout()
    # fig.savefig('plots/groundtruth.png')

    #plt.show()

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

def generate_groundtruth(GT,length=40,train_trials=400,val_trials=400):
    rs = check_random_state(546)

    X,hid = GT.sample(50,random_state=rs)

    X_train = [GT.sample(length,random_state=check_random_state(i))[0] for i in range(train_trials)]
    lengths = [x.shape[0] for x in X_train]
    X_train = np.stack(X_train) #if len(X_train)>0 else X_train[...,None]

    X_val = [GT.sample(length,random_state=check_random_state(i+10000))[0] for i in range(val_trials)]
    lengths_val = [x.shape[0] for x in X_val]
    X_val = np.stack(X_val) #if len(X_val)>0 else X_val[...,None]


    np.savetxt(f'X_train_length{length}_trials{train_trials}',X_train[...,0],fmt='%d')
    np.savetxt(f'X_val_length{length}_trials{val_trials}', X_val[...,0], fmt='%d')




def load_groundtruth(fname):
    X = np.loadtxt(fname).astype(int)
    if len(X.shape) < 2:
        X = X[:,None]
    lengths = [X.shape[1]] * X.shape[0]
    X = X.flatten()[...,None]
    return X,lengths

def load_model(model_path):
    with open(model_path,'rb') as f:
        model = pkl.load(f)
    return model

def plot_validation_scores(models,GT,val_X_lengths,reps = 4,n_components_range = np.arange(2,11,2)):

    X_val,lengths_val = val_X_lengths
    N = sum(lengths_val)
    GT_score = GT.score(X_val,lengths=lengths_val)
    GT_score /= N

    #fig,axs = plt.subplots(1,2,figsize=(12,5),sharey=True)
    fig, axs = plt.subplots(1, 1, figsize=(6, 5), sharey=True)
    # ax = axs[0]
    # ls = []
    # for i,n_comp in enumerate(n_components_range):
    #     for id in range(reps):
    #         model = models[n_comp][id]
    #         if np.array(model.monitor_.history).shape[0]<3:
    #             print(np.array(model.monitor_.history)[-1])
    #             l=ax.axhline(np.array(model.monitor_.history)[-1],color=f'C{i}',lw=1)
    #         l,=ax.plot(np.array(model.monitor_.history)[1:],color=f'C{i}',lw=1)
    #     ls.append(l)
    # l=ax.axhline(GT_score, ls='dashed', label='ground truth score', c='black')
    # ls = [l]+ls
    # labels = ['ground truth model'] + [r'$|X|$=%d'%n for n in n_components_range]
    # ax.legend(ls,labels)
    # ax.set_xlabel(r'Training steps')
    # ax.set_ylabel(r'Validation loglikelihood')

    ax = axs#[1]
    ls = []
    for i,n_comp in enumerate(n_components_range):
        for id in range(reps):
            model = models[n_comp][id]
            score = model.score(X_val, lengths=lengths_val)
            score /= N
            l=ax.scatter(n_comp,score, color=f'C{i}',s=10)
        ls.append(l)
    l=ax.axhline(GT_score, ls='dashed', label='ground truth score', c='black')
    ax.set_xlabel(r'Model $|\mathcal X|$')
    ax.set_ylabel(r'Test set loglikelihood')
    ls = [l] + ls
    labels = ['ground truth model'] + [r'$|\mathcal X|=%d$'%n for n in n_components_range]
    ax.legend(ls, labels,framealpha=0.4)


    fig.tight_layout()

    fig.savefig('plots/validation_score.png',dpi=200)
    fig.savefig('plots/validation_score.pdf')

def plot_self_consistency(models,GT,val_X_lengths,reps = 4,n_components_range = np.arange(2,11,2),eps=1e-3):

    X_val,lengths_val = val_X_lengths
    N = sum(lengths_val)
    D_JS,D_JS_shuffled = self_consistency(GT,X_val,length,Nsamples=1000)
    GT_SC = np.mean(D_JS)/np.mean(D_JS_shuffled)


    fig, axs = plt.subplots(1, 1, figsize=(6, 5), sharey=True)

    ax = axs#[1]
    ls = []
    for i,n_comp in enumerate(n_components_range):
        for id in range(reps):
            model = models[n_comp][id]
            D_JS, D_JS_shuffled = self_consistency(model, X_val, length, Nsamples=1000)
            #if np.mean(D_JS_shuffled)<1e-3:

            model_SC = np.mean(D_JS) / (np.mean(D_JS_shuffled)+eps)
            l=ax.scatter(n_comp,model_SC, color=f'C{i}',s=10)
        ls.append(l)
    l=ax.axhline(GT_SC, ls='dashed', label='ground truth', c='black')
    ax.set_xlabel(r'Model $|\mathcal X|$')
    ax.set_ylabel(r"Self consistency")
    ls = [l] + ls
    labels = ['ground truth model'] + [r'$|\mathcal X|=%d$'%n for n in n_components_range]
    ax.legend(ls, labels,framealpha=0.4)


    fig.tight_layout()
    fig.savefig('plots/validation_self_consistency.png',dpi=200)
    fig.savefig('plots/validation_self_consistency.pdf')


def plot_self_consistency_against_validation(models,GT,val_X_lengths,reps = 4,n_components_range = np.arange(2,11,2),eps=1e-3):
    X_val, lengths_val = val_X_lengths
    N = sum(lengths_val)
    D_JS, D_JS_shuffled = self_consistency(GT, X_val, length, Nsamples=1000)
    GT_SC = np.mean(D_JS) / np.mean(D_JS_shuffled)

    GT_score = GT.score(X_val,lengths=lengths_val)
    GT_score /= N

    fig, axs = plt.subplots(1, 1, figsize=(6, 5), sharey=True)

    ax = axs  # [1]
    ls = []
    for i, n_comp in enumerate(n_components_range):
        for id in range(reps):
            model = models[n_comp][id]
            D_JS, D_JS_shuffled = self_consistency(model, X_val, length, Nsamples=1000)

            score = model.score(X_val, lengths=lengths_val)
            score /= N

            model_SC = np.mean(D_JS) / (np.mean(D_JS_shuffled) + eps)
            l = ax.scatter(score, model_SC, color=f'C{i}', s=10)
        ls.append(l)
    ax.axvline(GT_score, ls='dashed', label='ground truth score', c='black')
    l = ax.axhline(GT_SC, ls='dashed', label='ground truth', c='black')
    ax.set_xlabel(r'Test set loglikelihood')
    ax.set_ylabel(r"Self consistency")
    ls = [l] + ls
    labels = ['ground truth model'] + [r'$|\mathcal X|=%d$' % n for n in n_components_range]
    ax.legend(ls, labels, framealpha=0.4)

    fig.tight_layout()
    fig.savefig('plots/validation_self_consistency_vs_loglikelihood2.png', dpi=200)
    fig.savefig('plots/validation_self_consistency_vs_loglikelihood2.pdf')

def self_consistency_and_validation_single(model,X_lengths,window_length = 10,eps=1e-3,Nsamples=500):
    X_val, lengths_val = X_lengths
    N = sum(lengths_val)

    D_JS, D_JS_shuffled = self_consistency(model, X_val, window_length, Nsamples=Nsamples)

    score = model.score(X_val, lengths=lengths_val)
    score /= N

    model_SC = np.mean(D_JS) / (np.mean(D_JS_shuffled) + eps)
    return score,model_SC


def gather_self_consistency_against_validation(models,GT,val_X_lengths,reps = 4,n_components_range = np.arange(2,11,2),eps=1e-3,model_name = 'Vanilla'):
    X_val, lengths_val = val_X_lengths
    N = sum(lengths_val)
    D_JS, D_JS_shuffled = self_consistency(GT, X_val, length, Nsamples=50000)
    GT_SC = np.mean(D_JS) / np.mean(D_JS_shuffled)

    GT_score = GT.score(X_val,lengths=lengths_val)
    GT_score /= N

    data = [{'model_name':'Ground truth',
             'model_id':0,
             'n_components':GT.n_components,
             'iterations':GT.monitor_.iter,
             'self_consistency':GT_SC,
             'score':GT_score}]

    ls = []
    for i, n_comp in enumerate(n_components_range):
        for id in range(reps):
            model = models[n_comp][id]
            D_JS, D_JS_shuffled = self_consistency(model, X_val, length, Nsamples=5000)

            score = model.score(X_val, lengths=lengths_val)
            score /= N

            model_SC = np.mean(D_JS) / (np.mean(D_JS_shuffled) + eps)

            data += [{'model_name':model_name,
             'model_id':id,
             'n_components':model.n_components,
             'iterations':model.monitor_.iter,
             'self_consistency':model_SC,
             'score':score}]

    return data


def self_consistency(model,X,window_length,Nsamples=10):
    T = X.shape[0]
    start_range = np.arange(window_length,T-window_length)
    relative_range = np.arange(-window_length+1, window_length-1)
    starts1 = np.random.choice(start_range,size=(Nsamples,))
    shifts = np.random.choice(relative_range,size=(Nsamples,))
    starts2 = starts1 + shifts

    #embed_space = np.zeros(window_length*3,)
    wl = window_length

    all_js_div = []
    all_js_div_shuffled = []

    for i in range(Nsamples):
        window1 = X[starts1[i]:starts1[i] + window_length]
        window2 = X[starts2[i]:starts2[i] + window_length]

        posterior1 = model.score_samples(window1)[1]
        posterior2 = model.score_samples(window2)[1]

        js_div = jenson_shannon_divergence(
            embed_in_nans(posterior1, 3 * wl, wl),
            embed_in_nans(posterior2, 3 * wl, wl + shifts[i])
        )
        all_js_div += [js_div]

        j = np.random.choice(Nsamples)
        window2_shuffled = X[starts2[j]:starts2[j] + window_length]

        posterior2 = model.score_samples(window2_shuffled)[1]

        js_div_shuffled = jenson_shannon_divergence(
            embed_in_nans(posterior1, 3 * wl, wl),
            embed_in_nans(posterior2, 3 * wl, wl + shifts[j])
        )
        all_js_div_shuffled += [js_div_shuffled]

    return np.array(all_js_div),np.array(all_js_div_shuffled)




def embed_in_nans(window,larger_window_size,start_point):
    window_length,features = window.shape
    embedded = np.zeros((larger_window_size,features))
    embedded[:] = np.nan
    embedded[start_point:start_point+window_length] = window
    return embedded


def jenson_shannon_divergence(p1,p2,compute_mean = True):
    D_KL1 = np.sum(p1 * (np.log(p1) - np.log(p2)),axis=-1)
    D_KL2 = np.sum(p2 * (np.log(p2) - np.log(p1)),axis=-1)
    D_JS = 0.5 * (D_KL1 + D_KL2)
    if compute_mean:
        D_JS = np.nanmean(D_JS)
    return D_JS


# if __name__ == '__main__':
#     length = 10
#     train_trials = 700
#     val_trials = 700
#     test_trials = 6000
#
#     GT = specify_groundtruth()
#
#     X_train,train_lengths = sample_hmm(GT,length=length*(train_trials+val_trials+test_trials),trials=1)
#
#     X_train,X_val,X_test = np.split(X_train,[length*train_trials,length*(train_trials+val_trials)])
#
#
#     val_lengths = [length]*val_trials
#     test_lengths = [length] * test_trials
#     reps = 14
#     #reps = 20
#     n_components_range = np.concatenate([np.arange(1, 2),np.arange(2,9,2),np.arange(15,21,5)])
#     #n_components_range = np.arange(6,7)
#
#     validation_X_lengths = (X_val, val_lengths)
#     #D_JS,D_JS_shuffled = self_consistency(GT,X_val,length,Nsamples=2000)
#     #plt.figure()
#     #bins = np.linspace(0,1)
#     #plt.hist(D_JS,alpha=0.4,label='shuffled',bins=bins)
#     #plt.hist(D_JS_shuffled,alpha=0.4,label='shuffled',bins=bins)
#     #plt.hist(D_JS/(D_JS_shuffled), alpha=0.4, bins=bins)
#     #plt.legend()
#     #plt.show()
#     #print()
#
#     train_lengths = None  # [length]*train_trials
#
#
#     models = {}
#     for n_comp in n_components_range:
#         models[n_comp] = []
#         for id in range(reps):
#             path = f'models_length{length}/ncomponents_{n_comp}_id{id}'
#             # train(
#             #     path,
#             #     (X_train,train_lengths),
#             #     validation_X_lengths,
#             #     #(X_val,val_lengths),
#             #     model_n_components=n_comp,
#             #     n_iter=500,
#             #     tol=1e-4,
#             #     shift_sampling_window = None
#             # )
#             model = load_model(path)
#             models[n_comp].append(model)
#
#     data = gather_self_consistency_against_validation(models, GT, validation_X_lengths,
#                                                       n_components_range=n_components_range, reps=reps,
#                                                       model_name='VanillaEM')
#
#     models = {}
#     for n_comp in n_components_range:
#         models[n_comp] = []
#         for id in range(reps):
#             path = f'models_shifted_length{length}/ncomponents_{n_comp}_id{id}'
#             # train(
#             #     path,
#             #     (X_train,train_lengths),
#             #     validation_X_lengths,
#             #     #(X_val,val_lengths),
#             #     model_n_components=n_comp,
#             #     n_iter=500,
#             #     tol=1e-3,
#             #     shift_sampling_window = 10
#             # )
#             model = load_model(path)
#             models[n_comp].append(model)
#
#     data += gather_self_consistency_against_validation(models, GT, validation_X_lengths,
#                                                       n_components_range=n_components_range, reps=reps,
#                                                       model_name='ShiftedEM')
#
#     # plot_validation_scores(models, GT, validation_X_lengths,n_components_range=n_components_range,reps=reps)
#     # plot_self_consistency(models, GT, validation_X_lengths, n_components_range=n_components_range, reps=reps)
#     # plot_self_consistency_against_validation(models, GT, validation_X_lengths, n_components_range=n_components_range, reps=reps)
#
#
#     DF = pd.DataFrame(data)
#     DF.to_csv('plots/collated.csv')


def mutual_info(joint_ab,marginal_a,marginal_b):
    eps = 1e-5
    return np.sum(
        joint_ab * (np.log(joint_ab+eps) - np.log(np.outer(marginal_a,marginal_b)+eps) )
    )

def compute_mutual_info_emission(model):
    B = model.emissionprob_
    pi = model.startprob_
    A = model.transmat_
    print(A.shape,B.shape,pi.shape)
    joint = B.T @ np.diag(pi)

    marginal_a = B.T @ pi
    marginal_b = pi
    return mutual_info(joint,marginal_a,marginal_b)

def projections_onto_features(model):
    B = model.emissionprob_
    pi = model.startprob_
    A = model.transmat_
    p_inf = model.get_stationary_distribution()
    projection_pi = B.T @ pi
    projection_pinf = B.T @ p_inf
    return projection_pi,projection_pinf



def compute_mutual_info_transition(model):
    B = model.emissionprob_
    pi = model.startprob_
    A = model.transmat_
    print(A.shape,B.shape,pi.shape)
    joint = A.T @ np.diag(pi)

    marginal_a = A.T @ pi
    marginal_b = pi
    return mutual_info(joint,marginal_a,marginal_b)


def compute_MI_predict_proba(model1,model2,test,window_length):
    hid_predicted1 = [model1.predict_proba(test[i:i+window_length]) for i in range(test.shape[0]//window_length)]
    hid_predicted1 = np.concatenate(hid_predicted1)
    hid_predicted2 = [model2.predict_proba(test[i:i+window_length]) for i in range(test.shape[0]//window_length)]
    hid_predicted2 = np.concatenate(hid_predicted2)

    joint = (hid_predicted1[:,None]*hid_predicted2[:,:,None]).mean(0).T
    p1 = hid_predicted1.mean(0)
    p2 = hid_predicted2.mean(0)
    print(joint,p1,p2)
    return mutual_info(joint,p1,p2)


def compute_entropy_steady_state(model):
    m = model.get_stationary_distribution()
    return np.sum( - m * np.log(m))

def compute_posterior_entropy(model,test,window_length):
    hid_predicted = [model.predict_proba(test[i:i+window_length]) for i in range(test.shape[0]//window_length)]
    hid_predicted = np.concatenate(hid_predicted)
    return np.sum( - hid_predicted * np.log(hid_predicted),1).mean(0)

def compute_D_JS_stationary_pi(model):
    eps = 1e-5
    dist = model.get_stationary_distribution() + eps
    pi = model.startprob_ + eps
    return jenson_shannon_divergence(dist,pi)

def compute_svd_hidden(model,test,window_length):
    hid_predicted = [model.predict_proba(test[i:i+window_length]) for i in range(test.shape[0]//window_length)]
    hid_predicted = np.concatenate(hid_predicted)
    s = np.linalg.svd(hid_predicted,compute_uv=False)
    return (s**2).sum()**2/(s**4).sum()



from sklearn.utils import check_random_state
def _generate_sample_from_state_batch(self,states,repeat = 1,random_state=None):
    cdf = np.cumsum(self.emissionprob_[states, :],axis=-1)
    random_state = check_random_state(random_state)
    return (cdf[None] > random_state.rand(repeat,*cdf.shape)).argmax(axis=-1)


def compute_joint_hidden_twomodels(model1,model2,hid1,hid2,batch_size = 5,log=True):
    #hid1 = np.stack([model1.sample(n_samples=window_length)[1] for _ in range(batch_size)])

    # obs_samp = GT._generate_sample_from_state(hid2)
    window_length = hid1.shape[0]

    obs_samp = _generate_sample_from_state_batch(model1, hid1, repeat=batch_size)
    lengths = [obs_samp.shape[1]] * obs_samp.shape[0]
    # print(obs_samp.shape)
    # print(np.stack([np.arange(10)]*3).flatten())
    hid_proba = model2.predict_proba(obs_samp.reshape(-1, 1), lengths=lengths)
    hid_proba = hid_proba.reshape(*obs_samp.shape[:2], hid_proba.shape[-1])
    hid_proba_avg = hid_proba.mean(0)
    #print(hid_proba_avg.shape,hid2.shape)
    #print(hid_proba_avg[np.arange(window_length),hid2[:,0]].shape)
    if log:
        return np.sum(np.log(hid_proba_avg[np.arange(window_length), hid2]))
    else:
        return np.prod(hid_proba_avg[np.arange(window_length), hid2])

def sample(proba,random_state=None,axis=-1):
    cdf = np.cumsum(proba,axis=axis)
    #random_state = check_random_state(random_state)
    return (cdf > np.random.uniform(size=(cdf.shape))).argmax(axis)

def compute_hid_prob(model,hid,log=True):
    # B = model.emissionprob_
    pi = model.startprob_
    A = model.transmat_

    #print(A[0].sum())
    if log:
        return  np.log(pi[hid[0]]) + np.sum(np.log(A[hid[:-1], hid[1:]]))
    else:
        return pi[hid[0]] * np.prod(A[hid[:-1], hid[1:]])

def compute_MI_twomodels(model1,model2,batch_size_main=10,batch_size_joint = 10):
    collect_samples = []
    for i in range(batch_size_main):
        obs,hid1 = model1.sample(n_samples=10)
        hid_proba = model2.predict_proba(obs,lengths=None)
        hid2 = sample(hid_proba)
        #print(hid2.shape)
        #fig,ax = plt.subplots()
        #ax = ax
        #ax.plot(hid1)
        #ax = axs[1]
        #ax.plot(hid2)
        #plt.show()
        #print(np.corrcoef(hid1,hid2)[0,1])

        log_joint = compute_joint_hidden_twomodels(model1,model2,hid1,hid2,batch_size=batch_size_joint,log=True)
        log_hid2_prob = compute_hid_prob(model2,hid2,log=True)
        #print(log_joint, log_hid2_prob)
        collect_samples += [log_joint - log_hid2_prob]

    return collect_samples


def compute_data_entropy(observations):
    ### i.e entropy of data
    bincounts = np.bincount(observations[:,0])
    pmf = bincounts/len(observations[:,0])
    return -(pmf * np.log(pmf)).sum()



@hydra.main(version_base='1.3', config_path=CONFIG_PATH, config_name=CONFIG_NAME)
def main(cfg):
    OmegaConf.resolve(cfg)
    #print(OmegaConf.to_yaml(cfg))


    length       = cfg.length
    train_trials = cfg.train_trials
    val_trials   = cfg.val_trials
    test_trials  = cfg.test_trials


    #print(hydra.utils.instantiate(cfg.model))

    train, val, test = hydra.utils.instantiate(cfg.all_data_tuples)

    if cfg.run_train:
        model_partial = hydra.utils.instantiate(cfg.model)
        lengths = hydra.utils.instantiate(cfg.training_specs.lengths)
        lengths_val = hydra.utils.instantiate(cfg.training_specs.lengths)
        model = model_partial(X_train = train,lengths = lengths)
        model.fit(X=train,lengths = lengths,X_val = val, lengths_val=lengths_val )
        if cfg.repeat_fit_without_shift:
            model.shift_limits = 0
            model.fit(X=train, lengths=lengths)

        model_save_path = cfg.model_save_path
        if not os.path.exists(os.path.dirname(model_save_path)):
            os.makedirs(os.path.dirname(model_save_path))
        with open(model_save_path,'wb') as f:
            pkl.dump(model,f)


    if cfg.run_analysis:
        if cfg.use_groundtruth_as_model:
            model = hydra.utils.instantiate(cfg.groundtruth)
            data = {'model_name': 'Groundtruth',
                    'model_id': None}
            results_path = os.path.join(cfg.groundtruth_savepath,'groundtruth')
        else:
            model_save_path = cfg.model_save_path
            with open(model_save_path,'rb') as f:
                model = pkl.load(f)
            data = {'model_name': cfg.training_mode_choice,
                    'model_id': cfg.model_index}

            results_path = cfg.model_save_path

        if cfg.analysis.compute_test_score_and_SC:
            test_score,test_model_SC = self_consistency_and_validation_single(model,
                                                                              (test,[length]*(test.shape[0]//length)),
                                                                              window_length=length )

            data.update({'n_components': model.n_components,
                         'iterations': model.monitor_.iter,
                         'test_self_consistency': test_model_SC,
                         'test_score': test_score,
                         'train_trials': cfg.train_trials})

        if cfg.analysis.compute_train_score_and_SC:
            train_score, train_model_SC = self_consistency_and_validation_single(model, (train, [length] * (train.shape[0] // length)),
                                                                               window_length=length)
            data.update({'train_self_consistency': train_model_SC,
                         'train_score': train_score})

        if cfg.analysis.compute_SSent:
            SSent = compute_entropy_steady_state(model)
            data.update({'steady_state_entropy':SSent})

        if cfg.analysis.compute_MI_emission:
            MI_emission = compute_mutual_info_emission(model)
            data.update({'MI_emission': MI_emission})

        if cfg.analysis.compute_MI_trans:
            MI_transition = compute_mutual_info_transition(model)
            data.update({'MI_transition': MI_transition})

        if cfg.analysis.compute_D_JS_stationary_pi:
            D_JS_stationary_pi = compute_D_JS_stationary_pi(model)
            data.update({'D_JS_stationary_pi': D_JS_stationary_pi})

        if cfg.analysis.compute_proj:
            projection_pi,projection_pinf = projections_onto_features(model)
            data.update({
                'projection_pi': projection_pi[0],
                'projection_pinf':projection_pinf[0]
            })

        if cfg.analysis.compute_svd_hidden:
            PR = compute_svd_hidden(model,test[:],window_length=cfg.length)
            data.update({'PR':PR})

        if cfg.analysis.compute_MI_models:
            GT = hydra.utils.instantiate(cfg.groundtruth)
            log_values = compute_MI_twomodels(model, GT, batch_size_joint=300, batch_size_main=2000)
            data.update({
                'mean_MI_with_GT'       : np.mean(log_values),
                'confidence_MI_with_GT' : np.std(log_values) / np.sqrt(len(log_values))
            })

        if cfg.analysis.compute_MI_predict_proba:
            GT = hydra.utils.instantiate(cfg.groundtruth)
            MI = compute_MI_predict_proba(model, GT, test[:], window_length=cfg.length)
            data.update({
                'MI_predict_proba'       : MI
            })

        if cfg.analysis.compute_posterior_entropy:
            posterior_entropy = compute_posterior_entropy(model,test[:2000], window_length=cfg.length)
            data.update({
                'posterior_entropy'       : posterior_entropy
            })

        if cfg.analysis.compute_data_entropy:
            minus_test_entropy = -compute_data_entropy(test)
            data.update({
                'minus_test_entropy': minus_test_entropy
            })

        save_results_loc = results_path + '.csv'
        if os.path.exists(save_results_loc):
            DFread = pd.read_csv(save_results_loc, index_col=None)
            data_dict = DFread.T.to_dict()[0]
            data_dict.update(data)
            data = data_dict


        print(data)
        DF = pd.DataFrame([data])
        DF.to_csv(save_results_loc, index=False)


if __name__ == '__main__':
    main()
    #print(specify_groundtruth_state(5,4).transmat_)
    #print(specify_groundtruth().transmat_.sum(1))

    # GT = specify_groundtruth_state(5,5)
    #
    # obs, hid = GT.sample(n_samples=10)
    # hid = hid[:,None]

    #print(compute_joint_hidden_twomodels(GT,GT,hid,hid))
    #print(compute_hid_prob(GT,hid))
    #print()
    # log_values = compute_MI_twomodels(GT,GT,batch_size_joint = 300,batch_size_main = 2000)
    # plt.hist(log_values,bins=50)
    # plt.show()

    #print(np.mean(log_values),np.std(log_values)/np.sqrt(len(log_values)))

    #compute_MI_twomodels(GT, GT, batch_size_joint=500, batch_size_main=1)

    # batch_size = 5
    # hid2 = np.stack([GT.sample(n_samples=10)[1] for _ in range(batch_size)])
    #
    #
    # #obs_samp = GT._generate_sample_from_state(hid2)
    # obs_samp = _generate_sample_from_state_batch(GT,hid2)
    # lengths = [obs_samp.shape[1]]*obs_samp.shape[0]
    # #print(obs_samp.shape)
    # #print(np.stack([np.arange(10)]*3).flatten())
    # hid_proba = GT.predict_proba(obs_samp.reshape(-1, 1),lengths=lengths)
    # hid_proba = hid_proba.reshape(*obs_samp.shape[:2],hid_proba.shape[-1])
    # hid_proba_avg = hid_proba.mean(0)
    # print(hid_proba_avg.shape)
    #
