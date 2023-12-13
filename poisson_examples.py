import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from omegaconf import DictConfig, OmegaConf
import hydra
from copy import deepcopy
from main import compute_svd_hidden
from sklearn.utils import check_random_state
from hmmlearn.hmm import GaussianHMM, CategoricalHMM, PoissonHMM, BernoulliHMM
from hmmlearn.vhmm import VariationalCategoricalHMM
from prepare_model import *
from metrics import bernoulli_bits_per_spike
import pickle as pkl
import os
from nlb_tools.evaluation import bits_per_spike,neg_log_likelihood
from prepare_model import CoHMM
from metrics import normalised_score, normalised_cosmoothing_score,normalised_cosmoothing_score2
import matplotlib as mpl
import seaborn as sns
from utils import normalise

mpl.rcParams['text.usetex'] = True

CONFIG_PATH = "configs"
# CONFIG_NAME = "poisson_example_config"



def main():
    length = 10
    train_trials = 100
    test_trials = 500

    n_components = 2
    n_heldin = 100
    n_heldout = 100
    trans_eps = 5e-1
    exp_scale = 0.7

    generator_model = PoissonHMM(n_components=n_components)
    generator_model.n_features = n_heldin + n_heldout
    generator_model.lambdas_ = np.random.exponential(exp_scale,size=(n_components,n_heldin+n_heldout))
    # generator_model.lambdas_ = np.exp(exp_scale*np.random.uniform(size=(n_components,n_heldin+n_heldout)))
    generator_model.startprob_ = normalise(np.ones(n_components))
    generator_model.transmat_ = normalise(np.roll(np.eye(n_components),1,axis=1) + trans_eps,axis=1)
    groundtruth = generator_model

    train_data = sample_hmm(groundtruth, length=length, trials=train_trials, seed_base=2, flatten=False)
    test_data = sample_hmm(groundtruth, length=length, trials=test_trials, seed_base=23454, flatten=False)



    #
    encoder, decoder = split_model_emission(
        generator_model,
        split_index=n_heldin
    )
    decoder.params = 'l'

    model = CoHMM(
        encoder=encoder,
        decoder=decoder,
    )

    corrupted_model = deepcopy(model)
    corrupted_model.encoder.lambdas_ = corrupted_model.encoder.lambdas_[:, np.random.permutation(n_heldin)]
    corrupted_model.decoder.lambdas_ = corrupted_model.decoder.lambdas_[:, np.random.permutation(n_heldin)]


    models = [model,corrupted_model]
    names = ['Ground truth','Corrupted']

    # testing
    test_heldin, test_heldout = np.split(test_data, [n_heldin], axis=-1)
    train_heldin, train_heldout = np.split(train_data, [n_heldin], axis=-1)

    co_bps_scores = []
    k_shot_scores = []
    for name,model in zip(names,models):
        pred_test_heldout = model.predict(
            test_heldin.reshape(-1,n_heldin),
            lengths=[length]*test_trials
        )
        pred_test_heldout = pred_test_heldout.reshape(test_trials, length, n_heldout)
        co_bps = bits_per_spike(
            pred_test_heldout,
            test_heldout
        )
        co_bps_scores.append(co_bps)


        modelMLE = deepcopy(model)
        modelMLE.co_fit(
            train_heldin.reshape(-1,n_heldin),
            train_heldout.reshape(-1,n_heldout),
            lengths=[length]*train_trials
        )

        pred_test_heldout = modelMLE.predict(
            test_heldin.reshape(-1, n_heldin),
            lengths=[length] * test_trials
        )
        pred_test_heldout = pred_test_heldout.reshape(test_trials, length, n_heldout)
        co_bps = bits_per_spike(
            pred_test_heldout,
            test_heldout
        )
        k_shot_scores.append(co_bps)




    print('co_bps scores')
    for name,score in zip(names,co_bps_scores):
        print(name,':',score)

    print('k-shot scores')
    for name,score in zip(names,k_shot_scores):
        print(name,':',score)

    #
    # proj_in  = (model.encoder.lambdas_[None] @ test_heldin.swapaxes(-1,-2)).swapaxes(-1,-2)
    # proj_out = (model.decoder.lambdas_[None] @ pred_test_heldout.swapaxes(-1, -2)).swapaxes(-1,-2)
    #
    # print('projection score:',(np.argmax(proj_in,axis=-1) == np.argmax(proj_out,axis=-1)).mean())


    # plt.imshow(state_proba[0,:,:].T)
    # plt.scatter(proj_in[:,:,0],proj_in[:,:,1])
    # plt.scatter(proj_out[:, :, 0], proj_out[:, :, 1])
    # plt.hist(pred_test_heldout.flatten(),density=True,bins=40)
    # plt.plot(np.bincount(test_data.flatten())/len(test_data.flatten()))
    # plt.plot(np.bincount(test_data.flatten())/len(test_data.flatten()))
    # proj = generator_model.lambdas_ @ test_data[:,0,:].T
    # plt.imshow(test_data[0].T)
    # plt.scatter(proj[0],proj[1])
    # plt.show()




def main():
    np.random.seed(0)
    length = 10
    train_trials = 3
    test_trials = 500

    n_components = 2
    n_heldin = 2
    n_heldout = 100
    trans_eps = 1e-1
    power_squeeze = 1


    generator_model = BernoulliHMM(n_components=n_components)
    generator_model.n_features = n_heldin + n_heldout
    generator_model.lambdas_ = np.random.uniform(size=(n_components,n_heldin+n_heldout))**power_squeeze
    generator_model.startprob_ = normalise(np.ones(n_components))
    generator_model.transmat_ = normalise(np.roll(np.eye(n_components) + trans_eps,1,axis=1),axis=1)
    groundtruth = generator_model

    train_data = sample_hmm(groundtruth, length=length, trials=train_trials, seed_base=2, flatten=False)
    test_data = sample_hmm(groundtruth, length=length, trials=test_trials, seed_base=23454, flatten=False)

    # plt.imshow(test_data[0].T)
    # plt.show()

    encoder, decoder = split_model_emission(
        generator_model,
        split_index=n_heldin
    )
    decoder.params = 'l'

    model = CoHMM(
        encoder=encoder,
        decoder=decoder,
    )

    corrupted_model = deepcopy(model)
    # corrupted_model.encoder.lambdas_ = corrupted_model.encoder.lambdas_[:, np.random.permutation(n_heldin)]
    # corrupted_model.decoder.lambdas_ = corrupted_model.decoder.lambdas_ #[:, np.random.permutation(n_heldin)]
    corrupted_model.encoder.transmat_ = normalise(
        # np.roll(corrupted_model.encoder.transmat_,1,axis=1),
        corrupted_model.encoder.transmat_[:,:] + 1e-1,
        axis=1
    )

    long_model = deepcopy(model)

    factor = 5
    n_components *= factor
    long_model.encoder.n_components = n_components
    long_model.decoder.n_components = n_components
    long_model.encoder.lambdas_ = np.concatenate([model.encoder.lambdas_]*factor,axis=0)
    long_model.decoder.lambdas_ = np.concatenate([model.decoder.lambdas_] * factor, axis=0)
    for i in range(factor):
        long_model.encoder.lambdas_[i::factor, :] = model.encoder.lambdas_
        long_model.decoder.lambdas_[i::factor, :] = model.decoder.lambdas_
    long_model.encoder.startprob_ = normalise(np.ones(n_components))
    long_model.encoder.transmat_ = normalise(np.roll(np.eye(n_components),1,axis=1) ,axis=1)
    # print(model.encoder.transmat_)
    # print(corrupted_model.encoder.transmat_)

    models = [model,corrupted_model,long_model]
    names = ['Ground truth','Corrupted','Long']

    # testing
    test_heldin, test_heldout = np.split(test_data, [n_heldin], axis=-1)
    train_heldin, train_heldout = np.split(train_data, [n_heldin], axis=-1)

    co_bps_scores = []
    k_shot_scores = []
    for name,mod in zip(names,models):
        pred_test_heldout = mod.predict(
            test_heldin.reshape(-1,n_heldin),
            lengths=[length]*test_trials
        )
        pred_test_heldout = pred_test_heldout.reshape(test_trials, length, n_heldout)
        co_bps = bernoulli_bits_per_spike(
            pred_test_heldout,
            test_heldout
        )
        co_bps_scores.append(co_bps)

        # train_heldout = np.random.choice(2, size=train_heldout.shape, p=np.array([0.1, 0.9])).astype(bool)
        # X[:,:,1] = ~X[:,:,1]
        modelMLE = deepcopy(mod)
        modelMLE.decoder.lambdas_ = modelMLE.decoder.lambdas_[:,np.random.permutation(n_heldout)]
        modelMLE.co_fit(
            train_heldin.reshape(-1,n_heldin),
            train_heldout.reshape(-1,n_heldout),
            lengths=[length]*train_trials
        )

        # print(model.decoder.lambdas_)
        # print(modelMLE.decoder.lambdas_)
        # plt.scatter(mod.decoder.lambdas_,modelMLE.decoder.lambdas_)
        # plt.savefig(f'plots/test_plots/{name}_lambdas.png')
        # plt.close()

        pred_test_heldout = modelMLE.predict(
            test_heldin.reshape(-1, n_heldin),
            lengths=[length] * test_trials
        )
        pred_test_heldout = pred_test_heldout.reshape(test_trials, length, n_heldout)
        # print(name,pred_test_heldout)
        # test_heldout = np.random.choice(2, size=test_heldout.shape, p=np.array([0.1, 0.9])).astype(int)
        plt.hist(pred_test_heldout.flatten(),bins=30)
        plt.savefig(f'plots/test_plots/{name}_pred_hist.png')
        plt.close()

        co_bps = bernoulli_bits_per_spike(
            pred_test_heldout,
            test_heldout
        )
        k_shot_scores.append(co_bps)



    print('co_bps scores')
    for name,score in zip(names,co_bps_scores):
        print(name,':',score)

    print('k-shot scores')
    for name,score in zip(names,k_shot_scores):
        print(name,':',score)

    titles = names

    fig, axs = plt.subplots(4, len(models), figsize=(len(models)*4, 12))

    fmt = '.2f'
    for i,mod in  enumerate(models):
        # print(normalised_score(mod,(test_data,[length]*test_trials)))

        # score = normalised_cosmoothing_score2(mod, (test_data, [length] * test_trials))
        # PR = compute_svd_hidden(mod,test_data[0],window_length=length)
        # print(titles[i],'PR:{:.4f}, score:{:.4f}'.format(PR,score))
        # prob = model.predict_proba(test_data[0],lengths=[length] * test_trials)

        fontsize = 15
        ax_row = axs[0]
        ax_row[i].set_title(titles[i])
        # ax_row[i].imshow(mod.transmat_,vmin=0,vmax=1)
        # print(mod.transmat_)
        sns.heatmap(mod.encoder.transmat_,vmin=0,vmax=1,annot=True,fmt=fmt,ax=ax_row[i],cbar=False)
        ax_row[0].set_ylabel(r'$A$',fontsize=fontsize)

        ax_row = axs[1]
        # ax_row[i].imshow(mod.emissionprob_,vmin=0,vmax=1)
        sns.heatmap(mod.encoder.lambdas_, vmin=0, vmax=1, annot=True, fmt=fmt, ax=ax_row[i], cbar=False)
        ax_row[0].set_ylabel(r'$B^{in}$',fontsize=fontsize)

        ax_row = axs[2]
        prob = mod.encoder.predict_proba(test_heldin[1,:length])
        # ax_row[i].imshow(prob.T,vmin=0,vmax=1)
        sns.heatmap(prob.T, vmin=0, vmax=1, annot=True, fmt=fmt, ax=ax_row[i], cbar=False,square=True)
        # print(mod.emissionprob_)
        ax_row[0].set_ylabel(r'$p(x_t|y_{1:T})$',fontsize=fontsize)

        ax_row = axs[3]
        # ax_row[i].imshow(mod.emissionprob2_,vmin=0,vmax=1)
        sns.heatmap(mod.decoder.lambdas_, vmin=0, vmax=1, annot=True, fmt=fmt, ax=ax_row[i], cbar=False,square=True)
        ax_row[0].set_ylabel(r'$\hat B^{out}$',fontsize=fontsize)
    fig.tight_layout()
    fig.savefig('plots/test_plots/three_models.png',dpi=200)
    fig.savefig('plots/test_plots/three_models.pdf')


def main():
    np.random.seed(0)
    length = 10
    train_trials = 5
    test_trials = 500

    n_components = 5
    n_heldin = 20
    n_heldout = 100
    trans_eps = 1e-2
    power_squeeze = 1
    frac_perm = 0.4


    generator_model = BernoulliHMM(n_components=n_components)
    generator_model.n_features = n_heldin + n_heldout
    generator_model.lambdas_ = np.random.uniform(size=(n_components,n_heldin+n_heldout))**power_squeeze
    generator_model.startprob_ = normalise(np.ones(n_components))
    generator_model.transmat_ = normalise(np.roll(np.eye(n_components) + trans_eps,1,axis=1),axis=1)
    groundtruth = generator_model

    train_data = sample_hmm(groundtruth, length=length, trials=train_trials, seed_base=2, flatten=False)
    test_data = sample_hmm(groundtruth, length=length, trials=test_trials, seed_base=23454, flatten=False)

    # plt.imshow(test_data[0].T)
    # plt.show()

    encoder, decoder = split_model_emission(
        generator_model,
        split_index=n_heldin
    )
    decoder.params = 'l'

    model = CoHMM(
        encoder=encoder,
        decoder=decoder,
    )

    corrupted_model = deepcopy(model)
    corrupted_model.encoder.lambdas_[:,:int(n_heldin*frac_perm)] = corrupted_model.encoder.lambdas_[:, np.random.permutation(int(n_heldin*frac_perm))]
    # corrupted_model.decoder.lambdas_ = corrupted_model.decoder.lambdas_ #[:, np.random.permutation(n_heldin)]
    # corrupted_model.encoder.transmat_ = normalise(
    #     # np.roll(corrupted_model.encoder.transmat_,1,axis=1),
    #     corrupted_model.encoder.transmat_[:,:] + 1e-1,
    #     axis=1
    # )


    long_model = deepcopy(model)

    factor = 3
    n_components_long = n_components * factor
    long_model.encoder.n_components = n_components_long
    long_model.decoder.n_components = n_components_long
    long_model.encoder.lambdas_ = np.concatenate([model.encoder.lambdas_] * factor,axis=0)
    long_model.decoder.lambdas_ = np.concatenate([model.decoder.lambdas_] * factor, axis=0)
    # for i in range(factor):
    #     long_model.encoder.lambdas_[i::factor, :] = model.encoder.lambdas_
    #     long_model.decoder.lambdas_[i::factor, :] = model.decoder.lambdas_
    long_model.encoder.startprob_ = normalise(
        np.repeat(
            (np.arange(factor)==0).astype(float),
            n_components)
    )
    long_model.encoder.transmat_ = normalise(np.kron(
        np.roll(np.eye(factor), 1, axis=1),
        model.encoder.transmat_
    ),axis=1)

    # print(model.encoder.transmat_)
    # print(corrupted_model.encoder.transmat_)

    models = [model,corrupted_model,long_model]
    names = ['Ground truth','Corrupted','Long']

    # testing
    test_heldin, test_heldout = np.split(test_data, [n_heldin], axis=-1)
    train_heldin, train_heldout = np.split(train_data, [n_heldin], axis=-1)

    co_bps_scores = []
    k_shot_scores = []
    for name,mod in zip(names,models):
        pred_test_heldout = mod.predict(
            test_heldin.reshape(-1,n_heldin),
            lengths=[length]*test_trials
        )
        pred_test_heldout = pred_test_heldout.reshape(test_trials, length, n_heldout)
        co_bps = bernoulli_bits_per_spike(
            pred_test_heldout,
            test_heldout
        )
        co_bps_scores.append(co_bps)

        # train_heldout = np.random.choice(2, size=train_heldout.shape, p=np.array([0.1, 0.9])).astype(bool)
        # X[:,:,1] = ~X[:,:,1]
        modelMLE = deepcopy(mod)
        modelMLE.decoder.lambdas_ = modelMLE.decoder.lambdas_[:,np.random.permutation(n_heldout)]
        modelMLE.co_fit(
            train_heldin.reshape(-1,n_heldin),
            train_heldout.reshape(-1,n_heldout),
            lengths=[length]*train_trials
        )
        # mod.decoder.lambdas2_  = modelMLE.decoder.lambdas_
        # print(model.decoder.lambdas_)
        # print(modelMLE.decoder.lambdas_)
        # plt.scatter(mod.decoder.lambdas_,modelMLE.decoder.lambdas_)
        # plt.savefig(f'plots/test_plots/{name}_lambdas.png')
        # plt.close()

        pred_test_heldout = modelMLE.predict(
            test_heldin.reshape(-1, n_heldin),
            lengths=[length] * test_trials
        )
        pred_test_heldout = pred_test_heldout.reshape(test_trials, length, n_heldout)
        # print(name,pred_test_heldout)
        # test_heldout = np.random.choice(2, size=test_heldout.shape, p=np.array([0.1, 0.9])).astype(int)
        # plt.hist(pred_test_heldout.flatten(),bins=30)
        # plt.savefig(f'plots/test_plots/{name}_pred_hist.png')
        # plt.close()

        co_bps = bernoulli_bits_per_spike(
            pred_test_heldout,
            test_heldout
        )
        k_shot_scores.append(co_bps)



    print('co_bps scores')
    for name,score in zip(names,co_bps_scores):
        print(name,':',score)

    print('k-shot scores')
    for name,score in zip(names,k_shot_scores):
        print(name,':',score)

    titles = names

    fig, axs = plt.subplots(4, len(models), figsize=(len(models)*4, 12))

    fmt = '.2f'
    for i,mod in  enumerate(models):
        # print(normalised_score(mod,(test_data,[length]*test_trials)))

        # score = normalised_cosmoothing_score2(mod, (test_data, [length] * test_trials))
        # PR = compute_svd_hidden(mod,test_data[0],window_length=length)
        # print(titles[i],'PR:{:.4f}, score:{:.4f}'.format(PR,score))
        # prob = model.predict_proba(test_data[0],lengths=[length] * test_trials)

        fontsize = 15
        ax_row = axs[0]
        ax_row[i].set_title(titles[i])
        # ax_row[i].imshow(mod.transmat_,vmin=0,vmax=1)
        # print(mod.transmat_)
        sns.heatmap(mod.encoder.transmat_,vmin=0,vmax=1,annot=True,fmt=fmt,ax=ax_row[i],cbar=False)
        ax_row[0].set_ylabel(r'$A$',fontsize=fontsize)

        ax_row = axs[1]
        # ax_row[i].imshow(mod.emissionprob_,vmin=0,vmax=1)
        sns.heatmap(mod.encoder.lambdas_, vmin=0, vmax=1, annot=True, fmt=fmt, ax=ax_row[i], cbar=False)
        ax_row[0].set_ylabel(r'$B^{in}$',fontsize=fontsize)

        ax_row = axs[2]
        prob = mod.encoder.predict_proba(test_heldin[1,:length])
        # ax_row[i].imshow(prob.T,vmin=0,vmax=1)
        sns.heatmap(prob.T, vmin=0, vmax=1, annot=True, fmt=fmt, ax=ax_row[i], cbar=False,square=True)
        # print(mod.emissionprob_)
        ax_row[0].set_ylabel(r'$p(x_t|y_{1:T})$',fontsize=fontsize)

        ax_row = axs[3]
        # ax_row[i].imshow(mod.emissionprob2_,vmin=0,vmax=1)
        sns.heatmap(mod.decoder.lambdas_, vmin=0, vmax=1, annot=True, fmt=fmt, ax=ax_row[i], cbar=False,square=True)
        # ax_row[i].scatter()
        ax_row[0].set_ylabel(r'$\hat B^{out}$',fontsize=fontsize)
    fig.tight_layout()
    fig.savefig('plots/test_plots/three_models.png',dpi=200)
    fig.savefig('plots/test_plots/three_models.pdf')


if __name__ == '__main__':
    main()
