import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from omegaconf import DictConfig, OmegaConf
import hydra
from copy import deepcopy
from main import compute_svd_hidden
from sklearn.utils import check_random_state
from hmmlearn.hmm import GaussianHMM, CategoricalHMM, PoissonHMM, MultinomialHMM
from hmmlearn.vhmm import VariationalCategoricalHMM
from prepare_model import *
import pickle as pkl
import os
from metrics import normalised_score, normalised_cosmoothing_score,normalised_cosmoothing_score2
import matplotlib as mpl
import seaborn as sns
mpl.rcParams['text.usetex'] = True


#mpl.rcParams['text.latex.preamble'] = [r'\usepackage{amsmath}']

CONFIG_PATH = "configs"
# CONFIG_NAME = "config"
CONFIG_NAME = "toy_example2_config"

OmegaConf.register_new_resolver("eval", eval)
OmegaConf.register_new_resolver("ind", lambda a,i: a[i])
OmegaConf.register_new_resolver("listmul", lambda l, i: [l]*i)
OmegaConf.register_new_resolver("getattr", getattr)


@hydra.main(version_base=None, config_path=CONFIG_PATH, config_name=CONFIG_NAME)
def main_plot(cfg):
    groundtruth = hydra.utils.instantiate(cfg.groundtruth)
    print(groundtruth)
    eps = 1e-1
    groundtruth.emissionprob_ = np.roll(np.eye(2), 1, axis=0) + eps
    groundtruth.emissionprob_ /= groundtruth.emissionprob_.sum(1,keepdims = True)
    print(groundtruth.emissionprob_)
    groundtruth.startprob_ = np.array([0.5,0.5])
    groundtruth.transmat_ = np.roll(np.eye(2), 1, axis=0)
    print(groundtruth.transmat_)
    outputs,state = groundtruth.sample(n_samples=11,random_state=4)
    # outputs = np.concatenate([np.ones(11,dtype=int),np.zeros(11,dtype=int)])[:,None]

    model = deepcopy(groundtruth)

    eps = 1e0
    model.transmat_ = np.roll(np.eye(2), 1, axis=0) + eps
    model.transmat_ /= model.transmat_.sum(1, keepdims=True)

    prob = model.predict_proba(outputs)
    fig = plt.figure(figsize=(8,3))
    ax = fig.add_subplot()
    im = ax.imshow(prob.T,vmin=0,vmax=1)
    fig.colorbar(im,ax=ax,shrink=0.7)
    # fig.tight_layout()
    print(prob)
    fig.savefig('plots/test_plots/alternating_prob.png',dpi=200)

@hydra.main(version_base=None, config_path=CONFIG_PATH, config_name=CONFIG_NAME)
def main3_plot(cfg):
    groundtruth = hydra.utils.instantiate(cfg.groundtruth3)
    print(groundtruth)
    eps = 1e-1
    groundtruth.emissionprob_ = np.eye(3) + eps
    groundtruth.emissionprob_ /= groundtruth.emissionprob_.sum(0,keepdims = True)
    print(groundtruth.emissionprob_)
    groundtruth.startprob_ = np.ones(3)
    groundtruth.startprob_ /= groundtruth.startprob_.sum()
    # groundtruth.transmat_ = np.eye(3)[::-1]
    print(groundtruth.transmat_)
    outputs,state = groundtruth.sample(n_samples=11,random_state=4)
    # outputs = np.concatenate([np.ones(11,dtype=int),np.zeros(11,dtype=int)])[:,None]

    model = deepcopy(groundtruth)

    eps = 3e-1
    model.transmat_ += eps
    model.transmat_ /= model.transmat_.sum(0, keepdims=True)

    prob = model.predict_proba(outputs)
    fig = plt.figure(figsize=(8,3))
    ax = fig.add_subplot()
    im = ax.imshow(prob.T,vmin=0,vmax=1)
    fig.colorbar(im,ax=ax,shrink=0.7)
    # fig.tight_layout()
    print(prob)
    fig.savefig('plots/test_plots/alternating_prob.png',dpi=200)


@hydra.main(version_base=None, config_path=CONFIG_PATH, config_name=CONFIG_NAME)
def main2(cfg):
    length = 10
    train_trials = 2
    test_trials = 500

    groundtruth = hydra.utils.instantiate(cfg.groundtruth)
    print(groundtruth)
    eps = 1e-1
    groundtruth.emissionprob_ = np.roll(np.eye(2), 1, axis=0) + eps
    groundtruth.emissionprob_ /= groundtruth.emissionprob_.sum(0,keepdims = True)
    groundtruth.startprob_ = np.array([0.5,0.5])

    model = deepcopy(groundtruth)
    groundtruth_model = deepcopy(groundtruth)

    eps = 1e0
    model.transmat_ = np.roll(np.eye(2), 1, axis=0) + eps
    model.transmat_ /= model.transmat_.sum(0, keepdims=True)

    # print(groundtruth.transmat_)
    # outputs,state = groundtruth.sample(n_samples=10,random_state=4)
    train_data = sample_hmm(groundtruth, length = length, trials = train_trials, seed_base=2)
    test_data  = sample_hmm(groundtruth, length = length, trials=test_trials, seed_base=23454)

    for mod in [groundtruth_model,model]:
        # mod.params = 'e'
        # mod.fit(train_data,lengths=[length]*train_trials)
        fit_model_emission(mod,train_data,train_data,lengths=[length]*train_trials)

    for mod in [groundtruth,groundtruth_model,model]:
        print(normalised_score(mod,(test_data,[length]*test_trials)))


@hydra.main(version_base=None, config_path=CONFIG_PATH, config_name=CONFIG_NAME)
def main5(cfg):
    length = 10
    train_trials = 50
    test_trials = 500

    groundtruth = hydra.utils.instantiate(cfg.groundtruth5)
    n = groundtruth.transmat_.shape[0]
    print(groundtruth.emissionprob_.shape)
    eps = 1e-3
    groundtruth.emissionprob_ = np.eye(n) + eps #* np.eye(n)[::-1]
    groundtruth.emissionprob_ /= groundtruth.emissionprob_.sum(1,keepdims = True)
    groundtruth.startprob_ = np.ones(n)
    groundtruth.startprob_ /= groundtruth.startprob_.sum()
    # groundtruth.emissionprob_ = np.roll(groundtruth.emissionprob_,2,axis=0)
    groundtruth.emissionprob2_ = groundtruth.emissionprob_[::-1]

    model = deepcopy(groundtruth)
    groundtruth_model = deepcopy(groundtruth)

    eps = 2e-1
    model.transmat_ += eps# * np.eye(n)[::-1]
    model.transmat_ /= model.transmat_.sum(1, keepdims=True)

    train_data = sample_hmm(groundtruth, length = length, trials = train_trials, seed_base=0, use_emmision2=True)
    test_data  = sample_hmm(groundtruth, length = length, trials = test_trials, seed_base=23454, use_emmision2=True)


    for mod in [groundtruth_model,model]:
        # mod.params = 'e'
        # mod.fit(train_data,lengths=[length]*train_trials)

        fit_model_emission(mod, train_data, train_data, lengths=[length] * train_trials)

    fig, axs = plt.subplots(4, 3, figsize=(10, 12))
    titles = ['Ground-truth',f'Ground-truth {train_trials}-shot',f'Corrupted Ground-truth {train_trials}-shot']

    for i,mod in  enumerate([groundtruth,groundtruth_model,model]):
        # print(normalised_score(mod,(test_data,[length]*test_trials)))

        print(normalised_cosmoothing_score(mod, (test_data, [length] * test_trials)))
        ax_row = axs[0]
        ax_row[i].set_title(titles[i])
        ax_row[i].imshow(mod.transmat_,vmin=0,vmax=1)
        ax_row[0].set_ylabel(r'$A$')

        ax_row = axs[1]
        ax_row[i].imshow(mod.emissionprob_,vmin=0,vmax=1)
        ax_row[0].set_ylabel(r'$B^{in}$')

        ax_row = axs[2]
        ax_row[i].imshow(mod.emissionprob2_,vmin=0,vmax=1)
        ax_row[0].set_ylabel(r'$B^{out}$')

        ax_row = axs[3]
        prob = model.predict_proba(test_data[:length])
        ax_row[i].imshow(prob.T,vmin=0,vmax=1)
        # print(mod.emissionprob_)
        ax_row[0].set_ylabel(r'$p(x_t|y_{1:T})$')
    fig.tight_layout()
    fig.savefig('plots/test_plots/three_models_emission.png')


@hydra.main(version_base=None, config_path=CONFIG_PATH, config_name=CONFIG_NAME)
def main6(cfg):
    length = 10
    train_trials = 5
    train_trials2 = 50
    test_trials = 500

    groundtruth = hydra.utils.instantiate(cfg.groundtruth5)
    n = groundtruth.transmat_.shape[0]
    print(groundtruth.emissionprob_.shape)
    eps = 1e-1

    groundtruth.transmat_ = np.roll(np.eye(n),1,axis=1) + 1e-1
    groundtruth.transmat_ /= groundtruth.transmat_.sum(1, keepdims=True)
    groundtruth.emissionprob_ = np.eye(n) + eps# * np.eye(n)[::-1]
    groundtruth.emissionprob_ /= groundtruth.emissionprob_.sum(1,keepdims = True)

    groundtruth.startprob_ = np.ones(n)
    groundtruth.startprob_ /= groundtruth.startprob_.sum()
    # groundtruth.emissionprob_ = np.roll(groundtruth.emissionprob_,2,axis=0)
    groundtruth.emissionprob2_ = groundtruth.emissionprob_[::-1]
    groundtruth.coemissionprob_ = np.stack([groundtruth.emissionprob_, groundtruth.emissionprob2_])

    model = deepcopy(groundtruth)

    groundtruth_model = deepcopy(groundtruth)



    eps = 0
    model.emissionprob_[:, -2:] = np.roll(model.emissionprob_[:, -2:], 1, axis=1)
    # model.transmat_[-2:] = np.roll(model.transmat_[-2:], 1, axis=0)
    # model.transmat_ += eps * np.eye(n)[::-1]
    # model.transmat_ /= model.transmat_.sum(1, keepdims=True)

    model2 = deepcopy(model)
    groundtruth_model2 = deepcopy(groundtruth_model)

    train_data = cosample_hmm(groundtruth,length = length, trials = train_trials, seed_base=0)
    train_data2 = cosample_hmm(groundtruth, length=length, trials=train_trials2, seed_base=87643)
    test_data  = cosample_hmm(groundtruth,length = length, trials = test_trials, seed_base=23454)



    for mod in [groundtruth_model,model]:
        # mod.params = 'e'
        # mod.fit(train_data,lengths=[length]*train_trials)
        # print(train_data[0].shape,train_data[1].shape,length,train_trials)
        fit_model_emission(mod, train_data[0], train_data[1], lengths=[length] * train_trials)
        mod.coemissionprob_ = np.stack([mod.emissionprob_, mod.emissionprob2_])

    for mod in [groundtruth_model2,model2]:
        # mod.params = 'e'
        # mod.fit(train_data,lengths=[length]*train_trials)
        # print(train_data2[0].shape,train_data2[1].shape,length,train_trials)
        fit_model_emission(mod, train_data2[0], train_data2[1], lengths=[length] * train_trials2)
        mod.coemissionprob_ = np.stack([mod.emissionprob_, mod.emissionprob2_])

    fig, axs = plt.subplots(4, 5, figsize=(18, 12))
    titles = ['Ground-truth',
              f'Ground-truth {train_trials}-shot',
              f'Corrupted Ground-truth {train_trials}-shot',
              f'Ground-truth {train_trials2}-shot',
              f'Corrupted Ground-truth {train_trials2}-shot'
              ]
    fmt = '.2f'
    for i,mod in  enumerate([groundtruth,groundtruth_model,model,groundtruth_model2,model2]):
        # print(normalised_score(mod,(test_data,[length]*test_trials)))

        score = normalised_cosmoothing_score2(mod, (test_data, [length] * test_trials))
        PR = compute_svd_hidden(mod,test_data[0],window_length=length)
        print(titles[i],'PR:{:.4f}, score:{:.4f}'.format(PR,score))
        # prob = model.predict_proba(test_data[0],lengths=[length] * test_trials)

        fontsize = 15
        ax_row = axs[0]
        ax_row[i].set_title(titles[i])
        # ax_row[i].imshow(mod.transmat_,vmin=0,vmax=1)
        # print(mod.transmat_)
        sns.heatmap(mod.transmat_,vmin=0,vmax=1,annot=True,fmt=fmt,ax=ax_row[i],cbar=False)
        ax_row[0].set_ylabel(r'$A$',fontsize=fontsize)

        ax_row = axs[1]
        # ax_row[i].imshow(mod.emissionprob_,vmin=0,vmax=1)
        sns.heatmap(mod.emissionprob_, vmin=0, vmax=1, annot=True, fmt=fmt, ax=ax_row[i], cbar=False)
        ax_row[0].set_ylabel(r'$B^{in}$',fontsize=fontsize)

        ax_row = axs[2]
        prob = mod.predict_proba(test_data[0,:length])
        # ax_row[i].imshow(prob.T,vmin=0,vmax=1)
        sns.heatmap(prob.T, vmin=0, vmax=1, annot=True, fmt=fmt, ax=ax_row[i], cbar=False,square=True)
        # print(mod.emissionprob_)
        ax_row[0].set_ylabel(r'$p(x_t|y_{1:T})$',fontsize=fontsize)

        ax_row = axs[3]
        # ax_row[i].imshow(mod.emissionprob2_,vmin=0,vmax=1)
        sns.heatmap(mod.emissionprob2_, vmin=0, vmax=1, annot=True, fmt=fmt, ax=ax_row[i], cbar=False,square=True)
        ax_row[0].set_ylabel(r'$\hat B^{out}$',fontsize=fontsize)
    fig.tight_layout()
    fig.savefig('plots/test_plots/three_models_emission.png',dpi=200)
    fig.savefig('plots/test_plots/three_models_emission.pdf')

@hydra.main(version_base=None, config_path=CONFIG_PATH, config_name=CONFIG_NAME)
def main7(cfg):
    length = 10
    train_trials = 5
    train_trials2 = 50
    test_trials = 500

    eps = 1e-1

    groundtruth = hydra.utils.instantiate(cfg.groundtruth5)
    big_model = hydra.utils.instantiate(cfg.groundtruth10)

    big_model.emissionprob_ = np.concatenate([np.eye(5),np.eye(5)],axis=1).T + eps
    big_model.emissionprob_ /= big_model.emissionprob_.sum(1, keepdims=True)
    big_model.emissionprob2_ = big_model.emissionprob_
    big_model.transmat_ += 1e-2
    big_model.transmat_ /= big_model.transmat_.sum(1, keepdims=True)

    big_model2 = deepcopy( big_model )


    n = groundtruth.transmat_.shape[0]
    print(groundtruth.emissionprob_.shape)


    groundtruth.transmat_ = np.roll(np.eye(n),1,axis=1) + 1e-2
    groundtruth.transmat_ /= groundtruth.transmat_.sum(1, keepdims=True)
    groundtruth.emissionprob_ = np.eye(n) + eps# * np.eye(n)[::-1]
    groundtruth.emissionprob_ /= groundtruth.emissionprob_.sum(1,keepdims = True)

    groundtruth.startprob_ = np.ones(n)
    groundtruth.startprob_ /= groundtruth.startprob_.sum()
    # groundtruth.emissionprob_ = np.roll(groundtruth.emissionprob_,2,axis=0)
    groundtruth.emissionprob2_ = groundtruth.emissionprob_[::-1]
    groundtruth.coemissionprob_ = np.stack([groundtruth.emissionprob_, groundtruth.emissionprob2_])

    model = deepcopy(groundtruth)

    groundtruth_model = deepcopy(groundtruth)



    eps = 0
    model.emissionprob_[:, -2:] = np.roll(model.emissionprob_[:, -2:], 1, axis=1)
    # model.transmat_[-2:] = np.roll(model.transmat_[-2:], 1, axis=0)
    # model.transmat_ += eps * np.eye(n)[::-1]
    # model.transmat_ /= model.transmat_.sum(1, keepdims=True)

    model2 = deepcopy(model)
    groundtruth_model2 = deepcopy(groundtruth_model)

    train_data = cosample_hmm(groundtruth,length = length, trials = train_trials, seed_base=0)
    train_data2 = cosample_hmm(groundtruth, length=length, trials=train_trials2, seed_base=87643)
    test_data  = cosample_hmm(groundtruth,length = length, trials = test_trials, seed_base=23454)



    for mod in [groundtruth_model,model,big_model]:
        # mod.params = 'e'
        # mod.fit(train_data,lengths=[length]*train_trials)
        # print(train_data[0].shape,train_data[1].shape,length,train_trials)
        fit_model_emission(mod, train_data[0], train_data[1], lengths=[length] * train_trials)
        mod.coemissionprob_ = np.stack([mod.emissionprob_, mod.emissionprob2_])

    for mod in [groundtruth_model2,model2,big_model2]:
        # mod.params = 'e'
        # mod.fit(train_data,lengths=[length]*train_trials)
        # print(train_data2[0].shape,train_data2[1].shape,length,train_trials)
        fit_model_emission(mod, train_data2[0], train_data2[1], lengths=[length] * train_trials2)
        mod.coemissionprob_ = np.stack([mod.emissionprob_, mod.emissionprob2_])


    titles = ['Ground-truth',
              f'Ground-truth {train_trials}-shot',
              f'Corrupted Ground-truth {train_trials}-shot',
              f'Ground-truth {train_trials2}-shot',
              f'Corrupted Ground-truth {train_trials2}-shot',
              f'big model {train_trials}-shot',
              f'big model {train_trials2}-shot']
    fmt = '.2f'
    all_models = [groundtruth,groundtruth_model,model,groundtruth_model2,model2,big_model,big_model2]
    fig, axs = plt.subplots(4, len(all_models), figsize=(len(all_models)/5*18, 12))
    for i,mod in  enumerate(all_models):
        # print(normalised_score(mod,(test_data,[length]*test_trials)))

        score = normalised_cosmoothing_score2(mod, (test_data, [length] * test_trials))
        PR = compute_svd_hidden(mod,test_data[0],window_length=length)
        print(titles[i],'PR:{:.4f}, score:{:.4f}'.format(PR,score))
        # prob = model.predict_proba(test_data[0],lengths=[length] * test_trials)

        fontsize = 15
        ax_row = axs[0]
        ax_row[i].set_title(titles[i])
        # ax_row[i].imshow(mod.transmat_,vmin=0,vmax=1)
        # print(mod.transmat_)
        sns.heatmap(mod.transmat_,vmin=0,vmax=1,annot=True,fmt=fmt,ax=ax_row[i],cbar=False)
        ax_row[0].set_ylabel(r'$A$',fontsize=fontsize)

        ax_row = axs[1]
        # ax_row[i].imshow(mod.emissionprob_,vmin=0,vmax=1)
        sns.heatmap(mod.emissionprob_, vmin=0, vmax=1, annot=True, fmt=fmt, ax=ax_row[i], cbar=False)
        ax_row[0].set_ylabel(r'$B^{in}$',fontsize=fontsize)

        ax_row = axs[2]
        prob = mod.predict_proba(test_data[0,:length])
        # ax_row[i].imshow(prob.T,vmin=0,vmax=1)
        sns.heatmap(prob.T, vmin=0, vmax=1, annot=True, fmt=fmt, ax=ax_row[i], cbar=False,square=True)
        # print(mod.emissionprob_)
        ax_row[0].set_ylabel(r'$p(x_t|y_{1:T})$',fontsize=fontsize)

        ax_row = axs[3]
        # ax_row[i].imshow(mod.emissionprob2_,vmin=0,vmax=1)
        sns.heatmap(mod.emissionprob2_, vmin=0, vmax=1, annot=True, fmt=fmt, ax=ax_row[i], cbar=False,square=True)
        ax_row[0].set_ylabel(r'$\hat B^{out}$',fontsize=fontsize)
    fig.tight_layout()
    fig.savefig('plots/test_plots/three_models_emission.png',dpi=200)
    fig.savefig('plots/test_plots/three_models_emission.pdf')


def test_main():
    n = 5
    idx = np.random.permutation(n)
    P = np.eye(n)[idx,:]
    D = np.diag(np.arange(n))

    print((np.ones(n)@P@D),(P@D@(np.ones(n))))

@hydra.main(version_base=None, config_path=CONFIG_PATH, config_name=CONFIG_NAME)
def test_main2(cfg):
    length = 10
    train_trials = 50
    test_trials = 5

    groundtruth = hydra.utils.instantiate(cfg.groundtruth5)
    groundtruth.emissionprob2_ = groundtruth.emissionprob_[::-1]
    groundtruth.coemissionprob_ = np.stack([groundtruth.emissionprob_,groundtruth.emissionprob2_])
    cosample_hmm(groundtruth,length = length, trials = test_trials, seed_base=23454)


# def split_and_test(model,n_heldin=):


def test_main3():
    model = PoissonHMM(n_components=2)
    model.fit(np.random.choice(3,size=(16,10)))
    print(model.transmat_.shape)
    # print(model.emissionprob_)
    # print(model.emissionprob_.shape)
    print(model.lambdas_.shape)
    model.lambdas_ = np.random.exponential(3,size=model.lambdas_.shape)
    model.predict_proba(np.random.choice(3, size=(16, 10)))

    model.lambdas_ = model.lambdas_[...,:5]
    model.n_features = model.lambdas_.shape[-1]

    print(
        model.predict_proba(np.random.choice(3, size=(16, 5)))
    )
    print(
        model.score(np.random.choice(3, size=(16, 5)))
    )
    print(
        model.predict(np.random.choice(3, size=(16, 5)))
    )


if __name__ == '__main__':
    # main3_plot()
    # main5()
    # main6()
    # main2()
    # test_main()
    # test_main2()
    main7()
    # test_main3()