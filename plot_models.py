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
import matplotlib as mpl
mpl.rcParams['text.usetex'] = True
#mpl.rcParams['text.latex.preamble'] = [r'\usepackage{amsmath}']

CONFIG_PATH = "configs"
# CONFIG_NAME = "config"
CONFIG_NAME = "config_OOD_protocol"

OmegaConf.register_new_resolver("eval", eval)
OmegaConf.register_new_resolver("ind", lambda a,i: a[i])
OmegaConf.register_new_resolver("listmul", lambda l, i: [l]*i)
OmegaConf.register_new_resolver("getattr", getattr)


@hydra.main(version_base=None, config_path=CONFIG_PATH, config_name=CONFIG_NAME)
def main(cfg):
    OmegaConf.resolve(cfg)

    train, val, test = hydra.utils.instantiate(cfg.all_data_tuples)

    if cfg.run_analysis:
        if cfg.use_groundtruth_as_model:
            model = hydra.utils.instantiate(cfg.groundtruth)
            data = {'model_name': 'Groundtruth',
                    'model_id': None}
            results_path = cfg.groundtruth_savepath + 'groundtruth'
        else:
            model_save_path = cfg.model_save_path
            with open(model_save_path,'rb') as f:
                model = pkl.load(f)
            data = {'model_name': cfg.model_name,
                    'model_id': cfg.model_index}

            results_path = cfg.model_save_path

    #print(model)

    ##### Plot eigenspectrum  ####
    lam = np.linalg.eigvals(model.transmat_)
    fig,ax = plt.subplots()
    ax.scatter(lam.real,lam.imag)
    ax.set_title(data['model_name']+' ')

    fig.savefig(results_path+'eigvals.png')
    plt.close()

    ##### Plot stationary distribution  ####
    dist = model.get_stationary_distribution()
    fig, ax = plt.subplots()
    ax.plot(np.sort(dist))
    ax.set_title(data['model_name'] + ' ')
    fig.savefig(results_path + 'steady_state.png')
    plt.close()

    ##### scatter stationary distribution and pi  ####
    dist = model.get_stationary_distribution()
    pi = model.startprob_
    fig, ax = plt.subplots()
    ax.scatter(dist,pi)
    ax.plot([0,1],[0,1],ls='dashed',c='black')
    ax.set_title(data['model_name'] + ' ')
    fig.savefig(results_path + 'steady_state_pi_scatter.png')
    plt.close()

    ##### Emission distribution  ####
    B = model.emissionprob_

    fig, ax = plt.subplots()
    ax.plot(np.sort(B[:,0]))
    #ax.plot([0,1],[0,1],ls='dashed',c='black')
    ax.set_title(data['model_name'] + ' ')
    fig.savefig(results_path + '_emission_distribution.png')
    plt.close()

    ##### predicted probability  ####
    hid_predicted = model.predict_proba(test[3*cfg.length:(3+1)*cfg.length])
    fig, ax = plt.subplots()
    im = ax.imshow(hid_predicted.T,vmin=0,vmax=1,interpolation='nearest',aspect='equal')
    fig.colorbar(im,ax=ax,shrink=0.5)
    ax.set_xlabel('time')
    ax.set_ylabel('hidden state')
    ax.set_xticks([])
    ax.set_yticks([])
    fig.tight_layout()
    fig.savefig(results_path + '_hid_predicted.png')
    plt.close()

    ##### mutual info  ####
    window_length = cfg.length
    GT = hydra.utils.instantiate(cfg.groundtruth)
    hid_predicted1 = [GT.predict_proba(test[i:i+window_length]) for i in range(test.shape[0]//window_length)]
    hid_predicted1 = np.concatenate(hid_predicted1)
    hid_predicted2 = [model.predict_proba(test[i:i+window_length]) for i in range(test.shape[0]//window_length)]
    hid_predicted2 = np.concatenate(hid_predicted2)

    joint = (hid_predicted1[:,None]*hid_predicted2[:,:,None]).mean(0).T
    p1 = hid_predicted1.mean(0)
    p2 = hid_predicted2.mean(0)
    vmax = np.concatenate([joint,np.outer(p1,p2)]).max()
    fig, axs = plt.subplots(1,2)
    ax = axs[0]
    im = ax.imshow(joint,vmin=0,vmax=vmax,interpolation='nearest',aspect='equal')
    ax = axs[1]
    im = ax.imshow(np.outer(p1,p2), vmin=0, vmax=vmax, interpolation='nearest', aspect='equal')
    #fig.colorbar(im,ax=ax,shrink=0.5)
    #ax.set_xlabel('time')
    #ax.set_ylabel('hidden state')
    ax.set_title(data['model_name'] + ' ')
    ax.set_xticks([])
    ax.set_yticks([])
    fig.tight_layout()
    fig.savefig(results_path + '_MI_probs.png')
    plt.close()

if __name__ == '__main__':
    main()