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
from copy import deepcopy
import matplotlib as mpl
mpl.rcParams['text.usetex'] = True

from metrics import normalised_score
#mpl.rcParams['text.latex.preamble'] = [r'\usepackage{amsmath}']

CONFIG_PATH = "configs"
# CONFIG_NAME = "config"
CONFIG_NAME = "toy_example_config"

OmegaConf.register_new_resolver("eval", eval)
OmegaConf.register_new_resolver("ind", lambda a,i: a[i])
OmegaConf.register_new_resolver("listmul", lambda l, i: [l]*i)
OmegaConf.register_new_resolver("getattr", getattr)


# @hydra.main(version_base=None, config_path=CONFIG_PATH, config_name=CONFIG_NAME)
def main(cfg):
    # OmegaConf.resolve(cfg)

    length       = cfg.length
    train_trials = cfg.train_trials
    val_trials   = cfg.val_trials
    test_trials  = cfg.test_trials


    train1, val1, test1 = hydra.utils.instantiate(cfg.all_data_tuples1)
    train2, val2, test2 = hydra.utils.instantiate(cfg.all_data_tuples2)

    groundtruth1 = hydra.utils.instantiate(cfg.groundtruth1)
    groundtruth2 = hydra.utils.instantiate(cfg.groundtruth2)
    model = hydra.utils.instantiate(cfg.model)
    model.emissionprob_ = np.stack([groundtruth1.emissionprob_[0]]*length)
    self_consistent_model = deepcopy(groundtruth1)
    self_consistent_model_large = deepcopy(model)
    model_before_finetuning = deepcopy(model)

    size = len(self_consistent_model_large.startprob_)
    self_consistent_model_large.startprob_ = np.ones(size)/size
    self_consistent_model_large.transmat_ = np.ones((size,size)) / size
    self_consistent_model_large.emissionprob_ = np.stack([groundtruth1.emissionprob_[0]]*size)

    self_consistent_model_large_before_finetuning = deepcopy(self_consistent_model_large)

    #hid_prob = model.predict_proba(val1[:cfg.length])
    #plt.imshow(hid_prob)
    #plt.show()
    #print(train2)
    for m in [model,self_consistent_model,self_consistent_model_large]:
        m.params = 'e'
        #print(dir(m))
        m.iter=1
        #print(m.shift_limits,m.shift_sampling_window)
        m.fit(train2,lengths = [length]*train_trials)
        m.emissionprob_[m.emissionprob_ == 0] = 1e-3
        m.emissionprob_[m.emissionprob_ == 1] = 1-1e-3
    # self_consistent_model.params = 'e'
    # self_consistent_model.fit(train2,lengths = [length]*train_trials)

    #print(groundtruth2.emissionprob_)
    #print(self_consistent_model.emissionprob_)

    #print(model.startprob_,model.transmat_,model.emissionprob_)


    # test1_score_model_before_finetuning = normalised_score(
    #     model_before_finetuning,
    #     (test1, [length] * (test1.shape[0] // length)),
    # )
    #
    # test1_score_GT1 = normalised_score(
    #     groundtruth1,
    #     (test1, [length] * (test1.shape[0] // length)),
    # )
    #
    # test1_score_SC_large_before_finetuning = normalised_score(
    #     self_consistent_model_large_before_finetuning,
    #     (test1, [length] * (test1.shape[0] // length)),
    # )


    test2_score_groundtruth2 = normalised_score(
        groundtruth2,
        (test2, [length] * (test2.shape[0] // length)),
    )


    test2_score_model = normalised_score(
        model,
        (test2, [length] * (test2.shape[0] // length)),
    )

    test2_score_self_consistent_model = normalised_score(
        self_consistent_model,
        (test2, [length] * (test2.shape[0] // length)),
    )

    test2_score_SC_large = normalised_score(
        self_consistent_model_large,
        (test2, [length] * (test2.shape[0] // length)),
    )

    print(model.emissionprob_[:,0].shape)
    emission_prob_var = np.std(model.emissionprob_[:,0])
    print(emission_prob_var**2,groundtruth2.emissionprob_[0,0]*(1-groundtruth2.emissionprob_[0,0])/cfg.train_trials)
    print(model.emissionprob_[:,0].mean(),groundtruth2.emissionprob_[0,0])

    # print('groundtruth1 on test1:',test1_score_GT1)
    # print('model before finetuning on test1:',test1_score_model_before_finetuning)
    # print('large model before finetuning on test1:', test1_score_SC_large_before_finetuning)
    #
    # print('groundtruth 2',test2_score_groundtruth2)
    # print('model test2:',test2_score_model)
    # print('self_consistent_model test2:',test2_score_self_consistent_model)
    # print('large self consistent model test2:', test2_score_SC_large)
    #
    # print('diff GT-model',test2_score_groundtruth2 - test2_score_model, 1/train_trials * (groundtruth2.emissionprob_[0,1]))
    # print('diff GT-consistent model', test2_score_groundtruth2 - test2_score_self_consistent_model, 1/(train_trials*length) * (groundtruth2.emissionprob_[0,0])* (groundtruth2.emissionprob_[0,1]))

    return test2_score_groundtruth2 - test2_score_model, test2_score_groundtruth2 - test2_score_self_consistent_model, test2_score_groundtruth2 - test2_score_SC_large
@hydra.main(version_base=None, config_path=CONFIG_PATH, config_name=CONFIG_NAME)
def main_repeat(cfg):
    train_trial_range = np.arange(1,27,3)[:]
    repeats = 10
    model_diff_all = []
    SC_model_diff_all = []
    SC_large_model_diff_all = []
    for i,train_trials in enumerate(train_trial_range[:]):
        model_diff_reps = []
        SC_model_diff_reps = []
        SC_large_model_diff_reps = []
        for rep in range(repeats)[:]:
            #print(type(cfg.train_trials),cfg.train_trials)
            setattr(cfg,'train_trials',int(train_trials))
            setattr(cfg.all_data1, 'seed_base', i*repeats+100+rep)
            setattr(cfg.all_data2, 'seed_base', i*repeats+10001+rep)
            print(type(cfg.train_trials),cfg.train_trials)
            model_diff,SC_model_diff,SC_large_model_diff = main(cfg)
            print('scores',model_diff,SC_model_diff,SC_large_model_diff)
            model_diff_reps.append(model_diff)
            SC_model_diff_reps.append(SC_model_diff)
            SC_large_model_diff_reps.append(SC_large_model_diff)

        model_diff_all.append(np.array(model_diff_reps))
        SC_model_diff_all.append(np.array(SC_model_diff_reps))
        SC_large_model_diff_all.append(np.array(SC_large_model_diff_reps))
    plt.scatter(train_trial_range, np.nanmean(np.stack(model_diff_all),1),c='C1',label='Chain')
    plt.scatter(train_trial_range, np.nanmean(np.stack(SC_model_diff_all),1),c='C0',label='Ground truth')
    plt.scatter(train_trial_range, np.nanmean(np.stack(SC_large_model_diff_all), 1), c='C3', label='Uniform',s=10)
    plt.plot(train_trial_range,1 / train_trial_range,ls='dashed',c='C1',label=r'$\frac{1}{K}$')
    plt.plot(train_trial_range, 1 / (train_trial_range*cfg.length), ls='dashed', c='C0',label=r'$\frac{1}{KT}$')
    # plt.yscale('log')
    plt.ylabel('Average K-shot generalisation error')
    plt.xlabel(rf'$K$, ($T={cfg.length}$)')
    plt.legend()
    plt.xticks(np.arange(0,30,5))
    plt.ylim(0)
    plt.xlim(0)
    plt.savefig('plots/fewshot_generalisation.pdf')
    plt.show()

if __name__ == '__main__':
    main_repeat()