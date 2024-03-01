import matplotlib.pyplot as plt
from omegaconf import DictConfig, OmegaConf
import hydra
import os
import pandas as pd
import pickle as pkl
from jax import numpy as jnp
from hmmlearn_dynamaxhmm_converter import hmmlearn_to_dynamaxhmm, dynamaxhmm_to_hmmlearn
import similarity
import numpy as np
from itertools import product
from tqdm import tqdm
# from hydra.utils import instantiate
from config_utils import instantiate
from main import jenson_shannon_divergence
from prepare_model import CoHMM,split_model_emission,fuse_model
from copy import deepcopy
from utils import flatten_with_lengths,HMM_Dataset, normalise, setattrs, setattrs_kwargs, make_path_if_not_exist, lfads_torch_datamodule_to_numpy
from metrics import bernoulli_bits_per_spike
from hmm_adapter import CoHMM3d as CoHMM
from main_cohmm import omegaconf_resolvers

import matplotlib as mpl
mpl.rcParams['text.usetex'] = True
plt.rcParams["font.family"] = "serif"
plt.rcParams["mathtext.fontset"] = "dejavuserif"
mpl.rcParams['text.latex.preamble'] = r'\usepackage{amsmath}'


CONFIG_PATH = "configs"
# CONFIG_NAME = "config"
CONFIG_NAME = "config_cohmm"

path_to_models = '/Users/kabir/Documents/code/hmm_analysis/all_models_validated_v2/teacher_state4_bernoulli_partial_eps0.01_length10/models_traintrials2000'
dataframe_file_name = 'latents_dataframe.pkl'

@hydra.main(version_base='1.3', config_path=CONFIG_PATH, config_name=CONFIG_NAME)
def load_models_and_store_latents(cfg):
    omegaconf_resolvers()

    all_files = os.listdir(path_to_models)

    model_files = [f for f in all_files if f[-4]!='.']
    # model_files = model_files[:4]


    models = []
    for f in model_files:
        full_path = os.path.join(path_to_models,f)
        with open(full_path,'rb') as _f:
            models.append(pkl.load(_f))

    print('Loaded',len(models),'models.')
    filtered_models = [m for m in models if not np.any(np.isnan(m.transmat_))]
    print('Removed', len(models)-len(filtered_models), 'models with NaN params.')
    models = filtered_models

    ### generating data
    instantiate(cfg.numpy_seed)
    if cfg.data_mode == 'student-teacher':
        teacher = instantiate(cfg.teacher)
        data = instantiate(cfg.generate_all_data_dictmodule,_convert_='partial')(hmm_model=teacher)
    else:
        datamodule = instantiate(cfg.datamodule,_convert_="all")
        data_numpy = lfads_torch_datamodule_to_numpy(datamodule)[:,:35,:].astype(int)
        # data_numpy[data_numpy>=1] = 1
        print(data_numpy.shape)
        data = instantiate(cfg.numpy_to_xarray_with_breakdownlabels,_convert_='partial')(data=data_numpy)

    all_model_data = []
    for model in tqdm(models):
        model_data = {}


        ### co-smoothing
        student = model
        bits_per_spike = instantiate(cfg.bits_per_spike_func)
        test_student = deepcopy(student)
        (
            test_student_in,
            test_student_out
        ) = split_model_emission(
            test_student,
            split_indices=instantiate(cfg.neurons_split_indices)[:1]
        )
        split_student = CoHMM(test_student_in, test_student_out)
        # print('transmat sum',split_student.encoder.transmat_.sum(-1))
        test_pred_out = split_student.predict(data.select(**cfg.breakups.cosmoothing.input), mode3d=True)
        test_pred_out = test_pred_out.reshape(*data.select(**cfg.breakups.cosmoothing.target).shape)
        test_pred_out[np.isnan(test_pred_out)] = 0
        co_bps = bits_per_spike(test_pred_out, data.select(**cfg.breakups.cosmoothing.target).to_numpy())
        model_data['co_bps'] = co_bps



        #### get latents for cross-decoding
        train_input_data = data.select(**cfg.breakups.decoding_full.fit.input).values
        test_input_data = data.select(**cfg.breakups.decoding_full.test.input).values
        student_inout = split_model_emission(student, split_indices=instantiate(cfg.neurons_split_indices)[1:2])[0]
        train_latents = student_inout.predict_proba(train_input_data,mode3d=True)
        test_latents = student_inout.predict_proba(test_input_data, mode3d=True)
        model_data['train_latents'] = train_latents
        model_data['test_latents'] = test_latents

        all_model_data.append(model_data)

    D = pd.DataFrame(all_model_data)
    saveloc = os.path.join(path_to_models,dataframe_file_name)
    with open(saveloc,'wb') as f:
        pkl.dump(D,f)


@hydra.main(version_base='1.3', config_path=CONFIG_PATH, config_name=CONFIG_NAME)
def cross_decoding(cfg):
    omegaconf_resolvers()

    saveloc = os.path.join(path_to_models,dataframe_file_name)
    with open(saveloc,'rb') as f:
        latents_dataframe = pkl.load(f)

    best_latents_dataframe = latents_dataframe[latents_dataframe.co_bps>(latents_dataframe.co_bps.max()-3.5e-3)]

    # best_latents_dataframe = best_latents_dataframe.head(3)

    n_models = len(best_latents_dataframe)

    print('n_models:',n_models)
    scores = np.zeros((n_models,n_models))
    for i,j in product(range(n_models),range(n_models)):
        X = best_latents_dataframe.iloc[i]['train_latents']
        y = best_latents_dataframe.iloc[j]['train_latents']
        X,y = [thing.reshape(-1,thing.shape[-1]) for thing in [X,y]]
        print(cfg.decoding)
        if hasattr(cfg.decoding, 'preprocess_target'):
            y = instantiate(cfg.decoding.preprocess_target)(y)
        X = np.log(X)
        # from sklearn.linear_model import LinearRegression
        model = instantiate(cfg.decoding.regression_model)
        model.fit(
            X,
            y
        )

        X = best_latents_dataframe.iloc[i]['test_latents']
        y = best_latents_dataframe.iloc[j]['test_latents']
        X, y = [thing.reshape(-1, thing.shape[-1]) for thing in [X, y]]
        X = np.log(X)
        pred_y = getattr(model, cfg.decoding.predict_method)(X)

        metric = instantiate(cfg.decoding.metric)
        score = np.stack([metric(
            y[sample_id],
            pred_y[sample_id]
        ) for sample_id in range(pred_y.shape[0])]).mean()
        scores[i,j] = score

    saveloc = os.path.join(path_to_models,'cross_decoding_scores')
    np.save(saveloc,scores)

@hydra.main(version_base='1.3', config_path=CONFIG_PATH, config_name=CONFIG_NAME)
def plotting(cfg):
    omegaconf_resolvers()

    saveloc = os.path.join(path_to_models,dataframe_file_name)
    with open(saveloc,'rb') as f:
        latents_dataframe = pkl.load(f)

    fig,ax = plt.subplots()
    ax.hist(latents_dataframe['co_bps'],bins=30)
    fig.savefig(os.path.join(path_to_models,'co_bps_hist.png'))


    best_latents_dataframe = latents_dataframe[latents_dataframe.co_bps>(latents_dataframe.co_bps.max()-3.5e-3)]

    fig,axs = plt.subplots(2,3,sharex=True)
    for i,ax in enumerate(axs.flatten()):
        x = best_latents_dataframe['train_latents'].iloc[i][0].T
        # x = np.log10(x)
        ax.imshow(x)
        ax.set_yticklabels([])
        xticks = np.arange(0,x.shape[1],5)
        ax.set_xticks(xticks)
        ax.set_xticklabels(xticks)
    axs[1,0].set_xlabel('time',fontsize=8)
    axs[1, 0].set_ylabel('HMM states', fontsize=8)
    fig.tight_layout()
    fig.savefig(os.path.join(path_to_models,'latents_samples.png'))

def plot_cross_decoding_scores():
    saveloc = os.path.join(path_to_models, 'cross_decoding_scores.npy')
    scores = np.load(saveloc)
    print(scores.shape)

    fig,ax = plt.subplots()
    im = ax.imshow(scores)
    ax.set_ylabel('input model')
    ax.set_xlabel('target model')
    fig.colorbar(im,ax=ax)
    fig.savefig(os.path.join(path_to_models,'cross_decoding.png'))

    plt.close(fig)

    fig,ax = plt.subplots()
    ax.hist(np.log10(scores.flatten()),bins=30)
    fig.savefig(os.path.join(path_to_models,'scores_histogram.png'))


if __name__ == '__main__':
    # load_models_and_store_latents()
    # plotting()
    # cross_decoding()
    plot_cross_decoding_scores()