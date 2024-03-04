from omegaconf import DictConfig, OmegaConf
import hydra
import os
import pandas as pd
import pickle as pkl
from jax import numpy as jnp
from hmmlearn_dynamaxhmm_converter import hmmlearn_to_dynamaxhmm, dynamaxhmm_to_hmmlearn
# import similarity
import numpy as np
import h5py
# from hydra.utils import instantiate
from config_utils import instantiate
from main import jenson_shannon_divergence
from prepare_model import CoHMM, split_model_emission, fuse_model
from copy import deepcopy
from utils import flatten_with_lengths, HMM_Dataset, normalise, setattrs, setattrs_kwargs, make_path_if_not_exist, \
    lfads_torch_datamodule_to_numpy
from metrics import bernoulli_bits_per_spike
from hmm_adapter import CoHMM3d as CoHMM
from nlb_tools.make_tensors import make_eval_input_tensors, make_train_input_tensors, save_to_h5, \
    make_eval_target_tensors
from nlb_tools.evaluation import evaluate

import matplotlib.pyplot as plt
import matplotlib as mpl

mpl.rcParams['text.usetex'] = True
plt.rcParams["font.family"] = "serif"
plt.rcParams["mathtext.fontset"] = "dejavuserif"
mpl.rcParams['text.latex.preamble'] = r'\usepackage{amsmath}'

import prepare_model

CONFIG_PATH = "configs"
# CONFIG_NAME = "config"
CONFIG_NAME = "config_cohmm_mc_maze"


@hydra.main(version_base='1.3', config_path=CONFIG_PATH, config_name=CONFIG_NAME)
def decorated_main(cfg):
    return main(cfg)


def main(cfg):
    omegaconf_resolvers()
    print('starting')
    instantiate(cfg.numpy_seed)
    if cfg.data_mode == 'student-teacher':
        teacher = instantiate(cfg.teacher)
        data = instantiate(cfg.generate_all_data_dictmodule, _convert_='partial')(hmm_model=teacher)
    else:
        # datamodule = instantiate(cfg.datamodule,_convert_="all")
        # data_numpy = lfads_torch_datamodule_to_numpy(datamodule)[:,:35,:].astype(int)
        # data_numpy[data_numpy>=1] = 1
        dataset_name = cfg.load_dataset.dataset_name
        bin_size_ms = cfg.load_dataset.bin_size
        binsuf = '' if bin_size_ms == 5 else f'_{bin_size_ms}'
        phase = cfg.load_dataset.phase
        train_save_path = '_'.join(['train_dict', dataset_name, binsuf, phase]) + '.h5'
        eval_save_path = '_'.join(['eval_dict', dataset_name, binsuf, phase]) + '.h5'
        eval_target_save_path = '_'.join(['eval_target_dict', dataset_name, binsuf, phase]) + '.h5'
        train_save_path, eval_save_path, eval_target_save_path = [
            os.path.join(cfg.teacher_save_path, path) for path in
            [train_save_path, eval_save_path, eval_target_save_path]
        ]
        paths = [train_save_path, eval_save_path] + ([eval_target_save_path] if phase == 'val' else [])
        paths_exist = [os.path.exists(path) for path in paths]
        print(paths_exist)
        print('tensors exist',all(paths_exist))
        if not all(paths_exist):
            dataset = instantiate(cfg.load_dataset.dataset)

            if phase == 'val':
                train_split = 'train'
                eval_split = 'val'
            else:
                train_split = ['train', 'val']
                eval_split = 'test'

            # if hasattr(cfg,'preprocess'):
            #     for item in cfg.preprocess:
            #         getattr(dataset,item.method_name)(*item.args)
            dataset.resample(bin_size_ms)
            print(train_split,eval_split)
            train_dict = make_train_input_tensors(dataset, dataset_name, train_split, save_file=True,
                                                  save_path=train_save_path)
            eval_dict = make_eval_input_tensors(dataset, dataset_name, eval_split, save_file=True,
                                                save_path=eval_save_path)
            if phase == 'val':
                print('making target dict')
                target_dict = make_eval_target_tensors(dataset, dataset_name, train_split, eval_split, save_file=True,
                                                       include_psth=True, save_path=eval_target_save_path)
        else:
            def load_h5_to_dict(path):
                D = {}
                with h5py.File(path, "r") as f:
                    print(f.keys())
                    for key in f.keys():
                        D[key] = f[key][()]
                return D

            train_dict = load_h5_to_dict(train_save_path)
            eval_dict = load_h5_to_dict(eval_save_path)
            if phase == 'val':
                target_dict = load_h5_to_dict(eval_target_save_path)

        train_spikes_heldin = train_dict['train_spikes_heldin'].astype(int)
        train_spikes_heldout = train_dict['train_spikes_heldout'].astype(int)
        print(train_spikes_heldin.shape, train_spikes_heldout.shape)
        data_numpy = np.concatenate([train_spikes_heldin, train_spikes_heldout], axis=-1)
        # print(data_numpy.shape)
        data = instantiate(cfg.numpy_to_xarray_with_breakdownlabels, _convert_='partial')(data=data_numpy)

    # print(data)

    np.random.seed(cfg.student_index)
    student = instantiate(cfg.student)

    if cfg.initialise_student_as_teacher:
        # student = hydra.utils.instantiate(teacher)
        (
            teacher_in,
            teacher_out,
            teacher_reallyheldout
        ) = split_model_emission(deepcopy(teacher), split_indices=instantiate(cfg.neurons_split_indices))
        split_teacher = CoHMM(teacher_in, teacher_out)
        teacher_inout = fuse_model(split_teacher)
        student = teacher_inout
        cfg.student.n_components = teacher.n_components
        print('initial manipulation', cfg.initial_manipulation)
        cfg.student_save_path = os.path.join(
            os.path.dirname(cfg.student_save_path),
            '_'.join(['init', 'groundtruth'])
        )

        if cfg.initial_manipulation:
            manipulation = instantiate(cfg.initial_manipulation)
            student = manipulation(student)
            cfg.student_save_path += manipulation.name

    if cfg.training_framework == 'dynamax':
        cfg.student_save_path += '_dynamax_' + cfg.dynamax.algorithm
        cfg.student_name = 'dynamax_' + cfg.dynamax.algorithm

    if cfg.run_train:
        if cfg.training_framework == 'hmmlearn':
            fit_args = {
                'X': data.select(**cfg.breakups.fit.train),
                'X_val': data.select(**cfg.breakups.fit.val),
                'mode3d': True
            }

            print(student)
            if hasattr(cfg, 'training_config'):
                for stage in instantiate(cfg.training_config):
                    setattrs_kwargs(student, **stage)
                    student.fit(**fit_args)
            else:
                student.fit(**fit_args)
            cfg.student_name = 'hmmlearn_fit_em'
            if cfg.initialise_student_as_teacher:
                cfg.student_save_path += f'_niter{student.monitor_.iter}_learned' + student.params
            # print('transmat sum just after training', student.transmat_.sum(-1))

        elif cfg.training_framework == 'dynamax':
            # print(data[0].shape,'here')
            train_data = data.select(**cfg.breakups.fit.train)

            # student._init(train_data[0])
            # print(student.lambdas_.shape,student.transmat_.shape,stu)
            # print(student.n_features,student.n_features)
            print('before dynamax')
            jnp.asarray([1, 2, 3])
            print('after jax line 1')
            student_dynamax, student_params, student_params_prop = hmmlearn_to_dynamaxhmm(student)
            print('after hmmlearn to dynamax')
            if cfg.initialise_dynamax:
                mean_val = np.asarray(train_data).mean()
                print('mean_val', mean_val)
                scale = len(np.asarray(train_data).flatten()) / student.n_components / 1000
                student_dynamax.emission_component.emission_prior_concentration0 = (1 - mean_val) * scale
                student_dynamax.emission_component.emission_prior_concentration1 = mean_val * scale
                student_dynamax.emission_component.emission_prior_concentration = mean_val
                student_dynamax.emission_component.emission_prior_rate = 1
                # student_dynamax.emission_prior_concentration=0.1
                # student_dynamax.emission_prior_rate=1.0
                student_params, student_params_prop = student_dynamax.initialize(
                    initial_probs=jnp.asarray(student.startprob_),
                    transition_matrix=jnp.asarray(student.transmat_),
                )
                # fig,ax = plt.subplots()
                # ax.hist(student_params.emissions.rates.flatten(),bins=30)
                #
                # plt.savefig('plots/test_plots/dynamax_initial_emission_prob.png')
                # plt.close(fig)
            # print('before training',student_params)

            # print(np.asarray(train_data))
            all_losses = []
            if hasattr(cfg.dynamax, 'shape_curriculum'):
                new_student_params = student_params
                for stage in cfg.dynamax.shape_curriculum:
                    windows = np.arange(0, train_data.shape[1] + 1, stage.window_length)
                    windows = np.stack([windows[:-1], windows[1:]], axis=1)
                    reshaped_train_data = np.concatenate([train_data[:, s:e, :] for (s, e) in windows])
                    print('reshaped train data shape', reshaped_train_data.shape)

                    new_student_params, losses = getattr(student_dynamax, cfg.dynamax.algorithm)(
                        new_student_params,
                        student_params_prop,
                        jnp.asarray(reshaped_train_data),
                        **instantiate(cfg.dynamax.fit_kwargs)
                    )
                    all_losses.append(losses)
                losses = np.concatenate(all_losses)
            else:
                print('here,')
                new_student_params, losses = getattr(student_dynamax, cfg.dynamax.algorithm)(
                    student_params,
                    student_params_prop,
                    jnp.asarray(train_data),
                    **instantiate(cfg.dynamax.fit_kwargs)
                )

            # fig,ax = plt.subplots()
            # ax.plot(losses)
            # fig.savefig('plots/test_plots/losses_dynamax_adam.png')
            # plt.close(fig)
            print('after training', new_student_params)
            student = dynamaxhmm_to_hmmlearn(student_dynamax, new_student_params)

        else:
            raise Exception('cfg.training_framework must be one of "hmmlearn" or "dynamax".')

        if cfg.save_trained:
            student_save_path = cfg.student_save_path
            print('saving', student_save_path)
            if not os.path.exists(os.path.dirname(student_save_path)):
                os.makedirs(os.path.dirname(student_save_path))
            print('current working dir', os.getcwd())
            with open(student_save_path, 'wb') as f:
                print('inside open')
                pkl.dump(student, f)

    if cfg.run_analysis:
        if cfg.use_teacher_as_student:
            # student = hydra.utils.instantiate(teacher)
            (
                teacher_in,
                teacher_out,
                teacher_reallyheldout
            ) = split_model_emission(deepcopy(teacher), split_indices=instantiate(cfg.neurons_split_indices))
            split_teacher = CoHMM(teacher_in, teacher_out)
            teacher_inout = fuse_model(split_teacher)
            student = teacher_inout
            result_data = {
                'model_name': 'Groundtruth',
                'model_id': None,
                'n_components': teacher.n_components,
                'iterations': teacher.monitor_.iter,
                'train_trials': cfg.train_trials,
                'params_learned': teacher.params,
            }
            results_path = os.path.join(cfg.teacher_save_path, 'groundtruth')
        else:
            student_save_path = cfg.student_save_path
            result_data = {
                'model_name': cfg.student_name,
                'model_id': cfg.student_index,
                'n_components': student.n_components,
                'iterations': student.monitor_.iter,
                'train_trials': cfg.train_trials,
                'params_learned': student.params,
            }
            if cfg.load_student_for_analysis:
                print('loading', student_save_path)
                results_path = cfg.student_save_path
                try:
                    with open(student_save_path, 'rb') as f:
                        student = pkl.load(f)
                except:
                    print('could not load student, cancelling analysis.')
                    keys = cfg.analysis.keys()
                    for key in keys:
                        cfg.analysis[key] = False

        print('manipulation', cfg.manipulation)

        if cfg.manipulation:
            manipulation = instantiate(cfg.manipulation)
            student = manipulation(student)
            result_data['model_name'] = '_'.join([result_data['model_name'], manipulation.name])
            results_path = os.path.join(cfg.teacher_save_path, manipulation.name)

        bits_per_spike = instantiate(cfg.bits_per_spike_func)
        if cfg.analysis.plot_student_matrices:
            fig, ax = plt.subplots()
            ax.imshow(student.transmat_)
            fig.savefig('plots/test_plots/student_transmat.png')

            fig, ax = plt.subplots()
            ax.imshow(student.lambdas_)
            fig.savefig('plots/test_plots/student_lambdas.png')

            if hasattr(student, 'lambda_partiliser_mat_'):
                fig, ax = plt.subplots()
                ax.imshow(student.lambda_partiliser_mat_)
                ax.set_ylabel('ground truth')
                ax.set_xlabel('intermediate projection')
                fig.savefig('plots/test_plots/student_partial.png')

        if cfg.analysis.compute_similarity_metrics:
            test_student = deepcopy(student)
            (
                student_encoder,
                _
            ) = split_model_emission(
                test_student,
                split_indices=instantiate(cfg.neurons_split_indices)[:1]
            )
            (
                teacher_encoder,
                _
            ) = split_model_emission(
                teacher,
                split_indices=instantiate(cfg.neurons_split_indices)[:1]
            )

            student_latent = student_encoder.predict_proba(data.select(**cfg.breakups.cosmoothing.input), mode3d=True)
            teacher_latent = teacher_encoder.predict_proba(data.select(**cfg.breakups.cosmoothing.input), mode3d=True)

            for name, measure in instantiate(cfg.similarity_measures).items():
                result_data['similarity.' + name] = measure(student_latent, teacher_latent)

        if cfg.analysis.compute_co_smoothing:
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
            result_data['original co-smoothing'] = co_bps
            print('original co-smoothing', co_bps)

        if cfg.analysis.compute_mutual_info:
            print(teacher.n_features, student.n_features)
            generate_all_data = instantiate(cfg.generate_all_data)
            teacher_ = split_model_emission(teacher, split_indices=instantiate(cfg.neurons_split_indices)[1:2])[0]
            student_ = split_model_emission(student, split_indices=instantiate(cfg.neurons_split_indices)[1:2])[0]
            print('n_features', teacher_.n_features, student_.n_features)
            print(instantiate(cfg.neurons_split_indices))
            for model1, model2, result_name in [
                                                   (teacher_, student_, 'MI_teacher->student'),
                                                   (student_, teacher_, 'MI_student->teacher')
                                               ][:]:
                all_data = instantiate(cfg.generate_all_data_with_states_dictmodule, _convert_='partial')(
                    hmm_model=model1)
                obs, states = all_data['data_xarray'], all_data['states_data_xarray']

                test_model2_in, _ = split_model_emission(model2,
                                                         split_indices=instantiate(cfg.neurons_split_indices)[0:1])
                # proba1 = model1.predict_proba(obs.select(**cfg.breakups.decoding.fit.input),mode3d=True)
                select_obs = obs.select(**cfg.breakups.decoding.fit.input)
                select_states = states.select(**cfg.breakups.decoding.fit.states)
                proba = test_model2_in.predict_proba(select_obs, mode3d=True)
                # def mutual_info(GT_states,inferred_proba,state_values):

                proba = proba.reshape(*select_obs.shape[:2], proba.shape[-1])
                print('proba shape', proba.shape)
                time_steps = obs.shape[1]
                T = np.ones((time_steps, model2.n_components, model1.n_components))
                prob_xt_is_j = np.zeros((time_steps, model1.n_components))
                for t in range(time_steps):
                    for j in range(model1.n_components):
                        xt_is_j = (select_states[:, t] == j)
                        if np.any(np.squeeze(xt_is_j)):
                            T[t, :, j] = proba[np.squeeze(xt_is_j), t, :].mean(axis=0)
                        prob_xt_is_j[t] = xt_is_j.astype(float).mean()

                joint_prob = (T * prob_xt_is_j[:, None]).mean(0)

                eps = 1e-10
                # plt.figure()
                # plt.imshow(T[0])
                # plt.savefig(f'plots/test_plots/{result_name}T_for_MI.png')
                MI = (
                        joint_prob * np.log(
                    (joint_prob + eps) / (
                            (joint_prob.sum(1, keepdims=True) + eps) * (joint_prob.sum(0, keepdims=True) + eps)
                    )
                )
                ).sum()
                result_data[result_name] = MI
                # print(result_name,MI)

        if cfg.analysis.compute_latent_decoding_stepwise:
            print(teacher.n_features, student.n_features)
            # generate_all_data = instantiate(cfg.generate_all_data)
            teacher_ = split_model_emission(teacher, split_indices=instantiate(cfg.neurons_split_indices)[1:2])[0]
            student_ = split_model_emission(student, split_indices=instantiate(cfg.neurons_split_indices)[1:2])[0]

            print('n_features', teacher_.n_features, student_.n_features)
            print(instantiate(cfg.neurons_split_indices))
            for model1, model2, result_name in [
                                                   (teacher_, student_, 'teacher->student'),
                                                   (student_, teacher_, 'student->teacher')
                                               ][:]:

                all_data = instantiate(cfg.generate_all_data_with_states_dictmodule, _convert_='partial')(
                    hmm_model=model1)
                obs, states = all_data['data_xarray'], all_data['states_data_xarray']

                test_model2_in, _ = split_model_emission(model2,
                                                         split_indices=instantiate(cfg.neurons_split_indices)[0:1])
                # proba1 = model1.predict_proba(obs.select(**cfg.breakups.decoding.fit.input),mode3d=True)
                proba = test_model2_in.predict_proba(obs.select(**cfg.breakups.decoding.fit.input), mode3d=True)

                T = np.ones((model2.n_components, model1.n_components, cfg.length))
                counts = np.zeros((model1.n_components, cfg.length))
                for j in range(model1.n_components):
                    # select_idx = np.reshape((states.select(**cfg.breakups.decoding.fit.states)==j).values,-1)
                    # print(proba.dtype,proba.shape)
                    # T[:,j] = proba[select_idx].sum(0)
                    state_is_j = (states.select(**cfg.breakups.decoding.fit.states) == j).values[..., 0]
                    # print(select_idx_per_t.shape,proba.shape,obs.select(**cfg.breakups.decoding.fit.input).shape)
                    # trials_where_state_is_j,t_where_state_is_j = np.where(state_is_j)
                    for t in range(cfg.length):
                        T[:, j, t] = proba[state_is_j[:, t], t, :].sum(0)
                        counts[j, t] += state_is_j[:, t].sum()
                T[np.isnan(T)] = 1
                T[np.isinf(T)] = 1
                T = normalise(T, axis=0)
                counts = normalise(counts, axis=0)

                eps = 1e-9

                D_KLs = (T[..., None] * (np.log(T[..., None] + eps) - np.log(T[..., None, :] + eps))).sum(
                    0)  # (model1.n_components,cfg.length,cfg.length)
                weights = (counts[:, None, :] * counts[:, :, None])  # (model1.n_components,cfg.length,cfg.length)
                weighted_D_KLs = (weights * D_KLs).sum(0)
                mean_D_KL = weighted_D_KLs.mean()
                print('consistency_' + result_name, mean_D_KL)
                result_data['consistency_' + result_name] = mean_D_KL
                #
                # fig,axs = plt.subplots(2,3,sharey=True,sharex=True)
                # for t in range(len(axs.flatten())):
                #     import scipy
                #     axs.flatten()[t].imshow(
                #         T[:,:,t]
                #     )
                #     # axs.flatten()[t].imshow(np.log(T[:, :, t]))
                # fig.savefig(f'plots/test_plots/T_{result_name}.png')
                # plt.close(fig)

                # fig, axs = plt.subplots(3, 5, sharey=True, sharex=True)
                # vmin = weighted_D_KLs.min()
                # vmax = weighted_D_KLs.max()
                # for i in range(len(axs.flatten())):
                #
                #     axs.flatten()[i].imshow(
                #         weighted_D_KLs,vmin=vmin,vmax=vmax,
                #     )
                #     # axs.flatten()[t].imshow(np.log(T[:, :, t]))
                # fig.savefig(f'plots/test_plots/D_KL_{result_name}.png')
                # plt.close(fig)

                # fig,ax = plt.subplots()
                # ax.plot(counts)
                # fig.savefig('plots/test_plots/counts.png')
                # plt.close(fig)

        if cfg.analysis.generate_submission:
            # dataset = instantiate(cfg.load_dataset.dataset)
            dataset_name = cfg.load_dataset.dataset_name
            bin_size_ms = cfg.load_dataset.bin_size
            phase = cfg.load_dataset.phase
            if phase == 'val':
                train_split = 'train'
                eval_split = 'val'
            else:
                train_split = ['train', 'val']
                eval_split = 'test'
            binsuf = '' if bin_size_ms == 5 else f'_{bin_size_ms}'
            # if hasattr(cfg,'preprocess'):
            #     for item in cfg.preprocess:
            #         # func = getattr(dataset,item.method_name)
            #         # dataset.resample(*item.args)
            #         # print(item)
            #         func(*item.args)
            # dataset.resample(20)
            # print(dataset)
            # eval_dict = make_eval_input_tensors(dataset, dataset_name, eval_split, save_file=False)
            # train_dict = make_train_input_tensors(dataset, dataset_name, train_split, save_file=False)
            train_spikes_heldin = train_dict['train_spikes_heldin'].astype(int)
            train_spikes_heldout = train_dict['train_spikes_heldout'].astype(int)
            eval_spikes_heldin = eval_dict['eval_spikes_heldin'].astype(int)
            print(train_spikes_heldin.shape, eval_spikes_heldin.shape)
            tlen = train_spikes_heldin.shape[1]

            test_student = deepcopy(student)
            (
                test_student_in,
                test_student_out
            ) = split_model_emission(
                test_student,
                split_indices=instantiate(cfg.neurons_split_indices)[:1]
            )
            split_student_full = CoHMM(test_student_in, student)
            train_rates_all = split_student_full.predict(train_spikes_heldin, mode3d=True)
            eval_rates_all = split_student_full.predict(eval_spikes_heldin, mode3d=True)
            print(train_rates_all.shape, eval_rates_all.shape)
            train_rates_all = train_rates_all.reshape(-1, tlen, train_rates_all.shape[-1])
            eval_rates_all = eval_rates_all.reshape(-1, tlen, eval_rates_all.shape[-1])
            print(train_rates_all.shape, eval_rates_all.shape)
            train_rates_heldin, train_rates_heldout = np.split(train_rates_all,
                                                               instantiate(cfg.neurons_split_indices)[0:1], axis=2)
            eval_rates_heldin, eval_rates_heldout = np.split(eval_rates_all,
                                                             instantiate(cfg.neurons_split_indices)[0:1], axis=2)

            # ---- Prepare/save output ---- #
            output_dict = {
                dataset_name + binsuf: {
                    'train_rates_heldin': train_rates_heldin,
                    'train_rates_heldout': train_rates_heldout,
                    'eval_rates_heldin': eval_rates_heldin,
                    'eval_rates_heldout': eval_rates_heldout,
                }
            }
            savepath = f'{dataset_name}{"" if bin_size_ms == 5 else f"_{bin_size_ms}"}_{phase}.h5'
            save_to_h5(output_dict, cfg.student_save_path + '_' + savepath, overwrite=True)
            if phase == 'val':
                print('local validation', evaluate(target_dict, output_dict))

        if cfg.analysis.compute_latent_decoding:
            print(teacher.n_features, student.n_features)
            # generate_all_data = instantiate(cfg.generate_all_data)
            teacher_ = split_model_emission(teacher, split_indices=instantiate(cfg.neurons_split_indices)[1:2])[0]
            student_ = split_model_emission(student, split_indices=instantiate(cfg.neurons_split_indices)[1:2])[0]
            print('n_features', teacher_.n_features, student_.n_features)
            print(instantiate(cfg.neurons_split_indices))
            for model1, model2, result_name in [
                                                   (teacher_, student_, 'teacher->student'),
                                                   (student_, teacher_, 'student->teacher')
                                               ][:]:
                # cfg.generate_all_data_dictmodule
                # obs,states = generate_all_data(model1,return_states=True)

                # obs = instantiate(cfg.numpy_to_xarray_with_breakdownlabels,_convert_='partial')(
                #     obs #[:,:,:]
                # )
                # # obs = obs.select(neurons_split=['heldin','heldout'])
                # states = instantiate(cfg.numpy_to_xarray_with_breakdownlabels_states,_convert_='partial')(
                #     states[...,0]
                # )
                all_data = instantiate(cfg.generate_all_data_with_states_dictmodule, _convert_='partial')(
                    hmm_model=model1)
                obs, states = all_data['data_xarray'], all_data['states_data_xarray']

                test_model2_in, _ = split_model_emission(model2,
                                                         split_indices=instantiate(cfg.neurons_split_indices)[0:1])
                # proba1 = model1.predict_proba(obs.select(**cfg.breakups.decoding.fit.input),mode3d=True)
                proba = test_model2_in.predict_proba(obs.select(**cfg.breakups.decoding.fit.input), mode3d=True)
                proba_r = proba.reshape(-1, proba.shape[-1])
                T = np.ones((model2.n_components, model1.n_components))
                for j in range(model1.n_components):
                    select_idx = np.reshape((states.select(**cfg.breakups.decoding.fit.states) == j).values, -1)
                    # print(proba.dtype,proba.shape)
                    # print(proba.shape,(states.select(**cfg.breakups.decoding.fit.states)==j).values.shape)
                    T[:, j] = proba_r[select_idx].sum(0)

                T = normalise(T, axis=0)

                ### testing ####
                # test_model1_in, _ = split_model_emission(model1, split_indices=instantiate(cfg.neurons_split_indices)[0:1])
                # proba1 = test_model1_in.predict_proba(obs.select(**cfg.breakups.decoding.test.input),mode3d=True)
                # proba2 = test_model2_in.predict_proba(obs.select(**cfg.breakups.decoding.test.input),mode3d=True)
                #
                # num_samples = proba2.shape[0]
                # M = (proba2.T @ proba1) / num_samples
                #
                eps = 1e-8
                # test_loglikelihood = (np.log(M+eps) * T).sum()
                # result_data[result_name] = test_loglikelihood

                proba = test_model2_in.predict_proba(obs.select(**cfg.breakups.decoding.test.input), mode3d=True)
                proba_r = proba.reshape(-1, proba.shape[-1])
                T_test = np.ones((model2.n_components, model1.n_components))

                counts_ = np.bincount(
                    states.select(
                        **cfg.breakups.decoding.test.states
                    ).values.flatten()
                )

                counts = np.zeros(model1.n_components)
                counts[:counts_.shape[0]] = counts_
                counts = normalise(counts + eps)

                mean_div = 0
                for j in range(model1.n_components):
                    select_idx = np.reshape((states.select(**cfg.breakups.decoding.test.states) == j).values, -1)
                    # print(proba.dtype,proba.shape)
                    # T_test[:,j] = normalise( proba[select_idx].sum(0) + eps )
                    select_proba = proba_r[select_idx]

                    if select_proba.shape[0] > 0:
                        mean_div += jenson_shannon_divergence(
                            # T[:,j],
                            # T_test[:, j]
                            select_proba + eps,
                            T_test[None, :, j]
                        ).mean() * counts[j]

                # test_d_js = (np.log(T + eps) * T_test).sum()

                result_data['decoder_' + result_name] = mean_div
                print('decoder_' + result_name, mean_div)

                mean_D_KL = (counts[None, :] * counts[:, None] * (
                            T[:, None, :] * np.log(T[:, None, :] / T[:, :, None] + eps)).sum(0)).sum()
                print('consistency_' + result_name, mean_D_KL)
                result_data['consistency_' + result_name] = mean_D_KL

                # fig,ax = plt.subplots()
                # ax.imshow(T)
                # fig.savefig('plots/test_plots/T.png')
                # plt.close(fig)
                #
                # fig,ax = plt.subplots()
                # ax.plot(counts)
                # fig.savefig('plots/test_plots/counts.png')
                # plt.close(fig)
                #
                # fig,axs = plt.subplots(1,3)
                # im = axs[0].imshow(T_test,vmin=0)
                # fig.colorbar(im, ax=axs[0],shrink=0.7)
                # im = axs[1].imshow(T, vmin=0)
                # fig.colorbar(im,ax=axs[1],shrink=0.7)
                # axs[2].plot(counts)
                # fig.tight_layout()
                # plt.savefig(f'plots/test_plots/T_Ttest{result_name}.png')

        if cfg.analysis.compute_latent_decoding_linear:
            teacher_inout = split_model_emission(teacher, split_indices=instantiate(cfg.neurons_split_indices)[1:2])[0]
            student_inout = split_model_emission(student, split_indices=instantiate(cfg.neurons_split_indices)[1:2])[0]
            for name, teacher_, student_ in [('teacher->student', teacher_inout, student_inout),
                                             ('student->teacher', student_inout, teacher_inout)]:
                # print(data.select(**cfg.breakups.decoding.fit.input))
                train_input_data = data.select(**cfg.breakups.decoding.fit.input).values
                test_input_data = data.select(**cfg.breakups.decoding.test.input).values
                # teacher_.predict_proba(input_data,mode3d=True)
                teacher_in, _ = split_model_emission(teacher_,
                                                     split_indices=instantiate(cfg.neurons_split_indices)[0:1])
                student_in, _ = split_model_emission(student_,
                                                     split_indices=instantiate(cfg.neurons_split_indices)[0:1])
                proba_teacher = teacher_in.predict_proba(train_input_data, mode3d=True)
                proba_student = student_in.predict_proba(train_input_data, mode3d=True)
                proba_teacher_r = proba_teacher.reshape(-1, proba_teacher.shape[-1])
                proba_student_r = proba_student.reshape(-1, proba_student.shape[-1])

                y = proba_student_r
                print(cfg.decoding)
                if hasattr(cfg.decoding, 'preprocess_target'):
                    y = instantiate(cfg.decoding.preprocess_target)(y)

                # from sklearn.linear_model import LinearRegression
                model = instantiate(cfg.decoding.regression_model)
                model.fit(
                    proba_teacher_r,
                    y
                )
                proba_teacher = teacher_in.predict_proba(test_input_data, mode3d=True)
                proba_student = student_in.predict_proba(test_input_data, mode3d=True)
                proba_teacher_r = proba_teacher.reshape(-1, proba_teacher.shape[-1])
                proba_student_r = proba_student.reshape(-1, proba_student.shape[-1])

                pred_r = getattr(model, cfg.decoding.predict_method)(proba_teacher_r)

                metric = instantiate(cfg.decoding.metric)
                score = np.stack([metric(
                    proba_student_r[i],
                    pred_r[i]
                ) for i in range(pred_r.shape[0])]).mean()

                print(name, 'decoding score', score)
                result_data['linear_decoder_' + name] = score

        # if cfg.analysis.compute_latent_decoding_logistic:
        #     teacher_inout = split_model_emission(teacher, split_indices=instantiate(cfg.neurons_split_indices)[1:2])[0]
        #     student_inout = split_model_emission(student, split_indices=instantiate(cfg.neurons_split_indices)[1:2])[0]
        #     for name, teacher_, student_ in [('teacher->student', teacher_inout, student_inout),
        #                                      ('student->teacher', student_inout, teacher_inout)]:
        #         # print(data.select(**cfg.breakups.decoding.fit.input))
        #         train_input_data = data.select(**cfg.breakups.decoding.fit.input).values
        #         test_input_data = data.select(**cfg.breakups.decoding.test.input).values
        #         # teacher_.predict_proba(input_data,mode3d=True)
        #         teacher_in, _ = split_model_emission(teacher_,
        #                                              split_indices=instantiate(cfg.neurons_split_indices)[0:1])
        #         student_in, _ = split_model_emission(student_,
        #                                              split_indices=instantiate(cfg.neurons_split_indices)[0:1])
        #         proba_teacher = teacher_in.predict_proba(train_input_data, mode3d=True)
        #         proba_student = student_in.predict_proba(train_input_data, mode3d=True)
        #         proba_teacher_r = proba_teacher.reshape(-1, proba_teacher.shape[-1])
        #         proba_student_r = proba_student.reshape(-1, proba_student.shape[-1])
        #
        #         # from sklearn.linear_model import LinearRegression
        #         model = instantiate(cfg.decoding.regression_model)
        #         model.fit(
        #             proba_teacher_r,
        #             proba_student_r
        #         )
        #         proba_teacher = teacher_in.predict_proba(test_input_data, mode3d=True)
        #         proba_student = student_in.predict_proba(test_input_data, mode3d=True)
        #         proba_teacher_r = proba_teacher.reshape(-1, proba_teacher.shape[-1])
        #         proba_student_r = proba_student.reshape(-1, proba_student.shape[-1])
        #
        #         score = model.score(
        #             proba_teacher_r,
        #             proba_student_r
        #         )
        #         print(name, 'decoding score', score)
        #         result_data['linear_decoder_' + name] = score

        if cfg.analysis.compute_latent_decoding_shuffled:
            print(teacher.n_features, student.n_features)
            generate_all_data = instantiate(cfg.generate_all_data)
            teacher_ = split_model_emission(teacher, split_indices=instantiate(cfg.neurons_split_indices)[1:2])[0]
            student_ = split_model_emission(student, split_indices=instantiate(cfg.neurons_split_indices)[1:2])[0]
            print('n_features', teacher_.n_features, student_.n_features)
            print(instantiate(cfg.neurons_split_indices))
            for model1, model2, result_name in [
                                                   (teacher_, student_, 'decoder_teacher->student shuffled'),
                                                   (student_, teacher_, 'decoder_student->teacher shuffled')
                                               ][:]:
                # cfg.generate_all_data_dictmodule
                # obs,states = generate_all_data(model1,return_states=True)

                # obs = instantiate(cfg.numpy_to_xarray_with_breakdownlabels,_convert_='partial')(
                #     obs #[:,:,:]
                # )
                # # obs = obs.select(neurons_split=['heldin','heldout'])
                # states = instantiate(cfg.numpy_to_xarray_with_breakdownlabels_states,_convert_='partial')(
                #     states[...,0]
                # )
                all_data = instantiate(cfg.generate_all_data_with_states_dictmodule, _convert_='partial')(
                    hmm_model=model1)
                obs, states = all_data['data_xarray'], all_data['states_data_xarray']

                obs = obs[np.random.permutation(obs.shape[0]), np.random.permutation(obs.shape[1])]
                states = states[np.random.permutation(states.shape[0]), np.random.permutation(states.shape[1])]

                test_model2_in, _ = split_model_emission(model2,
                                                         split_indices=instantiate(cfg.neurons_split_indices)[0:1])
                # proba1 = model1.predict_proba(obs.select(**cfg.breakups.decoding.fit.input),mode3d=True)
                proba = test_model2_in.predict_proba(obs.select(**cfg.breakups.decoding.fit.input), mode3d=True)
                proba_r = proba.reshape(-1, proba.shape[-1])

                T = np.ones((model2.n_components, model1.n_components))
                for j in range(model1.n_components):
                    select_idx = np.reshape((states.select(**cfg.breakups.decoding.fit.states) == j).values, -1)
                    # print(proba.dtype,proba.shape)
                    T[:, j] = proba_r[select_idx].sum(0)

                T = normalise(T, axis=0)

                ### testing ####
                # test_model1_in, _ = split_model_emission(model1, split_indices=instantiate(cfg.neurons_split_indices)[0:1])
                # proba1 = test_model1_in.predict_proba(obs.select(**cfg.breakups.decoding.test.input),mode3d=True)
                # proba2 = test_model2_in.predict_proba(obs.select(**cfg.breakups.decoding.test.input),mode3d=True)
                #
                # num_samples = proba2.shape[0]
                # M = (proba2.T @ proba1) / num_samples
                #
                eps = 1e-6
                # test_loglikelihood = (np.log(M+eps) * T).sum()
                # result_data[result_name] = test_loglikelihood

                proba = test_model2_in.predict_proba(obs.select(**cfg.breakups.decoding.test.input), mode3d=True)
                proba_r = proba.reshape(-1, proba.shape[-1])

                T_test = np.ones((model2.n_components, model1.n_components))

                counts_ = np.bincount(
                    states.select(
                        **cfg.breakups.decoding.test.states
                    ).values.flatten()
                )

                counts = np.zeros(model1.n_components)
                counts[:counts_.shape[0]] = counts_
                counts = normalise(counts + eps)

                mean_div = 0
                for j in range(model1.n_components):
                    select_idx = np.reshape((states.select(**cfg.breakups.decoding.test.states) == j).values, -1)
                    # print(proba.dtype,proba.shape)
                    # T_test[:,j] = normalise( proba[select_idx].sum(0) + eps )
                    select_proba = proba_r[select_idx]

                    if select_proba.shape[0] > 0:
                        mean_div += jenson_shannon_divergence(
                            # T[:,j],
                            # T_test[:, j]
                            select_proba + eps,
                            T_test[None, :, j]
                        ).mean() * counts[j]

                # test_d_js = (np.log(T + eps) * T_test).sum()

                result_data[result_name] = mean_div

                print(result_name, mean_div)
                #
                # fig,axs = plt.subplots(1,3)
                # im = axs[0].imshow(T_test,vmin=0)
                # fig.colorbar(im, ax=axs[0],shrink=0.7)
                # im = axs[1].imshow(T, vmin=0)
                # fig.colorbar(im,ax=axs[1],shrink=0.7)
                # axs[2].plot(counts)
                # fig.tight_layout()
                # plt.savefig(f'plots/test_plots/T_Ttest{result_name}.png')

        if cfg.analysis.compute_k_shot:
            # K_range = np.logspace(0.5,2,15).astype(int)
            K_range = np.array([6])
            repeats = cfg.train_trials // K_range
            print(repeats)
            test_students = []
            for i, k in enumerate(K_range):
                save_repeats = []
                for rep in range(repeats[i]):
                    test_student = deepcopy(student)
                    test_student_in, test_student_reallyout = split_model_emission(test_student,
                                                                                   split_indices=instantiate(
                                                                                       cfg.neurons_split_indices)[0:1])
                    test_student = CoHMM(test_student_in, test_student_reallyout)

                    test_student.decoder.lambdas_ = np.zeros(
                        (test_student.decoder.n_components, cfg.num_neurons_reallyheldout))
                    test_student.decoder.n_features = cfg.num_neurons_reallyheldout
                    select_k_idx = np.random.choice(cfg.train_trials, size=k, replace=False)

                    test_student.decoder.params = 'l'

                    test_student.co_fit(
                        data.select(**cfg.breakups.k_shot_fit.input)[select_k_idx]()['X'],
                        data.select(**cfg.breakups.k_shot_fit.target)[select_k_idx]()['X'],
                        lengths=data.select(**cfg.breakups.k_shot_fit.input)[select_k_idx]()['lengths']
                    )
                    test_student = fuse_model(test_student)
                    # save_repeats.append(test_student)

                    ## scoring
                    mod = test_student
                    (
                        mod_in,
                        mod_out,
                        mod_reallyout
                    ) = split_model_emission(mod, split_indices=instantiate(cfg.neurons_split_indices))

                    split_student = CoHMM(mod_in, mod_out)

                    # test_in_data, test_out_data, test_reallyout_data = test_data.split(cfg.neurons_split_indices,axis=-1)

                    test_pred_out = split_student.predict(**data.select(**cfg.breakups.k_shot_test.input)())
                    test_pred_out = test_pred_out.reshape(*data.select(**cfg.breakups.k_shot_test.target).shape)
                    test_pred_out[np.isnan(test_pred_out)] = 0
                    co_bps = bits_per_spike(test_pred_out, data.select(**cfg.breakups.k_shot_test.target).to_numpy())
                    save_repeats.append(co_bps)
                score_name = f'{k}-shot co-smoothing'
                result_data[score_name] = sum(save_repeats) / len(save_repeats)
                print(score_name, result_data[score_name])
                test_students.append(save_repeats)

            # score_names = ['co-smoothing']+[f'{k}-shot co-smoothing' for k in K_range]
            # score_names = [f'{k}-shot co-smoothing' for k in K_range]
            # for score_name, mod in zip(score_names,test_students):
            #     (
            #         mod_in,
            #         mod_out,
            #         mod_reallyout
            #     ) = split_model_emission(mod,split_indices=instantiate(cfg.neurons_split_indices))
            #
            #     split_student = CoHMM(mod_in, mod_out)
            #
            #     # test_in_data, test_out_data, test_reallyout_data = test_data.split(cfg.neurons_split_indices,axis=-1)
            #
            #     test_pred_out = split_student.predict(**data.select(**cfg.breakups.k_shot_test.input)())
            #     test_pred_out = test_pred_out.reshape(*data.select(**cfg.breakups.k_shot_test.target).shape)
            #     test_pred_out[np.isnan(test_pred_out)] = 0
            #     co_bps = bernoulli_bits_per_spike(test_pred_out,data.select(**cfg.breakups.k_shot_test.target))
            #     result_data[score_name] = co_bps

        if cfg.analysis.compute_fisher_info_matrix:

            from hmm_fisher_info import (
                bernoulli_loglikelihood_loss,
                batch_bernoulli_loglikelihood_loss,
                # compute_bernoulli_MLE,
                # batch_compute_poisson_MLE,
                # batch_compute_bernoulli_MLE,
                batch_batch_bernoulli_loglikelihood_loss,
                # value_and_grad_batch_bernoulli_loglikelihood_loss,
                # # batch_hess,
                # batch_real_hess,
                function_dict
            )

            test_student = deepcopy(student)
            test_student_in, test_student_reallyout = split_model_emission(test_student, split_indices=instantiate(
                cfg.neurons_split_indices)[0:1])
            print('input shape', data.select(**cfg.breakups.k_shot_fit.input).shape)
            test_student_in.implementation = 'scaling'
            # print('impl',test_student_in.implementation)
            Y = data.select(**cfg.breakups.k_shot_fit.input)
            Y_heldout = data.select(**cfg.breakups.k_shot_fit.target)
            posteriors = test_student_in.predict_proba(Y, mode3d=True)
            # posteriors = (log_posteriors)
            # posteriors = posteriors/posteriors.sum(-1,keepdims=True)
            # print('posteriors.shape',posteriors.shape)
            # print('lambdas.shape',lambdas.shape)
            # print(np.isnan(posteriors[0]).any())
            # print('lambdas',lambdas)
            # print('posterior summed',posteriors[0].sum(-1))
            # print(bernoulli_loglikelihood_loss(
            #     jnp.array(Y_heldout.values)[0],
            #     jnp.array(posteriors)[0],
            #     jnp.array(lambdas)
            # ))
            # print(batch_bernoulli_loglikelihood_loss(
            #     jnp.array(Y_heldout.values),
            #     jnp.array(posteriors),
            #     jnp.array(lambdas)
            # ))
            funcs = function_dict[cfg.emission_mode]

            trial_ids = np.arange(cfg.train_trials)

            # lambdas_inf = np.array(funcs['compute_MLE'](
            #     jnp.array(Y_heldout.values[trial_ids].reshape(-1,Y_heldout.shape[-1])),
            #     jnp.array(posteriors[trial_ids].reshape(-1,posteriors.shape[-1]))
            # ))
            lambdas_inf = test_student_reallyout.lambdas_
            print(
                jnp.array(Y_heldout.values).shape,
                jnp.array(posteriors).shape,
                jnp.array(lambdas_inf).shape
            )
            # loss_value = bernoulli_loglikelihood_loss(
            #     jnp.array(Y_heldout.values)[0],
            #     jnp.array(posteriors)[0],
            #     jnp.array(lambdas_inf)
            # )
            #
            loss_value = batch_bernoulli_loglikelihood_loss(
                # ,grad = funcs['value_and_grad_batch_loglikelihood_loss'](
                jnp.array(Y_heldout.values),
                jnp.array(posteriors),
                jnp.array(lambdas_inf)
            )
            # hess = funcs['batch_hessian'](
            #     jnp.array(Y_heldout.values),
            #     jnp.array(posteriors),
            #     jnp.array(lambdas_inf)
            # )
            # hess2 = np.outer(grad.flatten(), grad.flatten())/ (len(trial_ids)**2)
            # hess3 = batch_real_hess(
            #     jnp.array(Y_heldout.values),
            #     jnp.array(posteriors),
            #     jnp.array(lambdas_inf)
            # )
            # hess3 = hess3.reshape(hess2.shape)
            # print('hess.shape',hess2.shape,hess3.shape)

            ####### cramer rao with two hessians
            FI = funcs['batch_fisher_info'](
                jnp.array(Y_heldout.values),
                jnp.array(posteriors),
                jnp.array(lambdas_inf)
            )
            hessian = funcs['batch_hessian'](
                jnp.array(Y_heldout.values),
                jnp.array(posteriors),
                jnp.array(lambdas_inf)
            ).reshape(FI.shape)

            print(hessian.shape, FI.shape)

            make_path_if_not_exist(results_path + '_hessian_fisherinfo.npz')
            np.savez(results_path + '_hessian_fisherinfo.npz', hessian, FI)

            cond = np.linalg.cond(FI)
            print(cond)
            factor = None
            diag_factor = None
            if not cond > 1e12:
                factor = np.trace(hessian @ np.linalg.inv(FI))  # + 1e-3 * np.eye(hessian.shape[0])))
                diag_factor = (np.diag(hessian) / np.diag(FI)).sum()

            result_data['trace factor'] = factor
            result_data['diag trace factor'] = diag_factor
            print('trace factor', factor, diag_factor)

            # fig,ax = plt.subplots(figsize=(5,5))
            # ax.scatter(hess.flatten(),hess3.flatten() ,s=3)
            # all_hess=np.concatenate([hess.flatten(),hess3.flatten()])
            # minv,maxv =all_hess.min(),all_hess.max()
            # ax.plot([minv,maxv],[minv,maxv],ls='dashed',c='black')
            # ax.set_xlabel(r'$E\frac{\partial L}{\partial \phi_i}\frac{\partial L}{\partial \phi_j}$',fontsize=17)
            # ax.set_ylabel(r'$E\frac{\partial^2 L}{\partial \phi_i\partial \phi_j}$',fontsize=17)
            # ax.set_ylim(minv, maxv)
            # ax.set_xlim(minv, maxv)
            # ax.set_aspect('equal')
            # fig.tight_layout()
            # fig.savefig('plots/test_plots/verify_hess.png')
            # plt.close()
            # print(
            #     loss_value / len(trial_ids) / cfg.length,
            #     np.linalg.norm(grad) / len(trial_ids) / cfg.length ,
            #     np.linalg.norm(hess2)/ len(trial_ids) / cfg.length
            # )
            result_data['original likelihood jax'] = -loss_value / Y_heldout.shape[0]
            # hess_T = hess.reshape(cfg.student.n_components,cfg.num_neurons_reallyheldout,cfg.student.n_components,
            #                       cfg.num_neurons_reallyheldout).T.reshape(hess.shape)
            # hess3_T = hess3.reshape(cfg.student.n_components, cfg.num_neurons_reallyheldout, cfg.student.n_components,
            #                       cfg.num_neurons_reallyheldout).T.reshape(hess3.shape)

            # from matplotlib.colors import LogNorm
            # fig,axs = plt.subplots(1,2,figsize=(8,4),sharex=True,sharey=True)
            # ax= axs[0]
            # im = ax.imshow(np.abs(hess_T),norm=LogNorm(vmin=minv,vmax=maxv))
            # ax.set_title(r'$E\frac{\partial L}{\partial \phi_i}\frac{\partial L}{\partial \phi_j}$')
            # # fig.colorbar(im)
            #
            # ax = axs[1]
            # ax.imshow(np.abs(hess3_T),norm=LogNorm(vmin=minv,vmax=maxv))
            # ax.set_title(r'$E\frac{\partial^2 L}{\partial \phi_i\partial \phi_j}$')
            # fig.tight_layout()
            # fig.savefig('plots/test_plots/hessian_imshow.png',dpi=300)
            # plt.show()

            # print( )
            if cfg.analysis.compute_k_shot_jax:
                K_range = np.logspace(0.5, 2, 15).astype(int)
                repeats = cfg.train_trials // K_range // 2
                # repeats = 10
                # K = 5
                all_loss_values = []
                for reps, K in list(zip(repeats, K_range)):
                    trial_ids = np.random.choice(Y_heldout.shape[0], size=(reps, K))
                    print(Y_heldout.values[trial_ids].shape)
                    # print()
                    MLE_func = instantiate(cfg.jax_MLE_func)
                    lambdas_K = np.array(MLE_func(
                        jnp.array(Y_heldout.values[trial_ids].reshape(reps, -1, Y_heldout.shape[-1])),
                        jnp.array(posteriors[trial_ids].reshape(reps, -1, posteriors.shape[-1]))
                    ))
                    # print(lambdas_K.shape)
                    loss_values = batch_batch_bernoulli_loglikelihood_loss(
                        jnp.array(Y_heldout.values),
                        jnp.array(posteriors),
                        jnp.array(lambdas_K)
                    )
                    all_loss_values += [loss_values / Y_heldout.shape[0]]
                    score_name = f'{K}-shot likelihood jax'
                    result_data[score_name] = -np.nanmean(loss_values / Y_heldout.shape[0])
                    # print(loss_values/Y_heldout.shape[0]/cfg.length)

            ### verifying taylor
            #
            # delta_phi = (lambdas_K-lambdas_inf[None]).reshape(lambdas_K.shape[0],-1)
            # print(delta_phi.shape)
            # rhs1 = loss_value / Y_heldout.shape[0] + (0.5 * delta_phi[:, None, :] @ hess [None] @ delta_phi[:, :, None])[:, 0, 0]
            # rhs2 = loss_value / Y_heldout.shape[0] + (0.5 * delta_phi[:, None, :] @ hess2[None] @ delta_phi[:, :, None])[:, 0, 0]
            # rhs3 = loss_value / Y_heldout.shape[0] + (0.5 * delta_phi[:, None, :] @ hess3[None] @ delta_phi[:, :, None])[:, 0, 0]
            # lhs = loss_values/Y_heldout.shape[0]

            # print('lhs',lhs)
            # print('rhs',rhs)

            # fig, ax = plt.subplots()
            # ax.scatter(lhs.flatten(), rhs1.flatten(), s=10, c='C0')
            # ax.scatter(lhs.flatten(), rhs2.flatten(), s=10, c='C1')
            # ax.scatter(lhs.flatten(), rhs3.flatten(), s=10, c='C2')
            # both = np.concatenate([lhs,rhs1,rhs2,rhs3])
            # minv, maxv = both.min(), both.max()
            # ax.plot([minv, maxv], [minv, maxv], ls='dashed', c='black')
            # margin = (maxv-minv)*0.1
            # ax.set_ylim(minv-margin, maxv+margin)
            # ax.set_xlim(minv-margin, maxv+margin)
            # ax.set_aspect('equal')
            # fig.savefig('plots/test_plots/verify_taylor.png')
            # plt.close()

            # fig, ax = plt.subplots()
            # ax.plot(K_range, factor / 2 / K_range, ls='dashed', label=r'$\frac{1}{2K}\text{Tr}[H I^{-1}]$')
            # ax.scatter(K_range, [(np.nanmean(l)) - loss_value / Y_heldout.shape[0] for l in all_loss_values],
            #            label=r' $\langle L(\phi_\infty)-L(\phi_K) \rangle $')
            # ax.legend()
            # ax.set_xlabel(r'$K$')
            # fig.savefig('plots/test_plots/kshot_analytical.png')

            # plt.show()
            # plt.close(fig)

            # value,grad = value_and_grad_batch_bernoulli_loglikelihood_loss(
            #     jnp.array(Y_heldout.values),
            #     jnp.array(posteriors),
            #     jnp.array(lambdas)
            # )

            # print(value/len(trial_ids)/cfg.length,grad.mean()/len(trial_ids)/cfg.length)

            test_student_in.implementation = 'log'

        if cfg.save_results:
            # results_path = os.path.join(cfg.teacher_save_path, 'groundtruth')
            save_results_loc = results_path + '.csv'
            if not os.path.exists(os.path.dirname(save_results_loc)):
                os.makedirs(os.path.dirname(save_results_loc))
            if os.path.exists(save_results_loc):
                DFread = pd.read_csv(save_results_loc, index_col=None)
                data_dict = DFread.T.to_dict()[0]

                data_dict.update(result_data)
                result_data = data_dict
            print(save_results_loc)
            # print(data)

            DF = pd.DataFrame([result_data])
            DF.to_csv(save_results_loc, index=False)

            # return result_data
            print('done saved')
        print('returning', result_data)
        return result_data
    print('done')
    return


@hydra.main(version_base='1.3', config_path=CONFIG_PATH, config_name=CONFIG_NAME)
def xarray_test(cfg):
    instantiate(cfg.numpy_seed)

    teacher = instantiate(cfg.teacher)

    # generate_all_data = instantiate(cfg.generate_all_data)
    # data_xarray = instantiate(
    #     cfg.numpy_to_xarray_with_breakdownlabels
    # )(
    #         generate_all_data(teacher)
    # )
    # print(data_xarray.shape)
    # # print(data_xarray.select(trials='train',neurons='heldin').shape)
    # print(data_xarray.select(trials='train',neurons=['heldin','heldout']).shape)
    # # print(data_xarray.loc[:,:,['heldin','heldout']])

    numpy_to_xarray_with_breakdownlabels = instantiate(cfg.numpy_to_xarray_with_breakdownlabels, _convert_='partial')
    # numpy_to_xarray_with_breakdownlabels.keywords['coords'] = {k: tuple(v) for k, v in
    #                                                            numpy_to_xarray_with_breakdownlabels.keywords[
    #                                                                'coords'].items()}

    # print(numpy_to_xarray_with_breakdownlabels)
    # print(OmegaConf.to_object(cfg.numpy_to_xarray_with_breakdownlabels))
    # print(numpy_to_xarray_with_breakdownlabels)
    # from builtins import tuple
    # print(tuple([0,1,2]))


@hydra.main(version_base='1.3', config_path=CONFIG_PATH, config_name=CONFIG_NAME)
def main_old(cfg):
    instantiate(cfg.numpy_seed)
    teacher = instantiate(cfg.teacher)
    # print(cfg.generate_all_data_dictmodule)
    all_data = instantiate(cfg.generate_all_data_with_states_dictmodule, _convert_='partial')(hmm_model=teacher)
    obs, states = all_data['data_xarray'], all_data['states_data_xarray']
    # data,states = instantiate(cfg.generate_all_data_with_states, _convert_='partial')(hmm_model=teacher)
    #
    print(
        # instantiate(cfg.breakups_dictmodule.fit)(data=data)
        # instantiate(cfg.breakups_dictmodule.fit)(data=data)
        # instantiate(cfg.numpy_to_xarray_with_breakdownlabels_states,_convert_='partial')(data=states)
        obs.shape, states.shape
    )


def main_old():
    from hmmlearn.hmm import BernoulliHMM
    cfg = OmegaConf.create(
        """
        _target_            : hmm_adapter.adapt_hmm_class
        _partial_           : true
        adapted_class_name  :  AdaptedBernoulliHMM
        """
    )
    print(
        instantiate(cfg)(BernoulliHMM)
    )


def omegaconf_resolvers():
    resolvers = {
        'eval': eval,
        'ind': lambda a, i: a[i],
        'listmul': lambda l, i: [l] * i,
        'getattr': getattr,
        'setattrs': setattrs,
        'as_tuple': lambda *args: tuple(args),
        'relpath': lambda p: os.path.join(
            '/home/kabird/lfads-torch-fewshot-benchmark', p
        )
    }
    for resolver_name, resolver_val in resolvers.items():
        if not OmegaConf.has_resolver(resolver_name):
            OmegaConf.register_new_resolver(resolver_name, resolver_val)


if __name__ == '__main__':
    # omegaconf_resolvers()
    # OmegaConf.register_new_resolver("eval", eval)
    # OmegaConf.register_new_resolver("ind", lambda a, i: a[i])
    # OmegaConf.register_new_resolver("listmul", lambda l, i: [l] * i)
    # OmegaConf.register_new_resolver("getattr", getattr)
    # OmegaConf.register_new_resolver("setattrs", lambda target, attributes, values:  [setattr(target,attr,val) for attr,val in zip(attributes,values)])
    # OmegaConf.register_new_resolver('as_tuple', lambda *args: tuple(args))

    # xarray_test()
    decorated_main()
