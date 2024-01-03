import matplotlib.pyplot as plt
from omegaconf import DictConfig, OmegaConf
import hydra
import os
import pandas as pd
import pickle as pkl
import similarity
import numpy as np
# from hydra.utils import instantiate
from config_utils import instantiate
from main import jenson_shannon_divergence
from prepare_model import CoHMM,split_model_emission,fuse_model
from copy import deepcopy
from utils import flatten_with_lengths,HMM_Dataset, normalise, setattrs, setattrs_kwargs
from metrics import bernoulli_bits_per_spike
from hmm_adapter import CoHMM3d as CoHMM

import matplotlib as mpl

import prepare_model

mpl.rcParams['text.usetex'] = True

CONFIG_PATH = "configs"
# CONFIG_NAME = "config"
CONFIG_NAME = "config_cohmm"

@hydra.main(version_base='1.3', config_path=CONFIG_PATH, config_name=CONFIG_NAME)
def main_old(cfg):
    instantiate(cfg.numpy_seed)

    teacher = instantiate(cfg.teacher)

    generate_all_data = instantiate(cfg.generate_all_data)
    generate_train_val_test_split = instantiate(cfg.generate_train_val_test_split)
    train,val,test = generate_train_val_test_split(
        generate_all_data(teacher)
    )

    student =  instantiate(cfg.student)

    # train_data = dict(zip(['X', 'lengths'], flatten_with_lengths(train)))
    # val_data = dict(zip(['X_val', 'lengths_val'], flatten_with_lengths(train)))
    # test_data = dict(zip(['X', 'lengths'], flatten_with_lengths(test)))
    train_data = HMM_Dataset(train,argnames=['X', 'lengths'])
    val_data = HMM_Dataset(val, argnames=['X_val', 'lengths_val'])
    test_data = HMM_Dataset(test, argnames=['X', 'lengths'])
    # print(train_data()['X'].shape,val_data()['X_val'].shape)

    student.fit(**train_data(),**val_data())

    student_save_path = cfg.student_save_path
    if not os.path.exists(os.path.dirname(student_save_path)):
        os.makedirs(os.path.dirname(student_save_path))
    with open(student_save_path, 'wb') as f:
        pkl.dump(student, f)


    if cfg.run_analysis:
        if cfg.use_teacher_as_student:
            student = hydra.utils.instantiate(teacher)
            data = {
                'model_name': 'Groundtruth',
                'model_id': None,
                'n_components': teacher.n_components,
                'iterations': teacher.monitor_.iter,
                'train_trials': cfg.train_trials,
            }
            results_path = os.path.join(cfg.teacher_save_path,'groundtruth')
        else:
            student_save_path = cfg.student_save_path
            with open(student_save_path,'rb') as f:
                student = pkl.load(f)
            data = {
                'model_name': cfg.teacher.model_name,
                'model_id': cfg.student_index,
                'n_components': student.n_components,
                'iterations': student.monitor_.iter,
                'train_trials': cfg.train_trials,
            }

            results_path = cfg.student_save_path

        if cfg.analysis.compute_k_shot:
            K_range = np.logspace(0.5,2,15).astype(int)
            test_students = []
            for k in K_range:
                test_student = deepcopy(student)
                test_student_in, test_student_out = split_model_emission(test_student, split_indices=instantiate(cfg.neurons_split_indices)[:1])
                test_student = CoHMM(test_student_in, test_student_out)

                test_student.decoder.lambdas_ = np.zeros(test_student.decoder.lambdas_.shape)
                select_k_idx = np.random.choice(train_data.array.shape[0],size=k,replace=False)
                train_data_k = HMM_Dataset(train_data.array[select_k_idx])
                train_in_data, train_out_data, train_reallyout_data = train_data_k.split(cfg.neurons_split_indices, axis=-1)
                test_student.decoder.params = 'l'
                # print('shapes',train_in_data.array.shape,train_out_data.array.shape,test_student.decoder.lambdas_.shape,student.lambdas_.shape)
                test_student.co_fit(
                    train_in_data()['X'],
                    train_out_data()['X'],
                    lengths=train_out_data()['lengths']
                )
                test_student = fuse_model(test_student)
                test_students.append(test_student)

            # score_names = ['co-smoothing']+[f'{k}-shot co-smoothing' for k in K_range]
            score_names = [f'{k}-shot co-smoothing' for k in K_range]
            for score_name, mod in zip(score_names,test_students):
                (
                    mod_in,
                    mod_out,
                    mod_reallyout
                ) = split_model_emission(mod,split_indices=instantiate(cfg.neurons_split_indices))

                split_student = CoHMM(mod_in, mod_out)

                test_in_data, test_out_data, test_reallyout_data = test_data.split(cfg.neurons_split_indices,axis=-1)

                test_pred_out = split_student.predict(**test_in_data())
                test_pred_out = test_pred_out.reshape(test_out_data.array.shape)
                test_pred_out[np.isnan(test_pred_out)]= 0
                print('isnan',np.isnan(test_pred_out))
                co_bps = bernoulli_bits_per_spike(test_pred_out,test_out_data.array)
                data[score_name] = co_bps



    # results_path = os.path.join(cfg.teacher_save_path, 'groundtruth')
    save_results_loc = results_path + '.csv'
    if os.path.exists(save_results_loc):
        DFread = pd.read_csv(save_results_loc, index_col=None)
        data_dict = DFread.T.to_dict()[0]
        data_dict.update(data)
        data = data_dict

    # print(data)
    print(save_results_loc)
    DF = pd.DataFrame([data])
    DF.to_csv(save_results_loc, index=False)

    return



@hydra.main(version_base='1.3', config_path=CONFIG_PATH, config_name=CONFIG_NAME)
def main(cfg):

    instantiate(cfg.numpy_seed)

    teacher = instantiate(cfg.teacher)

    # generate_all_data = instantiate(cfg.generate_all_data)
    # data = instantiate(
    #     cfg.numpy_to_xarray_with_breakdownlabels, _convert_='partial'
    # )(
    #         generate_all_data(teacher)
    # )
    data = instantiate(cfg.generate_all_data_dictmodule,_convert_='partial')(hmm_model=teacher)

    np.random.seed(cfg.student_index)
    student =  instantiate(cfg.student)

        # train_data = dict(zip(['X', 'lengths'], flatten_with_lengths(train)))
        # val_data = dict(zip(['X_val', 'lengths_val'], flatten_with_lengths(train)))
        # test_data = dict(zip(['X', 'lengths'], flatten_with_lengths(test)))
        # train_data = HMM_Dataset(train,argnames=['X', 'lengths'])
        # val_data = HMM_Dataset(val, argnames=['X_val', 'lengths_val'])
        # test_data = HMM_Dataset(test, argnames=['X', 'lengths'])
        # print(train_data()['X'].shape,val_data()['X_val'].shape)

        # fit_args = {**data.select(**cfg.breakups.fit.train)(),
        #             **data.select(**cfg.breakups.fit.val)(keys=('X_val', 'lengths_val'))}
        #
        # fit_args = instantiate(cfg.breakups_dictmodule.fit)(data=data)


    if cfg.run_train:
        fit_args = {
            'X'     : data.select(**cfg.breakups.fit.train),
            'X_val' : data.select(**cfg.breakups.fit.val),
            'mode3d': True
        }

        print(student)
        if hasattr(cfg, 'training_config'):
            for stage in instantiate(cfg.training_config):
                setattrs_kwargs(student,**stage)
                student.fit(**fit_args)
        else:
            student.fit(**fit_args)

        student_save_path = cfg.student_save_path
        if not os.path.exists(os.path.dirname(student_save_path)):
            os.makedirs(os.path.dirname(student_save_path))
        with open(student_save_path, 'wb') as f:
            pkl.dump(student, f)


    if cfg.run_analysis:
        if cfg.use_teacher_as_student:
            # student = hydra.utils.instantiate(teacher)
            (
                teacher_in,
                teacher_out,
                teacher_reallyheldout
            ) = split_model_emission(deepcopy(teacher),split_indices=instantiate(cfg.neurons_split_indices))
            split_teacher = CoHMM(teacher_in,teacher_out)
            teacher_inout = fuse_model(split_teacher)
            student = teacher_inout
            result_data = {
                'model_name': 'Groundtruth',
                'model_id': None,
                'n_components': teacher.n_components,
                'iterations': teacher.monitor_.iter,
                'train_trials': cfg.train_trials,
            }
            results_path = os.path.join(cfg.teacher_save_path,'groundtruth')
        else:
            student_save_path = cfg.student_save_path
            with open(student_save_path,'rb') as f:
                student = pkl.load(f)
            result_data = {
                'model_name': cfg.student_name,
                'model_id': cfg.student_index,
                'n_components': student.n_components,
                'iterations': student.monitor_.iter,
                'train_trials': cfg.train_trials,
            }

            results_path = cfg.student_save_path

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

            student_latent = student_encoder.predict_proba(data.select(**cfg.breakups.cosmoothing.input),mode3d=True)
            teacher_latent = teacher_encoder.predict_proba(data.select(**cfg.breakups.cosmoothing.input),mode3d=True)

            for name,measure in instantiate(cfg.similarity_measures).items():
                result_data['similarity.'+name] = measure(student_latent, teacher_latent)

        if cfg.analysis.compute_co_smoothing:
            test_student = deepcopy(student)
            (
                test_student_in,
                test_student_out
            ) = split_model_emission(
                test_student,
                split_indices=instantiate(cfg.neurons_split_indices)[:1]
            )
            split_student = CoHMM(test_student_in,test_student_out)
            test_pred_out = split_student.predict(data.select(**cfg.breakups.cosmoothing.input),mode3d=True)
            test_pred_out = test_pred_out.reshape(*data.select(**cfg.breakups.cosmoothing.target).shape)
            test_pred_out[np.isnan(test_pred_out)] = 0
            co_bps = bernoulli_bits_per_spike(test_pred_out, data.select(**cfg.breakups.cosmoothing.target))
            result_data['original co-smoothing'] = co_bps

        if cfg.analysis.compute_latent_decoding:
            print(teacher.n_features, student.n_features)
            generate_all_data = instantiate(cfg.generate_all_data)
            teacher_ = split_model_emission(teacher,split_indices=instantiate(cfg.neurons_split_indices)[1:2])[0]
            student_ = split_model_emission(student, split_indices=instantiate(cfg.neurons_split_indices)[1:2])[0]
            print('n_features',teacher_.n_features,student_.n_features)
            print(instantiate(cfg.neurons_split_indices))
            for model1,model2,result_name in [
                (teacher_, student_, 'decoder_teacher->student'),
                (student_, teacher_, 'decoder_student->teacher')
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

                test_model2_in, _ = split_model_emission(model2, split_indices=instantiate(cfg.neurons_split_indices)[0:1])
                # proba1 = model1.predict_proba(obs.select(**cfg.breakups.decoding.fit.input),mode3d=True)
                proba = test_model2_in.predict_proba(obs.select(**cfg.breakups.decoding.fit.input), mode3d=True)

                T = np.ones((model2.n_components,model1.n_components))
                for j in range(model1.n_components):
                    select_idx = np.reshape((states.select(**cfg.breakups.decoding.fit.states)==j).values,-1)
                    # print(proba.dtype,proba.shape)
                    T[:,j] = proba[select_idx].sum(0)

                T = normalise(T,axis=0)



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

                T_test = np.ones((model2.n_components,model1.n_components))

                counts_ = np.bincount(
                    states.select(
                        **cfg.breakups.decoding.test.states
                    ).values.flatten()
                )

                counts = np.zeros(model1.n_components)
                counts [:counts_.shape[0]] = counts_
                counts = normalise(counts)

                mean_div = 0
                for j in range(model1.n_components):
                    select_idx = np.reshape((states.select(**cfg.breakups.decoding.test.states)==j).values,-1)
                    # print(proba.dtype,proba.shape)
                    T_test[:,j] = normalise( proba[select_idx].sum(0) )

                    mean_div += jenson_shannon_divergence(
                        T[:,j],
                        T_test[:, j]
                    ) * counts[j]

                # test_d_js = (np.log(T + eps) * T_test).sum()

                result_data[result_name] = mean_div

                # fig,axs = plt.subplots(1,2)
                # im = axs[0].imshow(T_test,vmin=0)
                # fig.colorbar(im, ax=axs[0],shrink=0.7)
                # im = axs[1].imshow(T, vmin=0)
                # fig.colorbar(im,ax=axs[1],shrink=0.7)
                # plt.savefig('plots/test_plots/T_Ttest.png')


        if cfg.analysis.compute_k_shot:
            K_range = np.logspace(0.5,2,15).astype(int)
            test_students = []
            for k in K_range:
                test_student = deepcopy(student)
                test_student_in, test_student_reallyout = split_model_emission(test_student, split_indices=instantiate(cfg.neurons_split_indices)[0:1])
                test_student = CoHMM(test_student_in, test_student_reallyout)

                test_student.decoder.lambdas_ = np.zeros((test_student.decoder.n_components,cfg.num_neurons_reallyheldout))
                test_student.decoder.n_features = cfg.num_neurons_reallyheldout
                select_k_idx = np.random.choice(cfg.train_trials, size=k, replace=False)

                test_student.decoder.params = 'l'

                test_student.co_fit(
                    data.select(**cfg.breakups.k_shot_fit.input) [select_k_idx]()['X'],
                    data.select(**cfg.breakups.k_shot_fit.target)[select_k_idx]()['X'],
                    lengths=data.select(**cfg.breakups.k_shot_fit.input) [select_k_idx]()['lengths']
                )
                test_student = fuse_model(test_student)
                test_students.append(test_student)

            # score_names = ['co-smoothing']+[f'{k}-shot co-smoothing' for k in K_range]
            score_names = [f'{k}-shot co-smoothing' for k in K_range]
            for score_name, mod in zip(score_names,test_students):
                (
                    mod_in,
                    mod_out,
                    mod_reallyout
                ) = split_model_emission(mod,split_indices=instantiate(cfg.neurons_split_indices))

                split_student = CoHMM(mod_in, mod_out)

                # test_in_data, test_out_data, test_reallyout_data = test_data.split(cfg.neurons_split_indices,axis=-1)

                test_pred_out = split_student.predict(**data.select(**cfg.breakups.k_shot_test.input)())
                test_pred_out = test_pred_out.reshape(*data.select(**cfg.breakups.k_shot_test.target).shape)
                test_pred_out[np.isnan(test_pred_out)] = 0
                co_bps = bernoulli_bits_per_spike(test_pred_out,data.select(**cfg.breakups.k_shot_test.target))
                result_data[score_name] = co_bps

        # results_path = os.path.join(cfg.teacher_save_path, 'groundtruth')
        save_results_loc = results_path + '.csv'
        if os.path.exists(save_results_loc):
            DFread = pd.read_csv(save_results_loc, index_col=None)
            data_dict = DFread.T.to_dict()[0]

            data_dict.update(result_data)
            result_data = data_dict

        # print(data)
        print(save_results_loc)
        DF = pd.DataFrame([result_data])
        DF.to_csv(save_results_loc, index=False)

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
    all_data = instantiate(cfg.generate_all_data_with_states_dictmodule,_convert_='partial')(hmm_model=teacher)
    obs,states = all_data['data_xarray'], all_data['states_data_xarray']
    # data,states = instantiate(cfg.generate_all_data_with_states, _convert_='partial')(hmm_model=teacher)
    #
    print(
        # instantiate(cfg.breakups_dictmodule.fit)(data=data)
        # instantiate(cfg.breakups_dictmodule.fit)(data=data)
        # instantiate(cfg.numpy_to_xarray_with_breakdownlabels_states,_convert_='partial')(data=states)
        obs.shape,states.shape
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


if __name__ == '__main__':
    resolvers = {
        'eval'      : eval,
        'ind'       : lambda a, i: a[i],
        'listmul'   : lambda l, i: [l] * i,
        'getattr'   : getattr,
        'setattrs'  : setattrs,
        'as_tuple'  :  lambda *args: tuple(args),
    }
    for resolver_name,resolver_val in resolvers.items():
        if not OmegaConf.has_resolver(resolver_name):
            OmegaConf.register_new_resolver(resolver_name,resolver_val)
    # OmegaConf.register_new_resolver("eval", eval)
    # OmegaConf.register_new_resolver("ind", lambda a, i: a[i])
    # OmegaConf.register_new_resolver("listmul", lambda l, i: [l] * i)
    # OmegaConf.register_new_resolver("getattr", getattr)
    # OmegaConf.register_new_resolver("setattrs", lambda target, attributes, values:  [setattr(target,attr,val) for attr,val in zip(attributes,values)])
    # OmegaConf.register_new_resolver('as_tuple', lambda *args: tuple(args))

    # xarray_test()
    main()
