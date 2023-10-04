import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.utils import check_random_state
from hmmlearn.hmm import GaussianHMM, CategoricalHMM
from hmmlearn.vhmm import VariationalCategoricalHMM
import pickle as pkl
from functools import partial
import os
import seaborn as sns
import matplotlib as mpl
mpl.rcParams['text.usetex'] = True
#mpl.rcParams['text.latex.preamble'] = [r'\usepackage{amsmath}']

# def collater():
#     #options = [('None',1),(10,1),(10,10)]
#     #options = [(10, 0), (10, 10)] #('None', 1),
#     # dir_names = [f'models_traintrials70_window{window}_shiftlimits{shift}' for (window,shift) in options]
#     options = ['augmented_mode','augmented_with_shift_mode','augmented_with_shift_repeatwithoutshift_mode','vanilla_sliced_mode']#'vanilla_mode'
#     main_dir = 'all_models_validated/state7_obs3_GT'
#     main_path = os.path.join(main_dir,'models_traintrials70_')
#     dir_names = [main_path + opt for opt in options]
#     #dir_names += ['models_traintrials700_'+'vanilla_sliced_mode']
#
#     all_DFs = []
#     for dir_name in dir_names:
#         files = os.listdir(dir_name)
#         #print(files,len(files))
#         for file_name in files:
#             fullpath = os.path.join(dir_name, file_name)
#             #print(dir_name,file_name,fullpath,os.path.splitext(fullpath)[1])
#             if os.path.splitext(fullpath)[-1]=='.csv':
#                 DF = pd.read_csv(fullpath,index_col=None)
#                 all_DFs.append(DF)
#     all_DFs += [pd.read_csv(os.path.join(main_dir,'groundtruth.csv'),index_col=None)] #.replace('_validated','')
#     DF = pd.concat(all_DFs,ignore_index=True)
#     DF = DF.sort_values(by=['model_name','n_components','model_id'])
#
#     return DF,main_path

def collater(main_dir = 'all_models_validated_finetuning/state5_obs5_eps0.1_emeps0.6_GT'):
    #options = ['augmented_mode','augmented_mode_long','augmented_with_shift_mode','augmented_with_shift_repeatwithoutshift_mode','vanilla_sliced_mode']#'vanilla_mode'
    #options = ['vanilla_sliced_mode','sliced_and_augmented_with_small_shifts']
    #options = ['training_vanilla', 'training_augmented_with_shift', 'training_augmented_with_shift_then_vanilla','training_augmented_with_shift_then_vanilla_frozen_te'] # 'training_augmented'
    #options = ['pretrain_vanilla_then_finetuning_emission_vanilla','pretrain_augmented_with_shift_then_finetuning_emission_vanilla']
    options = ['training_vanilla']
    main_path = os.path.join(main_dir,'models_traintrials1000_')
    dir_names = [main_path + opt for opt in options]
    #dir_names += ['models_traintrials700_'+'vanilla_sliced_mode']

    all_DFs = []
    for dir_name in dir_names:
        files = os.listdir(dir_name)
        #print(files,len(files))
        for file_name in files:
            fullpath = os.path.join(dir_name, file_name)
            #print(dir_name,file_name,fullpath,os.path.splitext(fullpath)[1])
            if os.path.splitext(fullpath)[-1]=='.csv':
                DF = pd.read_csv(fullpath,index_col=None)
                all_DFs.append(DF)
    all_DFs += [pd.read_csv(os.path.join(main_dir,'groundtruth.csv'),index_col=None)] #.replace('_validated','')
    DF = pd.concat(all_DFs,ignore_index=True)
    DF = DF.sort_values(by=['model_name','n_components','model_id'])

    return DF,main_path


def plot_scatter_with_lines(
        x: str,
        y: str,
        data: pd.DataFrame,
        data_lines: pd.DataFrame,
        save_path,
        hue: None,
        func1 = sns.scatterplot,
        func2 = None,
        xlabel=None,
        ylabel=None,
        xlim=[],
        ylim=[],
        hlines=[],
):
    fig,ax = plt.subplots()
    func1(x=x, y=y, hue=hue, data=data, ax=ax, legend=True, alpha=0.6)
    if func2:
        func2(x=x,y=y,hue=hue,data = data,ax=ax)

    l = ax.axhline(data_lines[y].values[0], ls='dashed', color='black')
    l = ax.axvline(data_lines[x].values[0], ls='dashed', color='black')
    ax.set_xlabel(x if xlabel is None else xlabel)
    ax.set_ylabel(y if ylabel is None else ylabel)
    for i,ls in enumerate(hlines):
        ax.axhline(ls,color='C%d'%i,ls='dashed',lw=1)
    handles, labels = ax.get_legend_handles_labels()
    handles += [l]
    labels  += ['Ground-truth']
    ax.legend(handles,labels,fontsize=5,framealpha=0.3)
    ax.set_xlim(*xlim)
    ax.set_ylim(*ylim)
    fig.tight_layout()
    if not os.path.exists(os.path.dirname(save_path)):
        os.makedirs(os.path.dirname(save_path))
    fig.savefig(save_path,dpi=200)



def main():
    #DF = pd.read_csv('plots/collated.csv',index_col=0)

    DF,main_dir = collater()
    DF = DF.replace(to_replace='Groundtruth',value='Ground truth')
    print(DF[DF['model_name']=='Ground truth'].n_components)

    #DF['score'] = DF['test_score']
    #DF.loc[DF.model_name == 'Ground truth', 'test_self_consistency'] = DF.loc[DF.model_name == 'Ground truth', 'test_self_consistency_modified_pi']
    DF['self_consistency'] = DF['test_self_consistency']

    #sns.scatterplot(data = DF)
    modelsDF = DF[DF.model_name!='Ground truth']
    modelsGT = DF[DF.model_name == 'Ground truth']
    #modelsDF = modelsDF[modelsDF.score>-2.31]
    #modelsDF = modelsDF[modelsDF.score > -1.70]
    #modelsDF = modelsDF[modelsDF.test_score > -1.05]
    #modelsDF = modelsDF[modelsDF.test_score > -2.0]
    #modelsDF = modelsDF[modelsDF.test_score > modelsGT.minus_test_entropy.values[0]]

    print(os.path.join('plots', main_dir, 'test_PR_vs_score.png'))

    plot_args_list = [
        # test score
        {
            'x'             :'n_components',
            'y'             :'test_score',
            'hue'           : 'model_name',
            'data'          : modelsDF,
            'data_lines'    : DF[DF.model_name=='Ground truth'],
            'save_path'     : os.path.join('plots',main_dir,'test_score.png'),
            'func2'         : sns.lineplot,
            #'xlim'          : (0, 51),
            #'ylim'          : [-2.5,-2.2], #-1.625, -1.550],
            'hlines'        : list(modelsDF.groupby('model_name').test_score.max())
        },
        # test2 score
        {
            'x': 'n_components',
            'y': 'test_score2',
            'hue': 'model_name',
            'data': modelsDF,
            'data_lines': DF[DF.model_name == 'Ground truth'],
            'save_path': os.path.join('plots', main_dir, 'test_score2.png'),
            'func2': sns.lineplot,
            # 'xlim'          : (0, 51),
            # 'ylim'          : [-2.5,-2.2], #-1.625, -1.550],
            'hlines': list(modelsDF.groupby('model_name').test_score2.max())
        },
        # test self consistency
        {
            'x': 'n_components',
            'y': 'self_consistency',
            'hue': 'model_name',
            'data': modelsDF,
            'data_lines': DF[DF.model_name == 'Ground truth'],
            'save_path': os.path.join('plots', main_dir, 'test_self_consistency.png'),
            'func2': sns.lineplot,
            # 'xlim'          : (0, 51),
            #'ylim': [-1.7, -1.5],
        },
        # test self consistency modified pi
        {
            'x': 'n_components',
            'y': 'test_self_consistency_modified_pi',
            'hue': 'model_name',
            'data': modelsDF,
            'data_lines': DF[DF.model_name == 'Ground truth'],
            'save_path': os.path.join('plots', main_dir, 'test_self_consistency_modified_pi.png'),
            'func2': sns.lineplot,
            # 'xlim'          : (0, 51),
            # 'ylim': [-1.7, -1.5],
        },
        # test participation ratio
        {
            'x': 'n_components',
            'y': 'PR',
            'hue': 'model_name',
            'data': modelsDF,  # [modelsDF.test_score > -1.62],
            'data_lines': DF[DF.model_name == 'Ground truth'],
            'save_path': os.path.join('plots', main_dir, 'test_PR.png'),
            'func2': sns.lineplot,
            # 'xlim'          : (0, 51),
            # 'ylim': [-1.7, -1.5],
        },
        # test participation ratio
        {
            'x': 'test_score',
            'y': 'PR',
            'hue': 'model_name',
            'data': modelsDF,  # [modelsDF.test_score > -1.62],
            'data_lines': DF[DF.model_name == 'Ground truth'],
            'save_path': os.path.join('plots', main_dir, 'test_PR_vs_score.png'),
            # 'func2': sns.lineplot,
            # 'xlim' : [-1.625, -1.550],
            # 'ylim': [-1.7, -1.5],
        },
        # test self consistency vs score
        {
            'x': 'test_score',
            'y': 'self_consistency',
            'hue': 'model_name',
            'data': modelsDF,
            'data_lines': DF[DF.model_name == 'Ground truth'],
            'save_path': os.path.join('plots', main_dir, 'test_self_consistency_vs_score.png'),
            # 'func2': sns.lineplot,
            # 'xlim'          : (0, 51),
            # 'ylim': [-1.7, -1.5],
        },
        # test self consistency vs score
        {
            'x': 'test_score',
            'y': 'test_self_consistency_modified_pi',
            'hue': 'model_name',
            'data': modelsDF,
            'data_lines': DF[DF.model_name == 'Ground truth'],
            'save_path': os.path.join('plots', main_dir, 'self_consistency_modified_pi_vs_score.png'),
            # 'func2': sns.lineplot,
            # 'xlim'          : (0, 51),
            # 'ylim': [-1.7, -1.5],
        },

        # test D_JS_stationary_pi
        {
            'x': 'n_components',
            'y': 'D_JS_stationary_pi',
            'hue': 'model_name',
            'data': modelsDF,
            'data_lines': DF[DF.model_name == 'Ground truth'],
            'save_path': os.path.join('plots', main_dir, 'test_D_JS_stationary_pi.png'),
            'func2': sns.lineplot,
            # 'xlim'          : (0, 51),
            # 'ylim': [-1.7, -1.5],
        },
        # test Mutual info predicted probability
        {
            'x': 'n_components',
            'y': 'MI_predict_proba',
            'hue': 'model_name',
            'data': modelsDF, #[modelsDF.test_score>-1.62],
            'data_lines': DF[DF.model_name == 'Ground truth'],
            'save_path': os.path.join('plots', main_dir, 'test_MI_predict_proba.png'),
            'func2': sns.lineplot,
            # 'xlim'          : (0, 51),
            # 'ylim': [-1.7, -1.5],
        },
        # test Mutual info predicted probability versus score
        {
            'x': 'MI_predict_proba',
            'y': 'test_score',
            'hue': 'model_name',
            'data': modelsDF, #[modelsDF.test_score>-1.62],
            'data_lines': DF[DF.model_name == 'Ground truth'],
            'save_path': os.path.join('plots', main_dir, 'test_MI_predict_proba_versus_score.png'),
            'func2': None,
            # 'xlim'          : (0, 51),
            #'ylim': [-1.625, -1.550],
        },
        # test Mutual info predicted probability
        {
            'x': 'n_components',
            'y': 'posterior_entropy',
            'hue': 'model_name',
            'data': modelsDF,#[modelsDF.test_score > -1.62],
            'data_lines': DF[DF.model_name == 'Ground truth'],
            'save_path': os.path.join('plots', main_dir, 'test_posterior_entropy.png'),
            'func2': sns.lineplot,
            # 'xlim'          : (0, 51),
            # 'ylim': [-1.7, -1.5],
        },
        # test Mutual info predicted probability versus score
        {
            'x': 'train_score',
            'y': 'test_score',
            'hue': 'model_name',
            'data': modelsDF,
            'data_lines': DF[DF.model_name == 'Ground truth'],
            'save_path': os.path.join('plots', main_dir, 'test_vs_train_score.png'),
            'func2': None,
            # 'xlim'          : (0, 51),
            # 'ylim': [-1.625, -1.550],
        },

    ]
    for arg in plot_args_list[:2]:
        plot_scatter_with_lines(**arg)

    # print(
    #     modelsDF.groupby('model_name').test_score.max() .sort_values(ascending=False)
    #     #
    # )



def main_summary():
    main_dirs = [
        'state5_obs3_GT',
        'state5_obs5_GT',
        'state7_obs3_GT',
        'state7_obs7_GT'
    ]
    DFs = []
    for main_dir in main_dirs:
        print(main_dir)
        path = os.path.join('all_models_validated',main_dir)
        print(path)
        DF,main_path = collater(main_dir=path)

        GT_state = int(main_dir.split('_')[0].split('state')[-1])
        GT_obs   = int(main_dir.split('_')[1].split('obs')[-1])
        DF['groundtruth_state_size'] = GT_state
        DF['groundtruth_observation_size'] = GT_obs

        DFs.append(DF)
    DF = pd.concat(DFs).reset_index()



    DFmodels = DF[DF.model_name!='Groundtruth']
    #DFmodels = DFmodels[DFmodels.n_components >= 10]
    groundtruth = DF[DF.model_name=='Groundtruth']
    best_models_idx = DFmodels.groupby(['groundtruth_state_size','groundtruth_observation_size','model_name'])['test_score'].idxmax()
    #print(best_models_idx.values)
    #print('best models ',len(DFmodels.loc[best_models_idx.values]))
    best_models_and_groundtruth = pd.concat([DFmodels.loc[best_models_idx.values],groundtruth])
    #print('GT',len(groundtruth))
    best_models_and_groundtruth['groundtruth_specifications'] = 'state'+best_models_and_groundtruth['groundtruth_state_size'].apply(str) + '_obs' + best_models_and_groundtruth['groundtruth_observation_size'].apply(str)
    #print(len(best_models_and_groundtruth.sort_values(by=['groundtruth_specifications'])))

    # fig,ax  = plt.subplots()
    # sns.scatterplot(x='groundtruth_specifications',y='test_score',hue='model_name',data = best_models_and_groundtruth,ax=ax,s=5)
    # fig.savefig('plots/summary_best_test_scores.pdf')
    # #plt.show()
    #
    fig,ax  = plt.subplots()
    sns.scatterplot(x='groundtruth_specifications',y='PR',hue='model_name',data = best_models_and_groundtruth,ax=ax,s=10)
    fig.tight_layout()
    fig.savefig('plots/summary_best_PR.pdf',dpi=200)
    #plt.show()
    #
    # fig,ax  = plt.subplots()
    # sns.scatterplot(x='groundtruth_specifications',y='test_self_consistency',hue='model_name',data = best_models_and_groundtruth,ax=ax)
    # fig.tight_layout()
    # fig.savefig('plots/summary_best_SC.png',dpi=160)


    DF_for_table = best_models_and_groundtruth[['groundtruth_specifications','test_score','model_name']].pivot(columns='groundtruth_specifications',index='model_name',values='test_score')

    df = DF_for_table.T.drop(columns=['Groundtruth'])
    df.columns = [c.replace('_',' ') for c in df.columns]
    df.index = [c.replace('_', ' ') for c in df.index]

    df_s = df.style.format("${:.4f}$")

    # loop through rows and find which column for each row has the highest value
    for row in df.index:
        col = df.loc[row].idxmax()
        # redo formatting for a specific cell
        df_s = df_s.format(lambda x: "$\mathbf{" + f'{x:.4f}' + "}$", subset=(row, col))

    print(df_s.to_latex())




if __name__ == '__main__':
    #collater()
    #main_summary()
    main()