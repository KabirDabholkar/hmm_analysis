import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
import pandas as pd
from sklearn.utils import check_random_state
from hmmlearn.hmm import GaussianHMM, CategoricalHMM
from hmmlearn.vhmm import VariationalCategoricalHMM
import pickle as pkl
from typing import Optional, List, Union, Dict, Callable
from functools import partial
import os
import re
from scipy.stats import pearsonr
from utils import make_path_if_not_exist, convert_to_int_string
import seaborn as sns
import matplotlib as mpl
from itertools import product

plt.rcParams.update({
    "text.usetex": True,  # Enable LaTeX
    "font.family": "serif",  # Set the font family
    "mathtext.fontset": "dejavuserif",
    "text.latex.preamble": r"\usepackage{amsmath} \usepackage{textcomp}"
})


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

def collater(
        main_dir='all_models_validated_finetuning/state5_obs5_eps0.1_emeps0.6_GT',
        sub_dir='models_traintrials2000',
):
    # options = ['augmented_mode','augmented_mode_long','augmented_with_shift_mode','augmented_with_shift_repeatwithoutshift_mode','vanilla_sliced_mode']#'vanilla_mode'
    # options = ['vanilla_sliced_mode','sliced_and_augmented_with_small_shifts']
    # options = ['training_vanilla', 'training_augmented_with_shift', 'training_augmented_with_shift_then_vanilla','training_augmented_with_shift_then_vanilla_frozen_te'] # 'training_augmented'
    # options = ['pretrain_vanilla_then_finetuning_emission_vanilla','pretrain_augmented_with_shift_then_finetuning_emission_vanilla']
    # options = ['training_vanilla']
    options = ['']
    main_path = os.path.join(main_dir, sub_dir)
    dir_names = [main_path + opt for opt in options]
    # dir_names += ['models_traintrials700_'+'vanilla_sliced_mode']
    all_DFs = []
    for dir_name in dir_names:
        files = os.listdir(dir_name)
        # print(files,len(files))
        for file_name in files:
            fullpath = os.path.join(dir_name, file_name)
            # print(dir_name,file_name,fullpath,os.path.splitext(fullpath)[1])
            if os.path.splitext(fullpath)[-1] == '.csv':
                DF = pd.read_csv(fullpath, index_col=None)
                all_DFs.append(DF)
    if sub_dir != '':
        for d in os.listdir(main_dir):
            if d[-4:] == '.csv':
                all_DFs += [pd.read_csv(os.path.join(main_dir, d), index_col=None)]  # .replace('_validated','')
    DF = pd.concat(all_DFs, ignore_index=True)
    DF = DF.sort_values(by=['model_name', 'n_components', 'model_id'])
    # print(DF['13-shot co-smoothing'])
    return DF, main_path


def plot_scatter_with_lines(
        x: Optional[str],
        y: Optional[str],
        data: pd.DataFrame,
        data_lines: Optional[pd.DataFrame],
        save_path: str,
        hue: Optional[str] = None,
        func1=sns.scatterplot,
        func2=None,
        figsize: Optional[tuple] = None,
        sortby: List[str] = [],
        xlabel: Optional[str] = None,
        ylabel: Optional[str] = None,
        label_fontsize: int = 13,
        xlim: List[Union[int, float]] = [],
        ylim: List[Union[int, float]] = [],
        hlines: List[Union[int, float]] = [],
        print_corrcoef: bool = False,
        zoom_inset: Optional[Dict] = None,
        data_arrows: Optional[pd.DataFrame] = None,
        arrow_text_fontsize: int = 10,
        legend_label_modifier: Optional[Callable] = None,
        legend=True,
):
    fig, axs = plt.subplots(figsize=figsize)
    all_axes = [axs]
    if zoom_inset is not None:
        ax_ins = axs.inset_axes(**zoom_inset)
        all_axes.append(ax_ins)

    for ax in all_axes:
        data_ = data.sort_values(by=sortby)
        func1(x=x, y=y, hue=hue, data=data_, ax=ax)
        if func2:
            func2(x=x, y=y, data=data_, ax=ax)
        if data_lines is not None:
            for color, (name_, x_, y_) in zip(['black', 'blue', 'red'], data_lines[['model_name', x, y]].values):
                ax.axhline(y_, ls='dashed', color=color, label=name_, lw=0.7)
                ax.axvline(x_, ls='dashed', color=color, lw=0.7)
        for i, ls in enumerate(hlines):
            ax.axhline(ls, color='C%d' % i, ls='dashed', lw=0.7)
        # if data_arrows is not None:
        #     for x_, y_, name_ in data_arrows[[x, y, 'model_name']].values:
        #         ax.annotate(
        #                 name_,
        #                 xy=(x_, y_),
        #                 xytext=(x_ + 0.1 * (ax.get_xlim()[1] - ax.get_xlim()[0]), y_ + 0.1 * (ax.get_ylim()[1] - ax.get_ylim()[0])),
        #                 #xytext=(ax.get_xlim()[1], y_),
        #                 fontsize=arrow_text_fontsize,
        #                 verticalalignment='center',
        #                 horizontalalignment='left',
        #                 arrowprops=dict(facecolor='black', shrink=0.05, width=0.5, headwidth=5))
        if data_arrows is not None:
            for x_, y_, color_ in data_arrows[[x, y, 'color']].values:
                ax.annotate('', xy=(x_, y_), xytext=(ax.get_xlim()[1], y_),
                            arrowprops=dict(facecolor=color_, edgecolor=color_, shrink=0.05, width=0.5, headwidth=5))

        handles, labels = ax.get_legend_handles_labels()
    if legend_label_modifier is not None:
        labels = [legend_label_modifier(l) for l in labels]
    if legend:
        axs.legend(handles, labels, fontsize=7, framealpha=0.3)
    axs.set_xlim(*xlim)
    axs.set_ylim(*ylim)
    axs.set_xlabel(x if xlabel is None else xlabel, fontsize=label_fontsize)
    axs.set_ylabel(y if ylabel is None else ylabel, fontsize=label_fontsize)
    if print_corrcoef:
        a, b = data[[x, y]].dropna().values.T
        corrcoef = np.corrcoef(a, b)[0, 1]
        ax = all_axes[0]
        ax.set_title(rf'$\rho=%.2f$' % corrcoef)
    if zoom_inset:
        ax_ins.set_xlabel(None)
        ax_ins.set_ylabel(None)
        legend = ax_ins.legend()
        legend.remove()
        axs.indicate_inset_zoom(ax_ins, edgecolor="black")
    fig.tight_layout()
    if not os.path.exists(os.path.dirname(save_path)):
        os.makedirs(os.path.dirname(save_path))
    fig.savefig(save_path, dpi=300, transparent=save_path.split('.')[-1] == 'pdf')
    print(f'saved plot to {save_path}')
    plt.close()


def main():
    # DF = pd.read_csv('plots/collated.csv',index_col=0)

    DF, main_dir = collater(
        # main_dir='all_models_validated_v2/teacher_state5_poisson_partial_eps0.1_length10',
        # sub_dir='models_traintrials500'
        # main_dir='/home/kabird/ray_results/all_models_validated_v2/teacher_state4_bernoulli_partial_eps0.01_length10',
        # sub_dir='combined3'
        # sub_dir = 'models_traintrials2000_numheldin20_numheldout100_numreallyheldout50_v3_entropy'

        main_dir='/Users/kabir/Documents/code/hmm_analysis/all_models_validated_test/teacher_state4_bernoulli_partial_eps0.01_length10',
        sub_dir='models_traintrials2000',

        # main_dir = '/home/kabird/ray_results/all_models_validated_v2/teacher_state4_bernoulli_partial_eps0.01_length10_traintrials2000_numheldin5_numheldout5_numreallyheldout50',
        # sub_dir='models'

        # main_dir='/home/kabird/ssm_analysis/results/teacher_LinearGaussianSSM_statedim4_emissiondim35_train20',
        # sub_dir=''
    )

    # plot_hessian_FI(
    #     main_dir='all_models_validated_v2/teacher_state4_bernoulli_partial_eps0.01_length10',
    #     sub_dir=''
    # )
    DF = DF.replace(to_replace='Groundtruth', value='Ground truth')
    # print(DF['params_learned'])
    DF.dropna(subset=['n_components', 'model_name'], inplace=True)
    # DF['n_components'] = pd.to_numeric(DF['n_components'])#.astype('Int64') #.apply(lambda x: int(x) if (~np.isnan(x) and ~np.isinf(x)) else x)
    print('n_components', DF.n_components)
    DF['params_learned'][DF['params_learned'].isna()] = 'stl'
    # print(DF[DF['model_name']=='Ground truth'].n_components)

    # DF['score'] = DF['test_score']
    # DF.loc[DF.model_name == 'Ground truth', 'test_self_consistency'] = DF.loc[DF.model_name == 'Ground truth', 'test_self_consistency_modified_pi']
    if 'test_self_consistency' in DF.columns:
        DF['self_consistency'] = DF['test_self_consistency']

    # DF['6-shot co-smoothing delta'] = DF['6-shot co-smoothing'] - DF['original co-smoothing']
    DF['log_linear_decoder_teacher->student'] = np.log(np.maximum(DF['linear_decoder_teacher->student'], 1e-10))
    # DF['decoder_student->teacher ratio'] = DF['decoder_student->teacher'] / DF['decoder_student->teacher shuffled']
    # DF['decoder_teacher->student ratio'] = DF['decoder_teacher->student'] / DF['decoder_teacher->student shuffled']
    # sns.scatterplot(data = DF)
    select_special = (DF.model_name == 'Ground truth') | (DF.model_name == 'Groundtruth_extraneous')
    select_special2 = (DF.model_name == 'Ground truth')
    modelsDF = DF[~select_special]

    modelsGT = DF[select_special2]

    # modelsDF = modelsDF[modelsDF.score>-2.31]
    # modelsDF = modelsDF[modelsDF.score > -1.70]
    # modelsDF = modelsDF[modelsDF.test_score > -1.05]
    # modelsDF = modelsDF[modelsDF.test_score > -2.0]
    # modelsDF = modelsDF[modelsDF.test_score > modelsGT.minus_test_entropy.values[0]]
    # print(modelsDF.columns)
    # print(modelsDF[modelsDF['original co-smoothing']<-5][modelsDF['original co-smoothing']>0.48].n_components)
    # print(os.path.join('plots', main_dir, 'test_PR_vs_score.png'))
    # print(DF[DF.model_name == 'Ground truth'])

    cosmoothing_columns = [c for c in modelsDF.columns if '-shot co-smoothing' in c]  # + ['original co-smoothing']

    # print(modelsDF.pivot(index=['model_id','n_components'],columns=cosmoothing_columns))

    # print(
    #     modelsDF['decoder_student->teacher'] #['co-smoothing']
    #     # models_k_shot['n_components'].apply(str).type
    #     # models_k_shot['model_id']
    # )

    best_cosmoothing_models = modelsDF[
        modelsDF['original co-smoothing'] > (modelsDF['original co-smoothing'].max() - 0.05)]
    best_cosmoothing_models = best_cosmoothing_models[best_cosmoothing_models.model_name == 'dynamax_fit_em']
    # print(
    #     'best',
    #     best_cosmoothing_models.columns,
    #     print(best_cosmoothing_models),
    #     best_cosmoothing_models.sort_values(by='linear_decoder_teacher->student').dropna(),
    # )
    modelsDF['linear_decoder_sum'] = (
                modelsDF['linear_decoder_teacher->student'] + modelsDF['linear_decoder_student->teacher'])
    modelsDF_sorted = modelsDF.sort_values(by='linear_decoder_sum', ascending=True, inplace=False)
    # print('Best decoding both ways (good example):',modelsDF_sorted.head(5))
    modelsDF_sorted = modelsDF[
        modelsDF['original co-smoothing'] > (modelsDF['original co-smoothing'].max() - 0.002)].sort_values(
        by='linear_decoder_teacher->student',
        ascending=False,
        inplace=False)
    # print('Good co-smoothing but worst decoding teacher->student (bad example):',modelsDF_sorted.head(5)[['model_id','n_components']])

    # best_and_worst = modelsDF[modelsDF['original co-smoothing'] > 100]
    # best_and_worst['color'] = []
    #
    # best_and_worst = modelsDF.set_index(['n_components', 'model_id'])
    # best_and_worst = best_and_worst.loc[[(10, 10), (8, 63)]]
    # print('best_and_worst', best_and_worst)
    # best_and_worst['model_name'] = ['``Good"\nmodel', '``Bad"\nmodel']
    # best_and_worst['color'] = ['green', 'red']
    #
    # best_4_state = modelsDF[modelsDF.n_components == 4].sort_values(by='original co-smoothing', ascending=False)
    # print(
    #     'best 4 state models',
    #     best_4_state.head(3)[['original co-smoothing', 'model_id']]
    # )
    # modelsGT = pd.concat([best_and_worst,modelsGT],axis=0)

    # format = "pdf"
    format = "png"
    figsize = (4,3)
    compact_figsize = np.array([5, 4]) * 0.5

    # print(modelsDF[['original co-smoothing','13-shot co-smoothing']])
    plot_args_list = [
        # {
        #     'x': 'original co-smoothing',
        #     'y': '13-shot co-smoothing',  # -angular
        #     'hue': 'n_components',  # 'unique_id',
        #     # 'hue' : 'iterations',
        #     # 'hue'   : 'model_name',
        #     'sortby' : ['model_name','n_components'],
        #     'data': modelsDF, # modelsDF[modelsDF['original co-smoothing'] > (modelsDF['original co-smoothing'].max() - 0.02)],
        #     'data_lines': modelsGT,
        #     # 'data_lines': None,
        #     'save_path': os.path.join('plots', main_dir, '13-shot_vs_original_cosmoothing.png'),
        #     'func1': partial(sns.scatterplot, style='model_name'),
        #     # 'func2': partial(sns.lineplot, hue='params_learned', legend=True, estimator=None, alpha=1, errorbar=None,
        #     #                  markers=False),
        #     # 'func2': partial(sns.lineplot, hue='params_learned', legend=True, estimator=None, alpha=1, errorbar=None,
        #     #      markers=False),
        #     # 'xlim': (0.44, ),
        #     # 'ylim': (0.37, ),
        #     # 'ylabel': r'Consistency KL divergence',
        #     # 'ylabel': r'teacher $\mapsto$ student',
        #     # 'hlines': list(modelsDF.groupby('model_name').test_score.max())
        #     'zoom_inset': {
        #         'bounds': [0.1, 0.4, 0.4, 0.4],
        #         'xlim': (0.44, 0.45),
        #         'ylim': (0.37, 0.38),
        #     },
        # },
        # {
        #     'x': 'original co-smoothing',
        #     'y': '6-shot co-smoothing',  # -angular
        #     'hue': 'n_components',  # 'unique_id',
        #     # 'hue': 'iterations',
        #     # 'hue': 'model_name',
        #     'sortby': ['model_name', 'n_components'],  # 'sortby': ['params_learned', 'iterations'],
        #     'data': modelsDF,
        #     'figsize': figsize,
        #     # modelsDF[modelsDF['original co-smoothing'] > (modelsDF['original co-smoothing'].max() - 0.02)],
        #     'data_lines': modelsGT,
        #     'data_arrows': best_and_worst,
        #     'arrow_text_fontsize': 7,
        #     # 'data_lines': None,
        #     'save_path': os.path.join('plots', main_dir, f'6-shot_vs_original_cosmoothing.{format}'),
        #     # 'func1' : partial(sns.scatterplot,style='model_name'),
        #     'func1': partial(sns.scatterplot),
        #     # 'func2': partial(sns.lineplot, hue='params_learned', legend=True, estimator=None, alpha=1, errorbar=None,
        #     #                  markers=False),
        #     'xlim': (0.0, 0.5),
        #     'ylim': (0.0, 0.3),
        #     'xlabel': 'S co-smoothing',
        #     'ylabel': 'S $6$-shot co-smoothing',
        #     # 'ylabel': r'Consistency KL divergence',
        #     # 'ylabel': r'teacher $\mapsto$ student',
        #     # 'hlines': list(modelsDF.groupby('model_name').test_score.max())
        #     'zoom_inset': {
        #         'bounds': [0.2, 0.2, 0.3, 0.3],
        #         'xlim': (0.43, 0.455),
        #         'ylim': (0.11, 0.28),
        #     },
        #     'print_corrcoef': True,
        # },
        # {
        #     'x': 'original co-smoothing',
        #     'y': 'similarity.procrustes',
        #     'hue': 'n_components',  # 'unique_id',
        #     # 'hue': 'iterations',
        #     # 'hue': 'model_name',
        #     'sortby' : ['model_name','n_components'], # 'sortby': ['params_learned', 'iterations'],
        #     'data': modelsDF,
        #     # modelsDF[modelsDF['original co-smoothing'] > (modelsDF['original co-smoothing'].max() - 0.02)],
        #     'data_lines': modelsGT,
        #     # 'data_lines': None,
        #     'save_path': os.path.join('plots', main_dir, 'procrustes_vs_original_cosmoothing.png'),
        #     'func1' : partial(sns.scatterplot,style='model_name'),
        #     # 'func2': partial(sns.lineplot, hue='params_learned', legend=True, estimator=None, alpha=1, errorbar=None,
        #     #                  markers=False),
        #     'xlim': (0.0,0.5),
        #     # 'ylim': (0.0,1.1),
        #     # 'ylabel': r'Consistency KL divergence',
        #     # 'ylabel': r'teacher $\mapsto$ student',
        #     # 'hlines': list(modelsDF.groupby('model_name').test_score.max())
        #     # 'zoom_inset': {
        #     #     'bounds': [0.6, 0.06, 0.3, 0.3],
        #     #     'xlim': (0.42,0.455),
        #     #     'ylim': (0.15,0.28),
        #     # },
        # },
        # {
        #     'x': '6-shot co-smoothing',
        #     'y': 'similarity.procrustes',
        #     'hue': 'n_components',  # 'unique_id',
        #     # 'hue': 'iterations',
        #     # 'hue': 'model_name',
        #     'sortby' : ['model_name','n_components'], # 'sortby': ['params_learned', 'iterations'],
        #     'data': modelsDF,
        #     # modelsDF[modelsDF['original co-smoothing'] > (modelsDF['original co-smoothing'].max() - 0.02)],
        #     'data_lines': modelsGT,
        #     # 'data_lines': None,
        #     'save_path': os.path.join('plots', main_dir, 'procrustes_vs_6-shot_cosmoothing.png'),
        #     'func1' : partial(sns.scatterplot,style='model_name'),
        #     # 'func2': partial(sns.lineplot, hue='params_learned', legend=True, estimator=None, alpha=1, errorbar=None,
        #     #                  markers=False),
        #     'xlim': (0.0,0.5),
        #     # 'ylim': (0.0,1.1),
        #     # 'ylabel': r'Consistency KL divergence',
        #     # 'ylabel': r'teacher $\mapsto$ student',
        #     # 'hlines': list(modelsDF.groupby('model_name').test_score.max())
        #     # 'zoom_inset': {
        #     #     'bounds': [0.6, 0.06, 0.3, 0.3],
        #     #     'xlim': (0.42,0.455),
        #     #     'ylim': (0.15,0.28),
        #     # },
        # },
        # {
        #     'x': '6-shot co-smoothing',  # '6-shot co-smoothing',
        #     'y': 'linear_decoder_teacher->student',  # -angular
        #     'hue': 'n_components',  # 'unique_id',
        #     # 'hue': 'model_id',
        #     # 'hue': 'model_name',
        #     'sortby': ['model_name', 'n_components'],  # 'sortby': ['params_learned', 'iterations'],
        #     'data': modelsDF[modelsDF['original co-smoothing'] > (modelsDF['original co-smoothing'].max() - 0.0013)],
        #     'data_lines': modelsGT,
        #     'data_arrows': best_and_worst,
        #     'arrow_text_fontsize': 7,
        #     'figsize': figsize,
        #     # 'data_lines': None,
        #     'save_path': os.path.join('plots', main_dir, f'decoder_teacher->student_6-shot.{format}'),
        #     # 'func1': partial(sns.scatterplot, style='model_name'),
        #     'func1': partial(sns.scatterplot),
        #     # 'func2': partial(sns.lineplot, hue='params_learned', legend=True, estimator=None, alpha=1, errorbar=None,
        #     #                  markers=False),
        #     'xlim': (0.0, 0.3),
        #     # 'ylim': (0.0,0.1),
        #     # 'ylabel': r'Consistency KL divergence',
        #     # 'ylabel': r'teacher $\mapsto$ student',
        #     'xlabel': r'S $6$-shot co-smoothing',
        #     # 'ylabel': r'Decoding error teacher $\rightarrow$ student',
        #     'ylabel': r'T$\rightarrow$S error',
        #     # 'hlines': list(modelsDF.groupby('model_name').test_score.max())
        #     # 'zoom_inset': {
        #     #     'bounds': [0.6, 0.06, 0.3, 0.3],
        #     #     'xlim': (0.42, 0.455),
        #     #     'ylim': (0.15, 0.28),
        #     # },
        #     'print_corrcoef': True,
        #
        # },
        # {
        #     'x': '6-shot co-smoothing',  # '6-shot co-smoothing',
        #     'y': 'linear_decoder_teacher->student',  # -angular
        #     'hue': 'n_components',  # 'unique_id',
        #     # 'hue': 'model_id',
        #     # 'hue': 'model_name',
        #     'sortby': ['model_name', 'n_components'],  # 'sortby': ['params_learned', 'iterations'],
        #     'data': modelsDF[modelsDF['original co-smoothing'] > (modelsDF['original co-smoothing'].max() - 0.002)],
        #     'data_lines': modelsGT,
        #     'figsize': figsize,
        #     # 'data_lines': None,
        #     'save_path': os.path.join('plots', main_dir, f'decoder_teacher->student_6-shot_best.{format}'),
        #     # 'func1': partial(sns.scatterplot, style='model_name'),
        #     'func1': partial(sns.scatterplot),
        #     # 'func2': partial(sns.lineplot, hue='params_learned', legend=True, estimator=None, alpha=1, errorbar=None,
        #     #                  markers=False),
        #     'xlim': (0.0, 0.3),
        #     # 'ylim': (0.0,0.1),
        #     # 'ylabel': r'Consistency KL divergence',
        #     # 'ylabel': r'teacher $\mapsto$ student',
        #     'xlabel': r'$6$-shot co-smoothing',
        #     'ylabel': r'Decoding error teacher $\rightarrow$ student',
        #     # 'hlines': list(modelsDF.groupby('model_name').test_score.max())
        #     # 'zoom_inset': {
        #     #     'bounds': [0.6, 0.06, 0.3, 0.3],
        #     #     'xlim': (0.42, 0.455),
        #     #     'ylim': (0.1, 0.28),
        #     # },
        #     'print_corrcoef': True,
        # },
        {
            'x': 'original co-smoothing',  # '6-shot co-smoothing',
            'y': 'linear_decoder_student->teacher',  # -angular
            # 'hue': None,
            'hue': 'n_components',  # 'unique_id',
            # 'hue': 'iterations',
            # 'hue': 'model_name',
            'sortby': ['model_name', 'n_components'],  # 'sortby': ['params_learned', 'iterations'],
            'data': modelsDF,  # [modelsDF['original co-smoothing'] > (modelsGT['original co-smoothing'].max() - 0.05)],
            'data_lines': modelsGT,
            # 'data_lines': None,
            # 'data_arrows': best_and_worst,
            'arrow_text_fontsize': 7,
            'figsize': figsize,
            'save_path': os.path.join('plots', main_dir, f'decoder_student->teacher_original_simple.{format}'),
            'func1': partial(sns.scatterplot, legend=True),  # style='model_name'
            # 'func2': partial(sns.lineplot, hue='params_learned', legend=True, estimator=None, alpha=1, errorbar=None,
            #                  markers=False),
            'xlim': (0.0, 0.47),
            # 'ylim': (0.0, 0.43),
            # 'ylabel': r'Consistency KL divergence',
            # 'ylabel': r'Decoding error student $\rightarrow$ teacher',
            'ylabel': r'S$\rightarrow$T error',
            # 'hlines': list(modelsDF.groupby('model_name').test_score.max())
            # 'ylabel': 'Error of decoding from teacher',
            'xlabel': 'S co-smoothing',
            'label_fontsize': 13,
            # 'zoom_inset': {
            #     'bounds': [0.35, 0.16, 0.4, 0.4],
            #     'xlim': (0.42, 0.452),
            #     'ylim': (0.002, 0.06),
            # },
            'print_corrcoef': True,
            'legend_label_modifier': lambda x: fr'$M={convert_to_int_string(x)}$' if re.match(r'^-?\d+(\.\d+)?$',
                                                                                              x) else x,
            ## convert to int and prefix with 'M=' only if numeric
        },

        {
            'x': 'original co-smoothing',  # '6-shot co-smoothing',
            'y': 'linear_decoder_teacher->student',  # -angular
            # 'hue': None,
            'hue': 'n_components',  # 'unique_id',
            # 'hue': 'iterations',
            # 'hue': 'model_name',
            'sortby': ['model_name', 'n_components'],  # 'sortby': ['params_learned', 'iterations'],
            'data': modelsDF,  # [modelsDF['original co-smoothing'] > (modelsGT['original co-smoothing'].max() - 0.05)],
            'data_lines': modelsGT,
            # 'data_arrows': best_and_worst,
            'arrow_text_fontsize': 7,
            # 'data_lines': None,
            'figsize': figsize,
            'save_path': os.path.join('plots', main_dir, f'decoder_teacher->student_original_simple.{format}'),
            # 'func1': partial(sns.scatterplot, style='model_name',legend=True),
            'func1': partial(sns.scatterplot, legend=True),
            # 'func2': partial(sns.lineplot, hue='params_learned', legend=True, estimator=None, alpha=1, errorbar=None,
            #                  markers=False),
            'xlim': (0.0, 0.47),
            # 'ylim': (0.0, 0.43),
            # 'ylim': (0.0, 0.1),
            # 'ylabel': r'Consistency KL divergence',
            # 'ylabel': r'Decoding error teacher $\rightarrow$ student',
            'ylabel': r'T$\rightarrow$S error',
            # 'hlines': list(modelsDF.groupby('model_name').test_score.max())
            # 'ylabel': 'Error of decoding from teacher',
            'xlabel': 'S co-smoothing',

            'label_fontsize': 13,
            # 'zoom_inset': {
            #     'bounds': [0.25, 0.16, 0.3, 0.3],
            #     'xlim': (0.44, 0.452),
            #     'ylim': (0.0, 0.38),
            # },
            'print_corrcoef': True,
            'legend_label_modifier': lambda x: fr'$M={convert_to_int_string(x)}$' if re.match(r'^-?\d+(\.\d+)?$',
                                                                                              x) else x,
            ## convert to int and prefix with 'M=' only if numeric
        },
        # {
        #     'x': '6-shot co-smoothing',  # '6-shot co-smoothing',
        #     'y': 'linear_decoder_teacher->student',  # -angular
        #     'hue': None,
        #     # 'hue': 'n_components',  # 'unique_id',
        #     # 'hue': 'iterations',
        #     # 'hue': 'model_name',
        #     'sortby': ['model_name', 'n_components'],  # 'sortby': ['params_learned', 'iterations'],
        #     'data': modelsDF[modelsDF['original co-smoothing'] > (modelsGT['original co-smoothing'].max() - 0.002)],
        #     'data_lines': modelsGT,
        #     # 'data_lines': None,
        #     'figsize': figsize,
        #     'save_path': os.path.join('plots', main_dir, f'decoder_teacher->student_6-shot_simple.{format}'),
        #     'func1': partial(sns.scatterplot, style='model_name', legend=False),
        #     # 'func2': partial(sns.lineplot, hue='params_learned', legend=True, estimator=None, alpha=1, errorbar=None,
        #     #                  markers=False),
        #     'xlim': (0.0, 0.3),
        #     # 'ylim': (0.0, 0.1),
        #     # 'ylabel': r'Consistency KL divergence',
        #     # 'ylabel': r'teacher $\mapsto$ student',
        #     # 'hlines': list(modelsDF.groupby('model_name').test_score.max())
        #     'ylabel': r'Decoding error: teacher $\rightarrow$ student',
        #     'xlabel': '6-shot co-smoothing',
        #     'label_fontsize': 13,
        #     # 'zoom_inset': {
        #     #     'bounds': [0.3, 0.3, 0.3, 0.3],
        #     #     'xlim': (0.24, 0.26),
        #     #     'ylim': (0.0, 0.025),
        #     # },
        #     'print_corrcoef': True,
        # },
        ########### compact scatter plots

        {
            'x': 'original co-smoothing',  # '6-shot co-smoothing',
            'y': 'linear_decoder_student->teacher',  # -angular
            'hue': None,
            # 'hue': 'n_components',  # 'unique_id',
            # 'hue': 'iterations',
            # 'hue': 'model_name',
            'sortby': ['model_name', 'n_components'],  # 'sortby': ['params_learned', 'iterations'],
            'data': modelsDF,  # [modelsDF['original co-smoothing'] > (modelsGT['original co-smoothing'].max() - 0.05)],
            'data_lines': modelsGT,
            # 'data_lines': None,
            # 'data_arrows': best_and_worst,
            'arrow_text_fontsize': 7,
            'figsize': compact_figsize,
            'save_path': os.path.join('plots', main_dir, f'decoder_student->teacher_original_compact.{format}'),
            'func1': partial(sns.scatterplot, legend=False, size=1, alpha=0.7),  # style='model_name'
            # 'func2': partial(sns.lineplot, hue='params_learned', legend=True, estimator=None, alpha=1, errorbar=None,
            #                  markers=False),
            'xlim': (0.0, 0.47),
            # 'ylim': (0.0, 0.43),
            # 'ylabel': r'Consistency KL divergence',
            # 'ylabel': r'Decoding error student $\rightarrow$ teacher',
            'ylabel': r'$\mathcal D_{\text{S}\rightarrow\text{T}}$',
            # 'hlines': list(modelsDF.groupby('model_name').test_score.max())
            # 'ylabel': 'Error of decoding from teacher',
            'xlabel': r'$\mathcal Q_\text{S}$',
            'label_fontsize': 12,
            # 'zoom_inset': {
            #     'bounds': [0.35, 0.33, 0.4, 0.4],
            #     'xlim': (0.42, 0.452),
            #     'ylim': (0.002, 0.06),
            # },
            'print_corrcoef': True,
            'legend_label_modifier': lambda x: fr'$M={convert_to_int_string(x)}$' if re.match(r'^-?\d+(\.\d+)?$',
                                                                                              x) else x,
            ## convert to int and prefix with 'M=' only if numeric
            'legend': False,
        },

        {
            'x': 'original co-smoothing',  # '6-shot co-smoothing',
            'y': 'linear_decoder_teacher->student',  # -angular
            'hue': None,
            # 'hue': 'n_components',  # 'unique_id',
            # 'hue': 'iterations',
            # 'hue': 'model_name',
            'sortby': ['model_name', 'n_components'],  # 'sortby': ['params_learned', 'iterations'],
            'data': modelsDF,  # [modelsDF['original co-smoothing'] > (modelsGT['original co-smoothing'].max() - 0.05)],
            'data_lines': modelsGT,
            # 'data_arrows': best_and_worst,
            'arrow_text_fontsize': 7,
            # 'data_lines': None,
            'figsize': compact_figsize,
            'save_path': os.path.join('plots', main_dir, f'decoder_teacher->student_original_compact.{format}'),
            # 'func1': partial(sns.scatterplot, style='model_name',legend=True),
            'func1': partial(sns.scatterplot, legend=False, size=1, alpha=0.7),
            # 'func2': partial(sns.lineplot, hue='params_learned', legend=True, estimator=None, alpha=1, errorbar=None,
            #                  markers=False),
            'xlim': (0.0, 0.47),
            # 'ylim': (0.0, 0.43),
            # 'ylim': (0.0, 0.1),
            # 'ylabel': r'Consistency KL divergence',
            # 'ylabel': r'Decoding error teacher $\rightarrow$ student',
            'ylabel': r'$\mathcal D_{\text{T}\rightarrow\text{S}}$',
            # 'hlines': list(modelsDF.groupby('model_name').test_score.max())
            # 'ylabel': 'Error of decoding from teacher',
            'xlabel': r'$\mathcal Q_\text{S}$',

            'label_fontsize': 12,
            # 'zoom_inset': {
            #     'bounds': [0.25, 0.26, 0.4, 0.4],
            #     'xlim': (0.44, 0.452),
            #     'ylim': (0.0, 0.38),
            # },
            'print_corrcoef': True,
            'legend_label_modifier': lambda x: fr'$M={convert_to_int_string(x)}$' if re.match(r'^-?\d+(\.\d+)?$',
                                                                                              x) else x,
            ## convert to int and prefix with 'M=' only if numeric
            'legend': False,
        },
        # {
        #     'x': '6-shot co-smoothing',  # '6-shot co-smoothing',
        #     'y': 'linear_decoder_teacher->student',  # -angular
        #     'hue': None,
        #     # 'hue': 'n_components',  # 'unique_id',
        #     # 'hue': 'iterations',
        #     # 'hue': 'model_name',
        #     'sortby': ['model_name', 'n_components'],  # 'sortby': ['params_learned', 'iterations'],
        #     'data': modelsDF[modelsDF['original co-smoothing'] > (modelsGT['original co-smoothing'].max() - 0.001)],
        #     'data_lines': modelsGT,
        #     'data_arrows': best_and_worst,
        #     'arrow_text_fontsize': 7,
        #     # 'data_lines': None,
        #     'figsize': compact_figsize,
        #     'save_path': os.path.join('plots', main_dir, f'decoder_teacher->student_6-shot_compact.{format}'),
        #     'func1': partial(sns.scatterplot, legend=False, size=1, alpha=0.7),  # style='model_name'
        #     # 'func2': partial(sns.lineplot, hue='params_learned', legend=True, estimator=None, alpha=1, errorbar=None,
        #     #                  markers=False),
        #     'xlim': (0.0, 0.28),
        #     # 'ylim': (0.0, 0.1),
        #     # 'ylabel': r'Consistency KL divergence',
        #     # 'ylabel': r'teacher $\mapsto$ student',
        #     # 'hlines': list(modelsDF.groupby('model_name').test_score.max())
        #     'ylabel': r'$\mathcal D_{\text{T}\rightarrow\text{S}}$',
        #     'xlabel': r'$\langle\mathcal Q^k_\text{S}\rangle$',
        #     'label_fontsize': 12,
        #     # 'zoom_inset': {
        #     #     'bounds': [0.3, 0.3, 0.3, 0.3],
        #     #     'xlim': (0.24, 0.26),
        #     #     'ylim': (0.0, 0.025),
        #     # },
        #     'print_corrcoef': True,
        #     'legend': False,
        # },

        # {
        #     'x': '3-shot co-smoothing',
        #     'y': 'consistency_teacher->student',  # -angular
        #     'hue': 'n_components',  # 'unique_id',
        #     'data': modelsDF[modelsDF['original co-smoothing'] > (modelsDF['original co-smoothing'].max() - 0.02)],
        #     'data_lines': modelsGT,
        #     # 'data_lines': None,
        #     'save_path': os.path.join('plots', main_dir, 'consistency_teacherstudent_vs_3shotcosmoothing.png'),
        #     # 'func1': partial(sns.lineplot, legend=False, units='unique_id', estimator=None, alpha=0.5, errorbar=None,
        #     #                  markers=False),
        #     'xlim': (-.3, 0.4),
        #     # 'ylim': (-10, 3),
        #     'ylabel': r'Consistency KL divergence',
        #     # 'ylabel': r'teacher $\mapsto$ student',
        #     # 'hlines': list(modelsDF.groupby('model_name').test_score.max())
        #     # 'zoom_inset': {
        #     #     'bounds': [0.1, 0.4, 0.4, 0.4],
        #     #     'xlim': (0.30, 0.33),
        #     #     'ylim': (0.0, 0.12),
        #     # },
        # },
        #
        # {
        #     'x': '8-shot co-smoothing',
        #     'y': 'consistency_teacher->student',  # -angular
        #     'hue': 'n_components',  # 'unique_id',
        #     'data': modelsDF[modelsDF['original co-smoothing'] > (modelsDF['original co-smoothing'].max() - 0.02)],
        #     'data_lines': modelsGT,
        #     # 'data_lines': None,
        #     'save_path': os.path.join('plots', main_dir, 'consistency_teacherstudent_vs_8shotcosmoothing.png'),
        #     # 'func1': partial(sns.lineplot, legend=False, units='unique_id', estimator=None, alpha=0.5, errorbar=None,
        #     #                  markers=False),
        #     'xlim': (0.0, 0.4),
        #     # 'ylim': (-10, 3),
        #     'ylabel': r'Consistency KL divergence',
        #     # 'ylabel': r'teacher $\mapsto$ student',
        #     # 'hlines': list(modelsDF.groupby('model_name').test_score.max())
        #     # 'zoom_inset': {
        #     #     'bounds': [0.1, 0.4, 0.4, 0.4],
        #     #     'xlim': (0.30, 0.33),
        #     #     'ylim': (0.0, 0.12),
        #     # },
        # },
        #
        # {
        #     'x': '13-shot co-smoothing',
        #     'y': 'consistency_teacher->student',  # -angular
        #     'hue': 'n_components',  # 'unique_id',
        #     'data': modelsDF, #[modelsDF['original co-smoothing'] > (modelsDF['original co-smoothing'].max() - 0.02)],
        #     'data_lines': modelsGT,
        #     # 'data_lines': None,
        #     'save_path': os.path.join('plots', main_dir, 'consistency_teacherstudent_vs_13shotcosmoothing.png'),
        #     # 'func1': partial(sns.lineplot, legend=False, units='unique_id', estimator=None, alpha=0.5, errorbar=None,
        #     #                  markers=False),
        #     # 'xlim': (0.0, 0.5),
        #     # 'ylim': (-10, 3),
        #     'ylabel': r'Consistency KL divergence',
        #     # 'ylabel': r'teacher $\mapsto$ student',
        #     # 'hlines': list(modelsDF.groupby('model_name').test_score.max())
        #     # 'zoom_inset': {
        #     #     'bounds': [0.1, 0.4, 0.4, 0.4],
        #     #     'xlim': (0.35, 0.38),
        #     #     'ylim': (0.0, 0.12),
        #     # },
        # },
        # # consistency
        # {
        #     'x': '17-shot co-smoothing',
        #     'y': 'consistency_teacher->student',  # -angular
        #     'hue': 'n_components',  # 'unique_id',
        #     'data': modelsDF[modelsDF['original co-smoothing'] > (modelsDF['original co-smoothing'].max() - 0.05)],
        #     'data_lines': modelsGT,
        #     # 'data_lines': None,
        #     'save_path': os.path.join('plots', main_dir, 'consistency_teacherstudent_vs_17shotcosmoothing.png'),
        #     # 'func1': partial(sns.lineplot, legend=False, units='unique_id', estimator=None, alpha=0.5, errorbar=None,
        #     #                  markers=False),
        #     # 'xlim': (0.0, 0.5),
        #     # 'ylim': (-10, 3),
        #     'ylabel': r'Consistency KL divergence',
        #     # 'ylabel': r'teacher $\mapsto$ student',
        #     # 'hlines': list(modelsDF.groupby('model_name').test_score.max())
        #     # 'zoom_inset': {
        #     #     'bounds': [0.1, 0.4, 0.4, 0.4],
        #     #     'xlim': (0.37, 0.4),
        #     #     'ylim'    : (0.0,0.12),
        #     # },
        # },
        # # consistency
        # {
        #     'x': 'original co-smoothing',
        #     'y': 'consistency_teacher->student',  # -angular
        #     'hue': 'n_components',  # 'unique_id',
        #     'data': modelsDF,  # [modelsDF['original co-smoothing'] > (modelsDF['original co-smoothing'].max() - 0.1)],
        #     'data_lines': modelsGT,
        #     # 'data_lines': None,
        #     'save_path': os.path.join('plots', main_dir, 'consistency_teacherstudent_vs_originalcosmoothing.png'),
        #     # 'func1': partial(sns.lineplot, legend=False, units='unique_id', estimator=None, alpha=0.5, errorbar=None,
        #     #                  markers=False),
        #     # 'xlim': (0,0.5),
        #     # 'ylim': (0,),
        #     'ylabel': r'Consistency KL divergence',
        #     # 'ylabel': r'teacher $\mapsto$ student',
        #     # 'hlines': list(modelsDF.groupby('model_name').test_score.max())
        #     'zoom_inset':{
        #         'bounds'  : [0.3, 0.4, 0.4, 0.4],
        #         # 'xlim'    : (0.435, 0.441),
        #         'xlim'    : (modelsDF['original co-smoothing'].max() - 2e-4, modelsDF['original co-smoothing'].max()+2e-4),
        #         # 'ylim'    : (0.0,0.12),
        #     },
        # },

        # # decoding
        # {
        #     'x': '22-shot co-smoothing',
        #     'y': 'MI_student->teacher',  # -angular
        #     'hue': 'n_components',  # 'unique_id',
        #     'data': modelsDF,  # [modelsDF['original co-smoothing'] > (modelsDF['original co-smoothing'].max() - 0.1)],
        #     'data_lines': modelsGT,
        #     # 'data_lines': None,
        #     'save_path': os.path.join('plots', main_dir, 'MI_studentteacher_vs_22shotcosmoothing.png'),
        #     # 'func1': partial(sns.lineplot, legend=False, units='unique_id', estimator=None, alpha=0.5, errorbar=None,
        #     #                  markers=False),
        #     'xlim': (-1, 0.6),
        #     'ylim': (-10, 3),
        #     'ylabel': r'student $\mapsto$ teacher',
        #     # 'ylabel': r'teacher $\mapsto$ student',
        #     # 'hlines': list(modelsDF.groupby('model_name').test_score.max())
        # },
        # # decoding
        # {
        #     'x': 'original co-smoothing',
        #     'y': 'MI_student->teacher',  # -angular
        #     'hue': 'n_components',  # 'unique_id',
        #     'data': modelsDF,  # [modelsDF['original co-smoothing'] > (modelsDF['original co-smoothing'].max() - 0.1)],
        #     'data_lines': modelsGT,
        #     # 'data_lines': None,
        #     'save_path': os.path.join('plots', main_dir, 'MI_studentteacher_vs_originalcosmoothing.png'),
        #     # 'func1': partial(sns.lineplot, legend=False, units='unique_id', estimator=None, alpha=0.5, errorbar=None,
        #     #                  markers=False),
        #     'xlim': (0.0, 0.6),
        #     'ylim': (-3.0, 3),
        #     'ylabel': r'student $\mapsto$ teacher',
        #     # 'ylabel': r'teacher $\mapsto$ student',
        #     # 'hlines': list(modelsDF.groupby('model_name').test_score.max())
        # },
        # # decoding
        # {
        #     'x': 'original co-smoothing',
        #     'y': 'MI_teacher->student',  # -angular
        #     'hue': 'n_components',  # 'unique_id',
        #     'data': modelsDF,  # [modelsDF['original co-smoothing'] > (modelsDF['original co-smoothing'].max() - 0.1)],
        #     'data_lines': modelsGT,
        #     # 'data_lines': None,
        #     'save_path': os.path.join('plots', main_dir, 'MI_teacherstudent_vs_originalcosmoothing.png'),
        #     # 'func1': partial(sns.lineplot, legend=False, units='unique_id', estimator=None, alpha=0.5, errorbar=None,
        #     #                  markers=False),
        #     'xlim': (0.0, 0.6),
        #     'ylim': (0.0, 3),
        #     'ylabel': r'teacher $\mapsto$ student',
        #     # 'ylabel': r'teacher $\mapsto$ student',
        #     # 'hlines': list(modelsDF.groupby('model_name').test_score.max())
        # },
        # # decoding
        # {
        #     'x': '10-shot co-smoothing',
        #     'y': 'MI_teacher->student',  # -angular
        #     'hue': 'n_components',  # 'unique_id',
        #     'data': modelsDF,  # [modelsDF['original co-smoothing'] > (modelsDF['original co-smoothing'].max() - 0.1)],
        #     'data_lines': modelsGT,
        #     # 'data_lines': None,
        #     'save_path': os.path.join('plots', main_dir, 'MI_teacherstudent_vs_10shotcosmoothing.png'),
        #     # 'func1': partial(sns.lineplot, legend=False, units='unique_id', estimator=None, alpha=0.5, errorbar=None,
        #     #                  markers=False),
        #     'xlim': (-0.3, 0.6),
        #     'ylim': (-3.0, 3),
        #     'ylabel': r'teacher $\mapsto$ student',
        #     # 'ylabel': r'teacher $\mapsto$ student',
        #     # 'hlines': list(modelsDF.groupby('model_name').test_score.max())
        # },
        # # decoding
        # {
        #     'x': 'decoder_student->teacher ratio',
        #     'y': 'decoder_student->teacher ratio',  # -angular
        #     'hue': 'n_components',  # 'unique_id',
        #     'data': modelsDF,  # [modelsDF['original co-smoothing'] > (modelsDF['original co-smoothing'].max() - 0.1)],
        #     'data_lines': modelsGT,
        #     # 'data_lines': None,
        #     'save_path': os.path.join('plots', main_dir, 'decoding_studentteacher_ratio_vs_originalcosmoothing.png'),
        #     # 'func1': partial(sns.lineplot, legend=False, units='unique_id', estimator=None, alpha=0.5, errorbar=None,
        #     #                  markers=False),
        #     # 'xlim': (0.0, 0.6),
        #     # 'ylim': (0.0, 0.1),
        #     'ylabel': r'student $\mapsto$ teacher',
        #     # 'ylabel': r'teacher $\mapsto$ student',
        #     # 'hlines': list(modelsDF.groupby('model_name').test_score.max())
        # },
        # # decoding
        # {
        #     'x': '22-shot co-smoothing',
        #     'y': 'decoder_teacher->student ratio',  # -angular
        #     'hue': 'n_components',  # 'unique_id',
        #     'data': modelsDF,  # [modelsDF['original co-smoothing'] > (modelsDF['original co-smoothing'].max() - 0.1)],
        #     'data_lines': modelsGT,
        #     # 'data_lines': None,
        #     'save_path': os.path.join('plots', main_dir, 'decoding_teacherstudent_ratio_vs_22shotcosmoothing.png'),
        #     # 'func1': partial(sns.lineplot, legend=False, units='unique_id', estimator=None, alpha=0.5, errorbar=None,
        #     #                  markers=False),
        #     'xlim': (0.0, 0.6),
        #     # 'ylim': (0.0, 0.1),
        #     'ylabel': r'student $\mapsto$ teacher',
        #     # 'ylabel': r'teacher $\mapsto$ student',
        #     # 'hlines': list(modelsDF.groupby('model_name').test_score.max())
        # },
        #
        # # decoding
        # {
        #     'x': 'original co-smoothing',
        #     'y': 'decoder_student->teacher ratio',  # -angular
        #     'hue': 'n_components',  # 'unique_id',
        #     'data': modelsDF,  # [modelsDF['original co-smoothing'] > (modelsDF['original co-smoothing'].max() - 0.1)],
        #     'data_lines': modelsGT,
        #     # 'data_lines': None,
        #     'save_path': os.path.join('plots', main_dir, 'decoding_studentteacher_ratio_vs_originalcosmoothing.png'),
        #     # 'func1': partial(sns.lineplot, legend=False, units='unique_id', estimator=None, alpha=0.5, errorbar=None,
        #     #                  markers=False),
        #     # 'xlim': (0.0, 0.6),
        #     # 'ylim': (0.0, 0.1),
        #     'ylabel': r'student $\mapsto$ teacher',
        #     # 'ylabel': r'teacher $\mapsto$ student',
        #     # 'hlines': list(modelsDF.groupby('model_name').test_score.max())
        # },
        # # decoding
        # {
        #     'x': 'original co-smoothing',
        #     'y': 'decoder_student->teacher',  # -angular
        #     'hue': 'n_components',  # 'unique_id',
        #     'data': modelsDF,  # [modelsDF['original co-smoothing'] > (modelsDF['original co-smoothing'].max() - 0.1)],
        #     'data_lines': modelsGT,
        #     # 'data_lines': None,
        #     'save_path': os.path.join('plots', main_dir, 'decoding_studentteacher_vs_originalcosmoothing.png'),
        #     'save_path': os.path.join('plots', main_dir, 'decoding_studentteacher_vs_originalcosmoothing.png'),
        #     # 'func1': partial(sns.lineplot, legend=False, units='unique_id', estimator=None, alpha=0.5, errorbar=None,
        #     #                  markers=False),
        #     # 'xlim': (0.0, 0.6),
        #     # 'ylim': (0.0, 0.1),
        #     'ylabel': r'student $\mapsto$ teacher',
        #     # 'ylabel': r'teacher $\mapsto$ student',
        #     # 'hlines': list(modelsDF.groupby('model_name').test_score.max())
        # },
        #
        # # decoding
        # {
        #     'x': 'original co-smoothing',
        #     'y': 'decoder_teacher->student',  # -angular
        #     'hue': 'n_components',  # 'unique_id',
        #     'data': modelsDF,  # [modelsDF['original co-smoothing'] > (modelsDF['original co-smoothing'].max() - 0.1)],
        #     'data_lines': modelsGT,
        #     # 'data_lines': None,
        #     'save_path': os.path.join('plots', main_dir, 'decoding_teacherstudent_vs_originalcosmoothing.png'),
        #     # 'func1': partial(sns.lineplot, legend=False, units='unique_id', estimator=None, alpha=0.5, errorbar=None,
        #     #                  markers=False),
        #     # 'xlim': (0.0, 0.6),
        #     # 'ylim': (0.0, 0.1),
        #     # 'xlabel': r'student $\mapsto$ teacher',
        #     'ylabel': r'teacher $\mapsto$ student',
        #     # 'hlines': list(modelsDF.groupby('model_name').test_score.max())
        # },
        #
        # # decoding
        # {
        #     'x': 'decoder_teacher->student shuffled',
        #     'y': 'decoder_teacher->student',  # -angular
        #     'hue': 'n_components',  # 'unique_id',
        #     'data': modelsDF,  # [modelsDF['original co-smoothing'] > (modelsDF['original co-smoothing'].max() - 0.1)],
        #     'data_lines': modelsGT,
        #     # 'data_lines': None,
        #     'save_path': os.path.join('plots', main_dir, 'decoding_shuffled_notshuffled.png'),
        #     # 'func1': partial(sns.lineplot, legend=False, units='unique_id', estimator=None, alpha=0.5, errorbar=None,
        #     #                  markers=False),
        #     # 'xlim': (0.0, 0.6),
        #     # 'ylim': (0.0, 0.1),
        #     'xlabel': r'teacher $\mapsto$ student shuffled',
        #     'ylabel': r'teacher $\mapsto$ student',
        #     # 'hlines': list(modelsDF.groupby('model_name').test_score.max())
        # },
        #
        # # decoding
        # {
        #     'x': '22-shot co-smoothing',
        #     'y': 'decoder_teacher->student',  # -angular
        #     'hue': 'n_components',  # 'unique_id',
        #     'data': modelsDF [modelsDF['original co-smoothing'] > (modelsDF['original co-smoothing'].max() - 0.1)],
        #     'data_lines': modelsGT,
        #     # 'data_lines': None,
        #     'save_path': os.path.join('plots', main_dir, 'decoding_vs_22shot.png'),
        #     # 'func1': partial(sns.lineplot, legend=False, units='unique_id', estimator=None, alpha=0.5, errorbar=None,
        #     #                  markers=False),
        #     'xlim': (0.0, 0.45),
        #     'ylim': (0.0, 0.1),
        #     # 'xlabel': r'student $\mapsto$ teacher',
        #     'ylabel': r'teacher $\mapsto$ student',
        #     # 'hlines': list(modelsDF.groupby('model_name').test_score.max())
        # },
        # # decoding
        # {
        #     'x': 'decoder_student->teacher ratio',
        #     'y': 'decoder_teacher->student ratio',  # -angular
        #     'hue': 'n_components',  # 'unique_id',
        #     'data': modelsDF,  # [modelsDF['original co-smoothing'] > (modelsDF['original co-smoothing'].max() - 0.1)],
        #     'data_lines': modelsGT,
        #     # 'data_lines': None,
        #     'save_path': os.path.join('plots', main_dir, 'decoding_ratio.png'),
        #     # 'func1': partial(sns.lineplot, legend=False, units='unique_id', estimator=None, alpha=0.5, errorbar=None,
        #     #                  markers=False),
        #     # 'xlim': (0.0, 0.1),
        #     # 'ylim': (0.0, 0.1),
        #     'xlabel': r'student $\mapsto$ teacher',
        #     'ylabel': r'teacher $\mapsto$ student',
        #     # 'hlines': list(modelsDF.groupby('model_name').test_score.max())
        # },
        #
        # # decoding
        # {
        #     'x': 'decoder_student->teacher',
        #     'y': 'decoder_teacher->student',  # -angular
        #     'hue': 'n_components',  # 'unique_id',
        #     'data': modelsDF,#[modelsDF['original co-smoothing'] > (modelsDF['original co-smoothing'].max() - 0.1)],
        #     'data_lines': modelsGT,
        #     # 'data_lines': None,
        #     'save_path': os.path.join('plots', main_dir, 'decoding.png'),
        #     # 'func1': partial(sns.lineplot, legend=False, units='unique_id', estimator=None, alpha=0.5, errorbar=None,
        #     #                  markers=False),
        #     # 'xlim': (0.0, 0.1),
        #     # 'ylim': (0.0, 0.1),
        #     'xlabel': r'student $\mapsto$ teacher',
        #     'ylabel': r'teacher $\mapsto$ student',
        #     # 'hlines': list(modelsDF.groupby('model_name').test_score.max())
        # },
        # k-shot lines
        # {
        #     'x': '10-shot co-smoothing',
        #     'y': 'decoder_teacher->student',  # -angular
        #     'hue': 'n_components',  # 'unique_id',
        #     'data': modelsDF[modelsDF['original co-smoothing'] > (modelsDF['original co-smoothing'].max() - 0.1)],
        #     'data_lines': modelsGT,
        #     # 'data_lines': None,
        #     'save_path': os.path.join('plots', main_dir, 'decoding_vs_10shot.png'),
        #     # 'func1': partial(sns.lineplot, legend=False, units='unique_id', estimator=None, alpha=0.5, errorbar=None,
        #     #                  markers=False),
        #     'xlim': (0.0, 0.45),
        #     # 'ylim': (0.0, 0.6)
        #     # 'hlines': list(modelsDF.groupby('model_name').test_score.max())
        # },
        # {
        #     'x': '3-shot co-smoothing',
        #     'y': 'decoder_teacher->student',  # -angular
        #     'hue': 'n_components',  # 'unique_id',
        #     'data': modelsDF[modelsDF['original co-smoothing'] > (modelsDF['original co-smoothing'].max() - 0.1)],
        #     'data_lines': modelsGT,
        #     # 'data_lines': None,
        #     'save_path': os.path.join('plots', main_dir, 'decoding_vs_3shot.png'),
        #     # 'func1': partial(sns.lineplot, legend=False, units='unique_id', estimator=None, alpha=0.5, errorbar=None,
        #     #                  markers=False),
        #     'xlim': (-2.0, 0.45),
        #     # 'ylim': (0.0, 0.6)
        #     # 'hlines': list(modelsDF.groupby('model_name').test_score.max())
        #     'ylabel':'decoder_teacher-student',
        # },
        # # k-shot lines
        # {
        #     'x': '22-shot co-smoothing',
        #     'y': 'similarity.procrustes', #-angular
        #     'hue': 'n_components',  # 'unique_id',
        #     'data': modelsDF[modelsDF['original co-smoothing']>(modelsDF['original co-smoothing'].max()-0.1)],
        #     # 'data_lines': DF[DF.model_name == 'Ground truth'],
        #     'data_lines': None,
        #     'save_path': os.path.join('plots', main_dir, 'similarity_vs_22shot.png'),
        #     'func2': partial(sns.lineplot, legend=False, units='unique_id', estimator=None, alpha=0.5, errorbar=None,
        #                      markers=False),
        #     'xlim': (0.0, 0.45),
        #     # 'ylim': (0.0, 0.6)
        #     # 'hlines': list(modelsDF.groupby('model_name').test_score.max())
        # },
        # k-shot lines
        # {
        #     'x': 'k',
        #     'y': 'co-smoothing',
        #     'hue': None,#'unique_id',
        #     'data': models_k_shot,
        #     # 'data_lines': DF[DF.model_name == 'Ground truth'],
        #     'data_lines':None,
        #     'save_path': os.path.join('plots', main_dir, 'k-shot.png'),
        #     'func1': partial(sns.lineplot,legend=True,units='unique_id',estimator=None,alpha=0.5,errorbar=None,markers=False,lw=0.5),
        #     # 'xlim': (0.0, 0.6),
        #     'ylim': (0.0, 0.6)
        #     # 'hlines': list(modelsDF.groupby('model_name').test_score.max())
        # },
        # # k shot
        # {
        #     'x': 'original co-smoothing',
        #     'y': '3-shot co-smoothing',
        #     'hue': 'n_components', #'similarity.procrustes',
        #     'data': modelsDF,
        #     'data_lines': DF[DF.model_name == 'Ground truth'],
        #     # 'data_lines':None,
        #     'save_path': os.path.join('plots', main_dir, 'v2_original_3shot.png'),
        #     # 'func2': partial(sns.scatterplot),
        #     'xlim': (0.0, 0.6),
        #     'ylim': (-3, 0.6)
        #     # 'hlines': list(modelsDF.groupby('model_name').test_score.max())
        # },
        # # k shot
        # {
        #     'x': 'original co-smoothing',
        #     'y': '10-shot co-smoothing',
        #     'hue': 'n_components',
        #     'data': modelsDF,
        #     'data_lines': modelsGT,
        #     # 'data_lines':None,
        #     'save_path': os.path.join('plots', main_dir, 'v2_original_10shot.png'),
        #     # 'func2': sns.scatterplot,
        #     'xlim': (0.0, 0.6),
        #     'ylim': (0.0, 0.6)
        #     # 'hlines': list(modelsDF.groupby('model_name').test_score.max())
        # },
        # # k shot
        # {
        #     'x': 'original co-smoothing',
        #     'y': '100-shot co-smoothing',
        #     'hue': None,
        #     'data': modelsDF,
        #     'data_lines': DF[DF.model_name == 'Ground truth'],
        #     # 'data_lines':None,
        #     'save_path': os.path.join('plots', main_dir, 'v2_original_100shot.png'),
        #     'func2': sns.scatterplot,
        #     'xlim': (0.0, 0.6),
        #     'ylim': (0.0, 0.6)
        #     # 'hlines': list(modelsDF.groupby('model_name').test_score.max())
        # },
        # # k shot
        # {
        #     'x': '100-shot co-smoothing',
        #     'y': '10-shot co-smoothing',
        #      'hue': None,
        #     'data': modelsDF,
        #     'data_lines': DF[DF.model_name == 'Ground truth'],
        #     # 'data_lines':None,
        #     'save_path': os.path.join('plots', main_dir, 'v2_100shot_3shot.png'),
        #     'func2': sns.scatterplot,
        #     'xlim'          : (0.0,0.6),
        #     'ylim'          : (0.0,0.6)
        #     # 'hlines': list(modelsDF.groupby('model_name').test_score.max())
        # },
        # {
        #     'x': 'original co-smoothing',
        #     'y': '10-shot co-smoothing',
        #     'hue': 'similarity.procrustes',
        #     'data': modelsDF,
        #     'data_lines': DF[DF.model_name == 'Ground truth'],
        #     # 'data_lines':None,
        #     'save_path': os.path.join('plots', main_dir, 'v2_original_vs_10shot.png'),
        #     'func2': sns.scatterplot,
        #     'xlim': (0.0, 0.6),
        #     'ylim': (0.0, 0.6)
        #     # 'hlines': list(modelsDF.groupby('model_name').test_score.max())
        # },
        # # test score
        # {
        #     'x'             :'n_components',
        #     'y'             :'test_score',
        #     'hue'           : 'model_name',
        #     'data'          : modelsDF,
        #     'data_lines'    : DF[DF.model_name=='Ground truth'],
        #     'save_path'     : os.path.join('plots',main_dir,'test_score.png'),
        #     'func2'         : sns.lineplot,
        #     #'xlim'          : (0, 51),
        #     #'ylim'          : [-2.5,-2.2], #-1.625, -1.550],
        #     'hlines'        : list(modelsDF.groupby('model_name').test_score.max())
        # },
        # # test2 score
        # {
        #     'x': 'n_components',
        #     'y': 'test_score2',
        #     'hue': 'model_name',
        #     'data': modelsDF,
        #     'data_lines': DF[DF.model_name == 'Ground truth'],
        #     'save_path': os.path.join('plots', main_dir, 'test_score2.png'),
        #     'func2': sns.lineplot,
        #     # 'xlim'          : (0, 51),
        #     # 'ylim'          : [-2.5,-2.2], #-1.625, -1.550],
        #     'hlines': list(modelsDF.groupby('model_name').test_score2.max())
        # },
        # # test self consistency
        # {
        #     'x': 'n_components',
        #     'y': 'self_consistency',
        #     'hue': 'model_name',
        #     'data': modelsDF,
        #     'data_lines': DF[DF.model_name == 'Ground truth'],
        #     'save_path': os.path.join('plots', main_dir, 'test_self_consistency.png'),
        #     'func2': sns.lineplot,
        #     # 'xlim'          : (0, 51),
        #     #'ylim': [-1.7, -1.5],
        # },
        # # test self consistency modified pi
        # {
        #     'x': 'n_components',
        #     'y': 'test_self_consistency_modified_pi',
        #     'hue': 'model_name',
        #     'data': modelsDF,
        #     'data_lines': DF[DF.model_name == 'Ground truth'],
        #     'save_path': os.path.join('plots', main_dir, 'test_self_consistency_modified_pi.png'),
        #     'func2': sns.lineplot,
        #     # 'xlim'          : (0, 51),
        #     # 'ylim': [-1.7, -1.5],
        # },
        # # test participation ratio
        # {
        #     'x': 'n_components',
        #     'y': 'PR',
        #     'hue': 'model_name',
        #     'data': modelsDF,  # [modelsDF.test_score > -1.62],
        #     'data_lines': DF[DF.model_name == 'Ground truth'],
        #     'save_path': os.path.join('plots', main_dir, 'test_PR.png'),
        #     'func2': sns.lineplot,
        #     # 'xlim'          : (0, 51),
        #     # 'ylim': [-1.7, -1.5],
        # },
        # # test participation ratio
        # {
        #     'x': 'test_score',
        #     'y': 'PR',
        #     'hue': 'model_name',
        #     'data': modelsDF,  # [modelsDF.test_score > -1.62],
        #     'data_lines': DF[DF.model_name == 'Ground truth'],
        #     'save_path': os.path.join('plots', main_dir, 'test_PR_vs_score.png'),
        #     # 'func2': sns.lineplot,
        #     # 'xlim' : [-1.625, -1.550],
        #     # 'ylim': [-1.7, -1.5],
        # },
        # # test self consistency vs score
        # {
        #     'x': 'test_score',
        #     'y': 'self_consistency',
        #     'hue': 'model_name',
        #     'data': modelsDF,
        #     'data_lines': DF[DF.model_name == 'Ground truth'],
        #     'save_path': os.path.join('plots', main_dir, 'test_self_consistency_vs_score.png'),
        #     # 'func2': sns.lineplot,
        #     # 'xlim'          : (0, 51),
        #     # 'ylim': [-1.7, -1.5],
        # },
        # # test self consistency vs score
        # {
        #     'x': 'test_score',
        #     'y': 'test_self_consistency_modified_pi',
        #     'hue': 'model_name',
        #     'data': modelsDF,
        #     'data_lines': DF[DF.model_name == 'Ground truth'],
        #     'save_path': os.path.join('plots', main_dir, 'self_consistency_modified_pi_vs_score.png'),
        #     # 'func2': sns.lineplot,
        #     # 'xlim'          : (0, 51),
        #     # 'ylim': [-1.7, -1.5],
        # },
        #
        # # test D_JS_stationary_pi
        # {
        #     'x': 'n_components',
        #     'y': 'D_JS_stationary_pi',
        #     'hue': 'model_name',
        #     'data': modelsDF,
        #     'data_lines': DF[DF.model_name == 'Ground truth'],
        #     'save_path': os.path.join('plots', main_dir, 'test_D_JS_stationary_pi.png'),
        #     'func2': sns.lineplot,
        #     # 'xlim'          : (0, 51),
        #     # 'ylim': [-1.7, -1.5],
        # },
        # # test Mutual info predicted probability
        # {
        #     'x': 'n_components',
        #     'y': 'MI_predict_proba',
        #     'hue': 'model_name',
        #     'data': modelsDF, #[modelsDF.test_score>-1.62],
        #     'data_lines': DF[DF.model_name == 'Ground truth'],
        #     'save_path': os.path.join('plots', main_dir, 'test_MI_predict_proba.png'),
        #     'func2': sns.lineplot,
        #     # 'xlim'          : (0, 51),
        #     # 'ylim': [-1.7, -1.5],
        # },
        # # test Mutual info predicted probability versus score
        # {
        #     'x': 'MI_predict_proba',
        #     'y': 'test_score',
        #     'hue': 'model_name',
        #     'data': modelsDF, #[modelsDF.test_score>-1.62],
        #     'data_lines': DF[DF.model_name == 'Ground truth'],
        #     'save_path': os.path.join('plots', main_dir, 'test_MI_predict_proba_versus_score.png'),
        #     'func2': None,
        #     # 'xlim'          : (0, 51),
        #     #'ylim': [-1.625, -1.550],
        # },
        # # test Mutual info predicted probability
        # {
        #     'x': 'n_components',
        #     'y': 'posterior_entropy',
        #     'hue': 'model_name',
        #     'data': modelsDF,#[modelsDF.test_score > -1.62],
        #     'data_lines': DF[DF.model_name == 'Ground truth'],
        #     'save_path': os.path.join('plots', main_dir, 'test_posterior_entropy.png'),
        #     'func2': sns.lineplot,
        #     # 'xlim'          : (0, 51),
        #     # 'ylim': [-1.7, -1.5],
        # },
        # # test Mutual info predicted probability versus score
        # {
        #     'x': 'train_score',
        #     'y': 'test_score',
        #     'hue': 'model_name',
        #     'data': modelsDF,
        #     'data_lines': DF[DF.model_name == 'Ground truth'],
        #     'save_path': os.path.join('plots', main_dir, 'test_vs_train_score.png'),
        #     'func2': None,
        #     # 'xlim'          : (0, 51),
        #     # 'ylim': [-1.625, -1.550],
        # },

    ]
    for arg in plot_args_list[:]:
        plot_scatter_with_lines(**arg)

    # print(
    #     modelsDF.groupby('model_name').test_score.max() .sort_values(ascending=False)
    #     #
    # )

    # modelsDF['unique_id'] = modelsDF['model_id'].astype(int).astype(str) + '_' + modelsDF['n_components'].astype(str)
    # modelsDF_ = modelsDF.set_index(['unique_id'])[cosmoothing_columns].T
    # modelsDF_['k'] = modelsDF_.index.str.split('-shot').str[0].astype(int)
    # modelsDF_=modelsDF_.set_index('k')
    # models_k_shot = modelsDF.melt(id_vars=['model_id', 'n_components'], value_vars=cosmoothing_columns, value_name='co-smoothing')
    # models_k_shot = models_k_shot[models_k_shot.variable.str.split(' ').str[0]!='original']
    # models_k_shot['k'] = models_k_shot.variable.str.split('-shot').str[0]
    # models_k_shot['k']  = models_k_shot['k'].apply(lambda x: int(x) if len(x)==1 else x)
    # models_k_shot['unique_id'] = models_k_shot['model_id'].astype(int).astype(str) + '_' + models_k_shot['n_components'].astype(str)
    #
    #
    # save_path = os.path.join('plots',main_dir,'k-shot2')
    # fig,ax=plt.subplots()
    # # best_modelsDF_ = modelsDF_[modelsDF_['original co-smoothing'] > (modelsDF_['original co-smoothing'].max() - 0.1)],
    # print(modelsDF_)
    # sns.lineplot(data=modelsDF_,ax=ax,legend=False,linewidth=0.5)
    # ax.set_ylim(0,0.6)
    # ax.set_xscale('log')
    # fig.tight_layout()
    # if not os.path.exists(os.path.dirname(save_path)):
    #     os.makedirs(os.path.dirname(save_path))
    # fig.savefig(save_path+'.pdf',dpi=300)
    # fig.savefig(save_path+'.png',dpi=300)

    # likelihood_columns = [c for c in modelsDF.columns if '-shot likelihood jax' in c]  # + ['original loglikelihood']
    # modelsDF_ = DF.set_index('model_name')[likelihood_columns].T
    # print(modelsDF_)
    # modelsDF_['k'] = modelsDF_.index.str.split('-').str[0].astype(int)
    # modelsDF_ = modelsDF_.melt(id_vars=['k'],value_vars=DF.model_name.unique(),value_name='Loglikelihood',var_name='model_name')
    # # print(modelsDF_)
    #
    # fig,ax=plt.subplots()
    # sns.scatterplot(x='k',y='Loglikelihood',data=modelsDF_,ax=ax,hue='model_name',legend=True)#,linewidth=0.5)
    # k_range = np.arange(2,100)
    # for i,name in enumerate(DF.model_name.unique()):
    #     f = DF[DF.model_name==name]['trace factor'].values
    #     f_diag = DF[DF.model_name == name]['diag trace factor'].values
    #     orig =  DF[DF.model_name==name]['original likelihood jax'].values
    #     # print(f.values,orig.values)
    #     ax.plot(k_range,orig-f/2/k_range,ls='dashed',label=name+r'  $L(\phi_{\infty})-\frac{Tr[I^{-1}H]}{2K}$',c='C%d'%i)
    #     ax.plot(k_range, orig - f_diag / 2 / k_range, ls='dotted',
    #             label=name + r'  $L(\phi_{\infty})- \frac{1}{2K}\sum_i\frac{H_{ii}}{I_{ii}}$',c='C%d'%i)
    # ax.legend()
    # save_path = os.path.join('plots', main_dir, 'kshot_loglikelihood.png')
    # make_path_if_not_exist(save_path)
    # fig.savefig(save_path,dpi=300)
    # plt.close(fig)
    #
    #
    # cosmoothing_columns = [c for c in modelsDF.columns if '-shot co-smoothing' in c] # + ['original co-smoothing']
    # modelsDF_ = DF.set_index('model_name')[cosmoothing_columns].T
    # print(modelsDF_)
    # modelsDF_['k'] = modelsDF_.index.str.split('-').str[0].astype(int)
    # modelsDF_ = modelsDF_.melt(id_vars=['k'],value_vars=DF.model_name.unique(),value_name='co-smoothing',var_name='model_name')
    # # print(modelsDF_)
    #
    # fig,ax=plt.subplots()
    # sns.lineplot(x='k',y='co-smoothing',data=modelsDF_,ax=ax,hue='model_name',legend=True,linewidth=0.5)
    # save_path = os.path.join('plots', main_dir, 'kshot_cosmoothing.png')
    # make_path_if_not_exist(save_path)
    # fig.savefig(save_path,dpi=300)
    # plt.close(fig)


def plot_hessian_FI(
        main_dir='all_models_validated_v2/teacher_state3_bernoulli_partial_eps0.01_length10',
        sub_dir='',
):
    path = os.path.join(main_dir, sub_dir)
    files = os.listdir(path)
    hessian_FI_files = [f for f in files if 'hessian_fisherinfo' in f]
    hessian_FI_files = list(reversed(hessian_FI_files))

    fig, axs = plt.subplots(2, len(hessian_FI_files))

    for i, file_name in enumerate(hessian_FI_files):
        arrays = np.load(os.path.join(path, file_name))
        hessian, FI = arrays.values()
        print(hessian.shape)
        ax = axs[0, i]
        ax.imshow(hessian[100:120, 100:120])
        ax.set_title(file_name.split('_hessian')[0])
        ax = axs[1, i]
        ax.imshow(FI[:15, :15])
    save_path = os.path.join('plots', main_dir, 'hesssian_FI_matrices.png')
    make_path_if_not_exist(save_path)
    fig.savefig(save_path, dpi=300)
    plt.close(fig)

    fig, axs = plt.subplots(2, 2, sharex=True, sharey=True)
    get_offdiag = lambda x: x[~np.eye(x.shape[0], dtype=bool)]
    for i, file_name in enumerate(hessian_FI_files):
        arrays = np.load(os.path.join(path, file_name))
        hessian, FI = arrays.values()
        bins = np.logspace(-5, 2.2, 50)

        ax = axs[0, i]
        ax.set_title(file_name.split('_hessian')[0])
        ax.hist(np.abs(np.diag(hessian)), density=True, bins=bins, label='diagonal', alpha=0.8)
        ax.hist(np.abs(get_offdiag(hessian)), density=True, bins=bins, label='off-diagonal', alpha=0.8)

        ax = axs[1, i]
        ax.hist(np.abs(np.diag(FI)), density=True, bins=bins, alpha=0.8)
        ax.hist(np.abs(get_offdiag(FI)), density=True, bins=bins, alpha=0.8)

    for ax in axs.flatten():
        ax.set_yscale('log')
        ax.set_xscale('log')
        # pass
    axs[0, 0].legend(fontsize=8, framealpha=0.5)
    axs[0, 0].set_ylabel('hessian')
    axs[1, 0].set_ylabel('fisher information')
    save_path = os.path.join('plots', main_dir, 'hesssian_FI_hist.png')
    make_path_if_not_exist(save_path)
    fig.tight_layout()
    fig.savefig(save_path, dpi=300)
    plt.close(fig)

    fig, axs = plt.subplots(1, 2, sharex=True, sharey=True)

    for i, file_name in enumerate(hessian_FI_files):
        arrays = np.load(os.path.join(path, file_name))
        hessian, FI = arrays.values()
        ax = axs[i]
        ax.scatter(hessian, np.linalg.inv(FI), label=file_name.split('_hessian')[0], alpha=0.7, s=10)
    #
    save_path = os.path.join('plots', main_dir, 'hesssian_FI_scatter.png')
    make_path_if_not_exist(save_path)
    fig.savefig(save_path, dpi=300)
    plt.close(fig)


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
        path = os.path.join('all_models_validated', main_dir)
        print(path)
        DF, main_path = collater(main_dir=path)

        GT_state = int(main_dir.split('_')[0].split('state')[-1])
        GT_obs = int(main_dir.split('_')[1].split('obs')[-1])
        DF['groundtruth_state_size'] = GT_state
        DF['groundtruth_observation_size'] = GT_obs

        DFs.append(DF)
    DF = pd.concat(DFs).reset_index()

    DFmodels = DF[DF.model_name != 'Groundtruth']
    # DFmodels = DFmodels[DFmodels.n_components >= 10]
    groundtruth = DF[DF.model_name == 'Groundtruth']
    best_models_idx = DFmodels.groupby(['groundtruth_state_size', 'groundtruth_observation_size', 'model_name'])[
        'test_score'].idxmax()
    # print(best_models_idx.values)
    # print('best models ',len(DFmodels.loc[best_models_idx.values]))
    best_models_and_groundtruth = pd.concat([DFmodels.loc[best_models_idx.values], groundtruth])
    # print('GT',len(groundtruth))
    best_models_and_groundtruth['groundtruth_specifications'] = 'state' + best_models_and_groundtruth[
        'groundtruth_state_size'].apply(str) + '_obs' + best_models_and_groundtruth[
                                                                    'groundtruth_observation_size'].apply(str)
    # print(len(best_models_and_groundtruth.sort_values(by=['groundtruth_specifications'])))

    # fig,ax  = plt.subplots()
    # sns.scatterplot(x='groundtruth_specifications',y='test_score',hue='model_name',data = best_models_and_groundtruth,ax=ax,s=5)
    # fig.savefig('plots/summary_best_test_scores.pdf')
    # #plt.show()
    #
    fig, ax = plt.subplots()
    sns.scatterplot(x='groundtruth_specifications', y='PR', hue='model_name', data=best_models_and_groundtruth, ax=ax,
                    s=10)
    fig.tight_layout()
    fig.savefig('plots/summary_best_PR.pdf', dpi=200)
    # plt.show()
    #
    # fig,ax  = plt.subplots()
    # sns.scatterplot(x='groundtruth_specifications',y='test_self_consistency',hue='model_name',data = best_models_and_groundtruth,ax=ax)
    # fig.tight_layout()
    # fig.savefig('plots/summary_best_SC.png',dpi=160)

    DF_for_table = best_models_and_groundtruth[['groundtruth_specifications', 'test_score', 'model_name']].pivot(
        columns='groundtruth_specifications', index='model_name', values='test_score')

    df = DF_for_table.T.drop(columns=['Groundtruth'])
    df.columns = [c.replace('_', ' ') for c in df.columns]
    df.index = [c.replace('_', ' ') for c in df.index]

    df_s = df.style.format("${:.4f}$")

    # loop through rows and find which column for each row has the highest value
    for row in df.index:
        col = df.loc[row].idxmax()
        # redo formatting for a specific cell
        df_s = df_s.format(lambda x: "$\mathbf{" + f'{x:.4f}' + "}$", subset=(row, col))

    print(df_s.to_latex())


def plot_krange():
    dir_name = '/Users/kabir/Documents/code/hmm_analysis/all_models_validated_test/teacher_state4_bernoulli_partial_eps0.01_length10/models_traintrials2000'
    GT = pd.read_csv(os.path.join(dir_name, '../groundtruth.csv'))
    best = pd.read_csv(os.path.join(dir_name, 'ncomp10_vanilla_id10_dynamax_fit_sgd.csv'))
    worst = pd.read_csv(os.path.join(dir_name, 'ncomp15_vanilla_id11_dynamax_fit_sgd.csv'))
    label_fontsize = 15
    all_models = []
    ncomps = [6, 8, 10, 15]
    ids = np.arange(40)
    for ncomp, id in product(ncomps, ids):
        fname = os.path.join(dir_name, f'ncomp{ncomp}_vanilla_id{id}_dynamax_fit_sgd.csv')
        if os.path.exists(fname):
            all_models.append(
                pd.read_csv(fname)
            )
    all_models = pd.concat(all_models)
    # print(all_models.shape)
    # print(all_models)

    krange = np.arange(4, 32, 2)
    krange = np.concatenate([krange, np.array([100, 500, 1000])])
    # krange = np.concatenate([krange,np.array([100])])
    score_names = [f'{k}-shot co-smoothing' for k in krange]
    score_std_names = [f'{k}-shot co-smoothing std' for k in krange]
    score_SEM_names = [f'{k}-shot co-smoothing SEM' for k in krange]

    std_across_models = all_models[score_names].std(axis=0)
    mean_of_SEM_models = all_models[score_SEM_names].mean(axis=0)
    all_models['linear_decoder_sum_both_directions'] = all_models[
        ['linear_decoder_teacher->student', 'linear_decoder_student->teacher']].sum(axis=1)
    epsilon = 2e-2
    best_cosmoothing_models = all_models[
        all_models['original co-smoothing'] > (all_models['original co-smoothing'] - 2e-2)]
    pearsonr_with_k = [
        # pearsonr(all_models[name].values,all_models['linear_decoder_sum_both_directions'].values).statistic
        pearsonr(best_cosmoothing_models[name].values, all_models['linear_decoder_teacher->student'].values).statistic
        for name in score_names
    ]

    print(std_across_models)
    print(mean_of_SEM_models)

    # fig,ax = plt.subplots(figsize=(4,3))
    # kwargs = {'lw':1,'marker':'o','markersize':3,'alpha':0.6}
    # ax.plot(krange, GT[score_names].values[0],c='black',ls='dashed',label='ground truth',**kwargs)
    # ax.plot(krange, best[score_names].values[0],c='green',label="`best'",**kwargs)
    # ax.plot(krange, worst[score_names].values[0],c='red',label="`worst'",**kwargs)
    # ax.legend()
    # ax.set_xscale('log')

    # ax.set_xlabel(r'$k$')
    # ax.set_ylabel(r'$k$-shot co-smoothing')

    # fig.tight_layout()
    # fig.savefig(os.path.join(dir_name,'krange_logscale.pdf'))
    # fig.savefig(os.path.join(dir_name,'krange_logscale.png'), dpi=500)
    # plt.close()

    fig, axs = plt.subplots(2, 2, figsize=(3 * 2, 2.2 * 2), sharex=True)
    axs = axs.flatten()
    ax = axs[0]
    kwargs = {'lw': 1, 'marker': 'o', 'markersize': 3, 'alpha': 0.6}
    ax.plot(krange, GT[score_names].values[0], c='black', ls='dashed', label=r'$\text{T}$', **kwargs)
    ax.plot(krange, best[score_names].values[0], c='green', label=r'$\text{S}_\text{good}$', **kwargs)
    ax.plot(krange, worst[score_names].values[0], c='red', label=r'$\text{S}_\text{bad}$', **kwargs)
    ax.legend()
    ax.set_xscale('log')
    ax.set_xlabel(r'$k$', fontsize=label_fontsize)
    # ax.set_ylabel(r'$k$-shot co-smoothing')
    ax.set_ylabel(r'$\langle \mathcal Q^k \rangle$', fontsize=label_fontsize)
    ax.set_ylim(0)

    ax = axs[1]
    kwargs = {'lw': 1, 'marker': 'o', 'markersize': 3, 'alpha': 0.6}
    ax.plot(krange, GT[score_std_names].values[0], c='black', ls='dashed', label='T', **kwargs)
    ax.plot(krange, best[score_std_names].values[0], c='green', label=r'$\text{S}_\text{good}$', **kwargs)
    ax.plot(krange, worst[score_std_names].values[0], c='red', label=r'$\text{S}_\text{bad}$', **kwargs)
    ax.legend()
    ax.set_xscale('log')
    ax.set_xlabel(r'$k$', fontsize=label_fontsize)
    ax.set_ylabel(r'$\sigma(\mathcal Q^k)$', fontsize=label_fontsize)
    ax.set_ylim(0)
    # ax = axs[2]
    # kwargs = {'lw':1,'marker':'o','markersize':3,'alpha':0.6}
    # # ax.plot(krange, GT[score_SEM_names].values[0],c='black',ls='dashed',label='ground truth',**kwargs)
    # # ax.plot(krange, best[score_SEM_names].values[0],c='green',label='best',**kwargs)
    # # ax.plot(krange, worst[score_SEM_names].values[0],c='red',label="`worst'",**kwargs)

    # ax.plot(krange,mean_of_SEM_models,label='mean of SEM')
    # ax.plot(krange,std_across_models,label='std across models')

    # ax.legend()
    # ax.set_xscale('log')
    # ax.set_xlabel(r'$k$')
    # ax.set_ylabel(r'Standard error of the mean (SEM)')
    # ax.set_yscale('log')

    ax = axs[2]
    kwargs = {'lw': 1, 'marker': 'o', 'markersize': 3, 'alpha': 0.6}
    # ax.plot(krange, GT[score_SEM_names].values[0],c='black',ls='dashed',label='ground truth',**kwargs)
    # ax.plot(krange, best[score_SEM_names].values[0],c='green',label='best',**kwargs)
    # ax.plot(krange, worst[score_SEM_names].values[0],c='red',label="`worst'",**kwargs)

    ax.plot(krange, std_across_models.values / mean_of_SEM_models.values, **kwargs)

    # ax.legend()
    ax.set_ylim(0)
    ax.set_xscale('log')
    ax.set_xlabel(r'$k$', fontsize=label_fontsize)
    # ax.set_ylabel(r'Signal to Noise ratio')
    ax.set_ylabel(r'$\frac{\sigma_{\text{across}}(\mathcal Q^k)}{\sigma_{\text{within}}(\mathcal Q^k)}$',
                  fontsize=label_fontsize)

    ax = axs[3]
    kwargs = {'lw': 1, 'marker': 'o', 'markersize': 3, 'alpha': 0.6}
    # ax.plot(krange, GT[score_SEM_names].values[0],c='black',ls='dashed',label='ground truth',**kwargs)
    # ax.plot(krange, best[score_SEM_names].values[0],c='green',label='best',**kwargs)
    # ax.plot(krange, worst[score_SEM_names].values[0],c='red',label="`worst'",**kwargs)

    ax.plot(krange, pearsonr_with_k, **kwargs)

    # ax.legend()
    ax.set_xscale('log')
    ax.set_xlabel(r'$k$', fontsize=label_fontsize)
    # ax.set_ylabel(r'$\rho$($k$-shot scores,sumGT)')
    ax.set_ylabel(r'$\rho(\langle \mathcal Q^k_\text{S} \rangle,\mathcal D_{\text{T} \rightarrow \text{S}})$',
                  fontsize=label_fontsize)

    # ax.set_yscale('log')

    # print(all_models.columns)
    # print(all_models[['linear_decoder_teacher->student','linear_decoder_student->teacher']])

    fig.tight_layout()
    fig.savefig(os.path.join(dir_name, 'krange_logscale.pdf'))
    fig.savefig(os.path.join(dir_name, 'krange_logscale.png'), dpi=500)
    plt.close()


if __name__ == '__main__':
    # collater()
    # main_summary()
    main()
    # plot_krange()
