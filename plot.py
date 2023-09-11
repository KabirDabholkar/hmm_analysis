import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.utils import check_random_state
from hmmlearn.hmm import GaussianHMM, CategoricalHMM
from hmmlearn.vhmm import VariationalCategoricalHMM
import pickle as pkl
import os
import seaborn as sns
import matplotlib as mpl
mpl.rcParams['text.usetex'] = True
#mpl.rcParams['text.latex.preamble'] = [r'\usepackage{amsmath}']

def collater():
    #options = [('None',1),(10,1),(10,10)]
    #options = [(10, 0), (10, 10)] #('None', 1),
    # dir_names = [f'models_traintrials70_window{window}_shiftlimits{shift}' for (window,shift) in options]
    options = ['augmented_mode','augmented_with_shift_mode','vanilla_sliced_mode']#'vanilla_mode'
    main_dir = 'all_models/state5_obs5_GT'
    main_path = os.path.join(main_dir,'models_traintrials700_')
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
    all_DFs += [pd.read_csv(os.path.join(main_dir,'groundtruth.csv'),index_col=None)]
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
    ax.legend(handles,labels)
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

    DF['score'] = DF['test_score']
    DF['self_consistency'] = DF['test_self_consistency']

    #sns.scatterplot(data = DF)
    modelsDF = DF[DF.model_name!='Ground truth']
    #modelsDF = modelsDF[modelsDF.score>-2.31]

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
            'data': modelsDF[modelsDF.test_score>-1.62],
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
            'data': modelsDF[modelsDF.test_score>-1.62],
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
            #'func2': sns.lineplot,
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
            #'func2': sns.lineplot,
            # 'xlim'          : (0, 51),
            # 'ylim': [-1.7, -1.5],
        },

    ]
    for arg in plot_args_list[:]:
        plot_scatter_with_lines(**arg)

    #
    # fig,ax = plt.subplots()
    # sns.lineplot(x='n_components',y='score',hue='model_name',data = modelsDF,ax=ax, estimator="max")
    # sns.scatterplot(x='n_components', y='score', hue='model_name', data=modelsDF, ax=ax,legend=None)
    # ax.set_xlabel(r'Model $|\mathcal X|$')
    # l = ax.axhline(DF[DF.model_name == 'Ground truth'].score.values[0], ls='dashed', color='black')
    # handles, labels = ax.get_legend_handles_labels()
    # handles += [l]
    # labels  += ['Ground-truth']
    # ax.set_ylabel('Test log likelihood')
    # ax.legend(handles,labels)
    # ax.set_xlim(0, 51)
    # ax.set_ylim(-0.68,-0.665)
    # fig.tight_layout()
    # print(modelsDF.columns)
    # fig.savefig('plots/test_score_comparison.png',dpi=200)
    #
    # fig, ax = plt.subplots()
    # sns.lineplot(x='n_components', y='train_score', hue='model_name', data=modelsDF, ax=ax)
    # sns.scatterplot(x='n_components', y='train_score', hue='model_name', data=modelsDF, ax=ax, legend=None)
    # ax.set_xlabel(r'Model $|\mathcal X|$')
    # l = ax.axhline(DF[DF.model_name == 'Ground truth'].train_score.values[0], ls='dashed', color='black')
    # handles, labels = ax.get_legend_handles_labels()
    # handles += [l]
    # labels  += ['Ground-truth']
    # ax.set_ylabel('Train log likelihood')
    # ax.legend(handles,labels)
    # ax.set_ylim(-.7,-0.635)
    # fig.tight_layout()
    # print(modelsDF.columns)
    # fig.savefig('plots/train_score_comparison.png',dpi=200)
    #
    # fig,ax = plt.subplots()
    # sns.lineplot(x='n_components',y='self_consistency',hue='model_name',data = modelsDF,ax=ax)
    # sns.scatterplot(x='n_components', y='self_consistency', hue='model_name', data=modelsDF, ax=ax, legend=False, alpha=0.6)
    # l = ax.axhline(DF[DF.model_name   == 'Ground truth'].self_consistency.values[0], ls='dashed', color='black')
    # ax.set_xlabel(r'Model $|\mathcal X|$')
    # handles, labels = ax.get_legend_handles_labels()
    # handles += [l]
    # labels  += ['Ground-truth']
    # ax.legend(handles,labels)
    # fig.tight_layout()
    # print(modelsDF.columns)
    # fig.savefig('plots/self_consistency_comparison.png',dpi=200)
    #
    # fig,ax = plt.subplots()
    # sns.lineplot(x='n_components',y='train_self_consistency',hue='model_name',data = modelsDF,ax=ax)
    # sns.scatterplot(x='n_components', y='train_self_consistency', hue='model_name', data=modelsDF, ax=ax, legend=False, alpha=0.6)
    # l = ax.axhline(DF[DF.model_name   == 'Ground truth'].self_consistency.values[0], ls='dashed', color='black')
    # ax.set_xlabel(r'Model $|\mathcal X|$')
    # handles, labels = ax.get_legend_handles_labels()
    # handles += [l]
    # labels  += ['Ground-truth']
    # ax.legend(handles,labels)
    # fig.tight_layout()
    # print(modelsDF.columns)
    # fig.savefig(os.path.join('plots',main_dir,'train_self_consistency_comparison.png'),dpi=200)
    #
    # fig,ax = plt.subplots()
    # sns.lineplot(x='n_components',y='iterations',hue='model_name',data = modelsDF,ax=ax)
    # sns.scatterplot(x='n_components', y='iterations', hue='model_name', data=modelsDF, ax=ax, legend=False, alpha=0.6)
    # l = ax.axhline(DF[DF.model_name   == 'Ground truth'].self_consistency.values[0], ls='dashed', color='black')
    # ax.set_xlabel(r'Model $|\mathcal X|$')
    # handles, labels = ax.get_legend_handles_labels()
    # handles += [l]
    # labels  += ['Ground-truth']
    # ax.legend(handles,labels)
    # ax.set_ylabel('iterations to convergence')
    # ax.set_ylim(0, 20)
    # fig.tight_layout()
    # print(modelsDF.columns)
    # fig.savefig(os.path.join('plots',main_dir,'iterations_comparison.png'),dpi=200)
    #
    #
    # fig,ax = plt.subplots()
    # sns.scatterplot(x='score', y='self_consistency', hue='model_name', data=modelsDF, ax=ax, legend=True)
    # l = ax.axhline(DF[DF.model_name   == 'Ground truth'].self_consistency.values[0], ls='dashed', color='black')
    # l = ax.axvline(DF[DF.model_name == 'Ground truth'].score.values[0], ls='dashed', color='black')
    # ax.set_xlabel(r'Test log likelihood')
    # handles, labels = ax.get_legend_handles_labels()
    #
    # handles += [l]
    # labels  += ['Ground-truth']
    # ax.legend(handles,labels)
    # ax.set_ylim(0,1)
    # ax.set_xlim(-0.75,-0.65)
    # fig.tight_layout()
    # print(modelsDF.columns)
    # fig.savefig(os.path.join('plots',main_dir,'SC_score_comparison.png'),dpi=200)
    #
    # ##### Steady state entropoy
    # fig,ax = plt.subplots()
    # sns.lineplot(x='n_components',y='steady_state_entropy',hue='model_name',data = modelsDF,ax=ax)
    # sns.scatterplot(x='n_components', y='steady_state_entropy', hue='model_name', data=modelsDF, ax=ax, legend=False, alpha=0.6)
    # l = ax.axhline(DF[DF.model_name   == 'Ground truth'].steady_state_entropy.values[0], ls='dashed', color='black')
    # ax.set_xlabel(r'Model $|\mathcal X|$')
    # handles, labels = ax.get_legend_handles_labels()
    # handles += [l]
    # labels  += ['Ground-truth']
    # ax.legend(handles,labels)
    # fig.tight_layout()
    # print(modelsDF.columns)
    # fig.savefig(os.path.join('plots',main_dir,'SSentropy_comparison.png'),dpi=200)
    #
    # ##### Mutual info emission
    # fig,ax = plt.subplots()
    # sns.lineplot(x='n_components',y='MI_emission',hue='model_name',data = modelsDF,ax=ax)
    # sns.scatterplot(x='n_components', y='MI_emission', hue='model_name', data=modelsDF, ax=ax, legend=False, alpha=0.6)
    # l = ax.axhline(DF[DF.model_name   == 'Ground truth'].MI_emission.values[0], ls='dashed', color='black')
    # ax.set_xlabel(r'Model $|\mathcal X|$')
    # ax.set_ylabel('Mutual info emission')
    # handles, labels = ax.get_legend_handles_labels()
    # handles += [l]
    # labels  += ['Ground-truth']
    # ax.legend(handles,labels)
    # fig.tight_layout()
    # print(modelsDF.columns)
    # fig.savefig(os.path.join('plots',main_dir,'MI_emission_comparison.png'),dpi=200)
    #
    # ##### Mutual info transition
    # fig,ax = plt.subplots()
    # sns.lineplot(x='n_components',y='MI_transition',hue='model_name',data = modelsDF,ax=ax)
    # sns.scatterplot(x='n_components', y='MI_transition', hue='model_name', data=modelsDF, ax=ax, legend=False, alpha=0.6)
    # l = ax.axhline(DF[DF.model_name   == 'Ground truth'].MI_transition.values[0], ls='dashed', color='black')
    # ax.set_xlabel(r'Model $|\mathcal X|$')
    # ax.set_ylabel('Mutual info transition')
    # handles, labels = ax.get_legend_handles_labels()
    # handles += [l]
    # labels  += ['Ground-truth']
    # ax.legend(handles,labels)
    # fig.tight_layout()
    # print(modelsDF.columns)
    # fig.savefig(os.path.join('plots',main_dir,'MI_transition_comparison.png'),dpi=200)
    #
    # ##### Steady state, pi: Jensen shannon div
    # fig,ax = plt.subplots()
    # sns.lineplot(x='n_components',y='D_JS_stationary_pi',hue='model_name',data = modelsDF,ax=ax)
    # sns.scatterplot(x='n_components', y='D_JS_stationary_pi', hue='model_name', data=modelsDF, ax=ax, legend=False, alpha=0.6)
    # l = ax.axhline(DF[DF.model_name   == 'Ground truth'].D_JS_stationary_pi.values[0], ls='dashed', color='black')
    # ax.set_xlabel(r'Model $|\mathcal X|$')
    # ax.set_ylabel(r'$D_{JS}(\pi,p_{\infty}(A))$')
    # handles, labels = ax.get_legend_handles_labels()
    # handles += [l]
    # labels  += ['Ground-truth']
    # ax.legend(handles,labels)
    # fig.tight_layout()
    # print(modelsDF.columns)
    # fig.savefig(os.path.join('plots',main_dir,'D_JS_stationary_pi_comparison.png'),dpi=200)
    #
    # ##### Steady state, pi: Jensen shannon div
    # fig,ax = plt.subplots()
    # #DF['projection_pi'] = DF['projection_pi'][0]
    # #DF['projection_pinf'] = DF['projection_pinf'][0]
    # #print(DF['projection_pi'].values)
    # #sns.lineplot(x='projection_pi',y='projection_pinf',hue='model_name',data = DF,ax=ax)
    # sns.scatterplot(x='projection_pi', y='projection_pinf', hue='model_name', data = modelsDF, ax=ax, legend=True, alpha=0.6)
    # d = DF[DF.model_name=='Ground truth']
    # ax.scatter(d.projection_pi, d.projection_pinf, color='black')
    # ax.set_xlim(0,1)
    # ax.set_ylim(0,1)
    # #sns.scatterplot(x='projection_pi', y='projection_pinf', hue='model_name', data=DF[DF.model_name   == 'Ground truth'], ax=ax, legend=False,alpha=0.6)
    # #l = ax.axhline(DF[DF.model_name   == 'Ground truth'].D_JS_stationary_pi.values[0], ls='dashed', color='black')
    # #ax.set_xlabel(r'Model $|\mathcal X|$')
    # #ax.set_ylabel(r'$D_{JS}(\pi,p_{\infty}(A))$')
    # #handles, labels = ax.get_legend_handles_labels()
    # #handles += [l]
    # #labels  += ['Ground-truth']
    # #ax.legend()#handles,labels)
    # fig.tight_layout()
    # fig.savefig(os.path.join('plots',main_dir,'projection_comparison.png'),dpi=200)
    #
    # ##### Participation ratio hidden state prob #########
    # fig,ax = plt.subplots()
    # sns.lineplot(x='n_components',y='PR',hue='model_name',data = modelsDF,ax=ax)
    # sns.scatterplot(x='n_components', y='PR', hue='model_name', data=modelsDF, ax=ax, legend=False, alpha=0.6)
    # l = ax.axhline(DF[DF.model_name   == 'Ground truth'].PR.values[0], ls='dashed', color='black')
    # ax.set_xlabel(r'Model $|\mathcal X|$')
    # ax.set_ylabel(r'Participation ratio')
    # handles, labels = ax.get_legend_handles_labels()
    # handles += [l]
    # labels  += ['Ground-truth']
    # ax.legend(handles,labels)
    # ax.set_ylim(0,10)
    # fig.tight_layout()
    # print(modelsDF.columns)
    # fig.savefig(os.path.join('plots',main_dir,'PR_comparison.png'),dpi=200)
    #
    # ##### Participation ratio hidden state prob #########
    # fig,ax = plt.subplots()
    # #sns.lineplot(x='score',y='PR',hue='model_name',data = modelsDF,ax=ax)
    # sns.scatterplot(x='score', y='PR', hue='model_name', data=modelsDF, ax=ax, legend=False, alpha=0.6)
    # l = ax.axhline(DF[DF.model_name   == 'Ground truth'].PR.values[0], ls='dashed', color='black')
    # l = ax.axvline(DF[DF.model_name == 'Ground truth'].score.values[0], ls='dashed', color='black')
    # ax.set_xlim(-0.69, -0.67)
    # ax.set_xlabel(r'Test score')
    # ax.set_ylabel(r'Participation ratio')
    # handles, labels = ax.get_legend_handles_labels()
    # #handles += [l]
    # #labels  += ['Ground-truth']
    # #ax.legend(handles,labels)
    # fig.tight_layout()
    # print(modelsDF.columns)
    # fig.savefig(os.path.join('plots',main_dir,'score_PR_comparison.png'),dpi=200)


    # g = sns.PairGrid(modelsDF,hue='model_name', diag_sharey=False, corner=True)
    # g.map_lower(sns.scatterplot)
    # g.map_diag(sns.kdeplot)
    # fig = g.figure
    # fig.savefig('plots/plots_pairplot.pdf')
    print(
        list(modelsDF.groupby('model_name').test_score.max())
    )

if __name__ == '__main__':
    #collater()
    main()