import numpy as np
import matplotlib.pyplot as plt
from functools import partial
# from scipy.stats.distributions import norm,
from scipy.stats import poisson,norm
# import config_utils
from omegaconf import OmegaConf
from utils import *
import hydra
# from hydra.utils import instantiate
from config_utils import instantiate
from config_utils.dict_module import DictModule,DictSequential

CONFIG_PATH = "cramer_rao_configs"

CONFIG_NAME = "poisson"

@hydra.main(version_base='1.3', config_path=CONFIG_PATH, config_name=CONFIG_NAME)
def main(cfg):
    estimator_variance = []
    mean_log_likelihood = []
    mean_log_likelihood2 = []
    sample_sizes = np.logspace(0.0,1.5,5).round().astype(int)

    repeats = 5000
    # test_samples = 1000

    # sampler = instantiate( cfg.sampler )



    # norm = instantiate( cfg.norm )
    dist = instantiate(cfg.dist) #norm
    print(dist.func)

    sampler = dist(**cfg.params).rvs

    # print(dist(loc=1.0,scale=1.0).pdf)
    # setattrs_kwargs(norm,**cfg.params)
    # def mean(x):
    #     return sum(x)/len(x)



    # estimator =
    estimator = DictModule(
        module= instantiate( cfg.estimator ),#partial(np.mean,axis=0,keepdims=True),
        in_keys=[['data','a']],
        out_keys=[cfg.estimated_param_name]
    )

    full_estimator = DictModule(
        module= dist.func.fit,
        in_keys=[['data','data']],
        out_keys=list(cfg.params.keys())
    )

    estimator = DictModule(
        module = DictModule(
            module=dist.func.fit,
            in_keys=[['data', 'data']],
            out_keys=list(cfg.params.keys())
        ),
        in_keys=[['data', 'data']],
        out_keys=cfg.estimated_param_name,
    )

    # print(instantiate(cfg.estimator_dictmodule))

    test_X = sampler(size=(repeats))[None, :]

    optimal_loglike = (
                getattr(
                    dist(**cfg.params),
                    cfg.func_type
                )(test_X)
            ).mean()


    for samples in sample_sizes:
        # samples = 5
        test_X = sampler(size=(repeats))[None, :]

        X = sampler(size=(samples,repeats))

        # sample_mean = X.mean(axis=0,keepdims=True)
        # setattrs_kwargs(norm, loc = estimator(X), scale = cfg.params.scale)

        repeat_log_likelihood = []
        for rep in range(repeats):
            estimated_params = estimator(X[:,rep])

            params = deepcopy( cfg.params )

            estimated_params = {**params,**estimated_params}
            repeat_log_likelihood += [
                # - np.log(np.sqrt(2*np.pi)*sigma) - np.mean((test_X-sample_mean)**2/(2*sigma**2))
                # dist.logpdf(test_X, loc = estimator(X), scale = cfg.params.scale).mean()
                optimal_loglike - np.nanmean(
                    getattr(
                        dist(**estimated_params),
                        cfg.func_type
                    )(test_X)
            )]
        mean_log_likelihood += [np.nanmean(np.array(repeat_log_likelihood))]



    # plt.plot(sample_sizes,mean_log_likelihood,color='C0')
    fig,axs=plt.subplots(1,1)
    # ax = axs[0]
    # ax.plot(sample_sizes,scale**2/sample_sizes,color='C0',ls='dashed')
    # ax.plot(sample_sizes,estimator_variance,color='C1')


    ax = axs #[1]
    print(mean_log_likelihood)
    # ax.plot(sample_sizes, - np.log(np.sqrt(2*np.pi)*scale)-0.5*(1+1/sample_sizes),color='C0',ls='dashed')
    ax.plot(sample_sizes, 1/sample_sizes/2, color='C0', ls='dashed')
    ax.plot(sample_sizes,mean_log_likelihood,color='C1')
    # ax.plot(sample_sizes,mean_log_likelihood2,color='C2',ls='dashed')
    fig.savefig(f'plots/test_plots/{cfg.name}_estimator_stats.png')
    plt.close()

    # plt.figure()
    # plt.plot(sample_sizes,sigma**2/sample_sizes,color='C0',ls='dashed')
    # plt.plot(sample_sizes,estimator_variance,color='C1')
    # plt.savefig('plots/test_plots/gaussian_estimator_variance.png')
    # plt.close()

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

    main()
