from hmm_adapter import PoissonHMM3d as hmmlearn_PoissonHMM
from hmm_adapter import BernoulliHMM3d as hmmlearn_BernoulliHMM
from dynamax.hidden_markov_model.models.poisson_hmm import PoissonHMM as jax_PoissonHMM
from dynamax.hidden_markov_model.models.bernoulli_hmm import BernoulliHMM as jax_BernoulliHMM
import jax.numpy as jnp
import jax.random as jr
import numpy as np
from jax import vmap
import optax
from utils import setattrs_kwargs

num_states = 4
emission_dim = 10

def hmmlearn_to_dynamaxhmm(hmmlearn_model):
    num_states   = hmmlearn_model.n_components
    emission_dim = hmmlearn_model.n_features
    hmm_class = {
        'BernoulliHMM' : jax_BernoulliHMM,
        'PoissonHMM'   : jax_PoissonHMM,
    }[hmmlearn_model.__class__.__base__.__name__]
    emission_param_name = {
        'BernoulliHMM' : 'emission_probs',
        'PoissonHMM'   : 'emission_rates'
    }[hmmlearn_model.__class__.__base__.__name__]
    jax_model = hmm_class(num_states, emission_dim)
    init_params = {
        'initial_probs'     : jnp.asarray(hmmlearn_model.startprob_),
        'transition_matrix' : jnp.asarray(hmmlearn_model.transmat_),
    }
    init_params[emission_param_name] = jnp.asarray(hmmlearn_model.lambdas_)
    params, param_props = jax_model.initialize(
        **init_params
    )
    return jax_model, params, param_props

def dynamaxhmm_to_hmmlearn(dynamax_model,params):
    n_components = dynamax_model.num_states
    n_features = dynamax_model.emission_dim

    hmm_class = {
        'BernoulliHMM' : hmmlearn_BernoulliHMM,
        'PoissonHMM'   : hmmlearn_PoissonHMM,
    }[dynamax_model.__class__.__name__]
    emission_param_name = {
        'BernoulliHMM':'emission_probs',
        'PoissonHMM': 'emission_rates'
    }[dynamax_model.__class__.__name__]
    hmmlearn_model = hmm_class(n_components=n_components)
    # print(params.initial.probs,params.transitions.transition_matrix,n_components,params.emissions.probs)

    setattrs_kwargs(
        hmmlearn_model,
        **{
            'startprob_' : np.asarray(params.initial.probs),
            'transmat_'  : np.asarray(params.transitions.transition_matrix),
            'lambdas_'   : np.asarray(params.emissions.probs),
            'n_features' : n_features
        }
    )
    # init_params[emission_param_name] = jnp.asarray(hmmlearn_model[emission_param_name])
    # params, param_props = jax_model.initialize(
    #     **init_params
    # )
    return hmmlearn_model


CONFIG_PATH = "configs"
CONFIG_NAME = "config_cohmm"

import hydra
from omegaconf import OmegaConf
from utils import setattrs
from config_utils import instantiate

@hydra.main(version_base='1.3', config_path=CONFIG_PATH, config_name=CONFIG_NAME)
def main(cfg):

    model = instantiate(cfg.teacher)

    transformed_model, params, param_props = hmmlearn_to_dynamaxhmm(model)
    all_data = instantiate(cfg.generate_all_data_dictmodule, _convert_='partial')(hmm_model=model)
    data = all_data[:20]
    score = model.score(data,mode3d=True)

    # proba1 = model.predict_proba(data[3])
    # hmm.smoother(params, emissions)
    proba2 = vmap(transformed_model.smoother,(None,0),0)(params,jnp.asarray(data))
    # print(proba1.shape,proba2.smoothed_probs.shape)
    score2 = proba2.marginal_loglik.sum(0)
    print(score,score2)

    # print(scores)
    # import matplotlib.pyplot as plt
    # fig,axs = plt.subplots(1,2)
    # # plt.scatter(proba1,proba2.predicted_probs)
    # ax = axs[0]
    # ax.imshow(proba1)
    # ax = axs[1]
    # ax.imshow(proba2.smoothed_probs)
    # plt.show()

    model_again = dynamaxhmm_to_hmmlearn(transformed_model,params)

    print(
        model_again.score(data,mode3d=True)
    )

    key = jr.PRNGKey(0)
    sgd_key, key = jr.split(key)
    sgd_params, sgd_losses = transformed_model.fit_sgd(
        params,
        param_props,
        jnp.asarray(all_data),
        optimizer=optax.sgd(
            learning_rate=1e-2,
            momentum=0.95
        ),
        batch_size=10,
        num_epochs=400,
        key = sgd_key,
    )
    print(sgd_losses)
    # print(transformed_model.num_states,transformed_model.emission_dim,transformed_model.__class__.__name__)




if __name__ == '__main__':
    resolvers = {
        'eval': eval,
        'ind': lambda a, i: a[i],
        'listmul': lambda l, i: [l] * i,
        'getattr': getattr,
        'setattrs': setattrs,
        'as_tuple': lambda *args: tuple(args),
    }
    for resolver_name, resolver_val in resolvers.items():
        if not OmegaConf.has_resolver(resolver_name):
            OmegaConf.register_new_resolver(resolver_name, resolver_val)
    main()
