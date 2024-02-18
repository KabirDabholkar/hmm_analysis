from functools import partial

import jax.numpy
import jax.numpy as jnp
import jax.random as jr
import matplotlib.pyplot as plt
from jax import vmap
from jax.nn import one_hot

from dynamax.hidden_markov_model import CategoricalHMM

# initial_probs = jnp.array([0.5, 0.5])
# transition_matrix = jnp.array([[0.95, 0.05],
#                                [0.10, 0.90]])
# emission_probs = jnp.array([[1/6,  1/6,  1/6,  1/6,  1/6,  1/6],    # fair die
#                             [1/10, 1/10, 1/10, 1/10, 1/10, 5/10]])  # loaded die
#
#
# print(f"A.shape: {transition_matrix.shape}")
# print(f"B.shape: {emission_probs.shape}")
#
# num_states = 2      # two types of dice (fair and loaded)
# num_emissions = 1   # only one die is rolled at a time
# num_classes = 6     # each die has six faces
#
# # Construct the HMM
# hmm = CategoricalHMM(num_states, num_emissions, num_classes)
#
# # Initialize the parameters struct with known values
# params, _ = hmm.initialize(initial_probs=initial_probs,
#                            transition_matrix=transition_matrix,
#                            emission_probs=emission_probs.reshape(num_states, num_emissions, num_classes))
#
# print(params)
#
# num_timesteps = 300
# true_states, emissions = hmm.sample(params, jr.PRNGKey(42), num_timesteps)
#
# print(f"true_states.shape: {true_states.shape}")
# print(f"emissions.shape: {emissions.shape}")
# print("")
# print("First few states:    ", true_states[:5])
# print("First few emissions: ", emissions[:5, 0])
#
# # To sample multiple sequences, just use vmap
# num_batches = 5
#
# batch_states, batch_emissions = \
#     vmap(partial(hmm.sample, params, num_timesteps=num_timesteps))(
#         jr.split(jr.PRNGKey(0), num_batches))
#
# print(f"batch_states.shape: {batch_states.shape}")
# print(f"batch_emissions.shape: {batch_emissions.shape}")
#
#
# posterior = hmm.filter(params, emissions)
# print(f"marginal likelihood: {posterior.marginal_loglik: .2f}")
# print(f"posterior.filtered_probs.shape: {posterior.filtered_probs.shape}")
#
# posterior = hmm.smoother(params, emissions)
# print(f"posterior.smoothed_probs.shape: {posterior.smoothed_probs.shape}")


from hmmlearn.hmm import PoissonHMM as hmmlearn_PoissonHMM
from dynamax.hidden_markov_model.models.poisson_hmm import PoissonHMM as jax_PoissonHMM
from dynamax.hidden_markov_model.models.bernoulli_hmm import BernoulliHMM as jax_BernoulliHMM
import jax.numpy as jnp

num_states = 4
emission_dim = 10

# model = jax_PoissonHMM(num_states,emission_dim)




def hmmlearn_to_dynamaxhmm(hmmlearn_model,):
    num_states   = hmmlearn_model.n_components
    emission_dim = hmmlearn_model.n_features
    hmm_class = {
        'BernoulliHMM': jax_BernoulliHMM,
        'PoissonHMM': jax_PoissonHMM,
    }[hmmlearn_model.__class__.__base__.__name__]
    emission_param_name = {
        'BernoulliHMM':'emission_probs',
        'PoissonHMM': 'emission_rates'
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




#
# key = jr.PRNGKey(0)
# key1, key2 = jr.split(key, 2)
# params, param_props = model.initialize(key1)
# params2, _ = model.initialize(key2)
#
# print(params)
#
# num_timesteps = 300
# true_states, emissions = model.sample(params2, jr.PRNGKey(42), num_timesteps)
#
#
# num_iters = 10
#
# params, losses = model.fit_em(params, param_props, emissions[None, ...], num_iters=num_iters)
#
# print(losses)
#
#

CONFIG_PATH = "configs"
CONFIG_NAME = "config_cohmm"

import hydra
from omegaconf import OmegaConf
from utils import setattrs
from config_utils import instantiate

@hydra.main(version_base='1.3', config_path=CONFIG_PATH, config_name=CONFIG_NAME)
def main(cfg):

    model = instantiate(cfg.teacher)

    transformed_model,params, param_props = hmmlearn_to_dynamaxhmm(model)
    data = instantiate(cfg.generate_all_data_dictmodule, _convert_='partial')(hmm_model=model)
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