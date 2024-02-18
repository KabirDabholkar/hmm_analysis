import jax
import jax.numpy as np
from jax import vmap,grad,value_and_grad,hessian
from compose import compose
from functools import partial
from metrics import bernoulli_neg_log_likelihood
from jax.scipy.special import gammaln

def compute_bernoulli_MLE(Y_heldout, posteriors,eps=1e-8):
    obs_zero = posteriors.T @ (~Y_heldout).astype(float)
    obs_one = posteriors.T @ (Y_heldout).astype(float)
    lambdas_ = obs_one / (obs_zero + obs_one + eps)
    return lambdas_

def compute_poisson_MLE(Y_heldout, posteriors,eps=1e-8):
    print(Y_heldout.shape,posteriors.shape)
    obs = posteriors.T @ (Y_heldout).astype(float)
    post = posteriors.sum(axis=0)[:,None]
    lambdas_ = obs / post
    return lambdas_

batch_compute_bernoulli_MLE = vmap(compute_bernoulli_MLE,(0,0),0)


batch_compute_poisson_MLE = compose(
    vmap(compute_poisson_MLE,(0,0),0),
)


def bernoulli_neg_log_likelihood(rates, spikes,eps=1e-9):
    # rates_spiked_only_nonzero = rates * spikes
    # rates_notspiked_only_nonzero = (1-rates) * ~spikes
    # print('rates',rates)
    # fspikes = spikes.astype(float)
    rates_ = spikes.astype(float) * (rates) + (~spikes).astype(float) * (1 - rates)
    #(fspikes) + (-2*fspikes+1) * rates
    # print(rates_)
    result = - np.log(rates_+eps)
    return np.sum(result)

def bernoulli_loglikelihood_loss(Y_heldout, posteriors, phi):
    rate_pred = posteriors @ phi
    return bernoulli_neg_log_likelihood(rate_pred,Y_heldout)

def poission_neg_log_likelihood(rates, spikes):
    """
    adapted from nlb_tools
    """
    result = rates - spikes * np.log(rates) + gammaln(spikes + 1.0)
    return np.sum(result)


def get_loglikelihood_loss(neg_log_likelihood_func):
    def loglikelihood_loss(Y_heldout, posteriors, phi):
        rate_pred = posteriors @ phi
        return neg_log_likelihood_func(rate_pred, Y_heldout)
    return loglikelihood_loss

def batchify_over_samples_loglikelihood_loss(loglikelihood_loss):
    return compose(
        np.sum,
        vmap(loglikelihood_loss,(0,0,None),0)
    )

def batchify_over_params_loglikelihood_loss(loglikelihood_loss):
    return vmap(loglikelihood_loss,(None,None,0),0)



def get_fisher_info_func(batch_grad):
    def fisher_info(Y_heldout, posteriors, phi):
        grads = batch_grad(Y_heldout, posteriors, phi)
        grads = grads.reshape(grads.shape[0],-1)
        return (grads[:,:,None] @ grads[:,None,:]).mean(0)
    return fisher_info

bernoulli_loglikelihood_loss,poisson_loglikelihood_loss = [
    get_loglikelihood_loss(func)
    for func in [bernoulli_neg_log_likelihood,poission_neg_log_likelihood]
]



function_dict = {}
for key,func in zip(['bernoulli','poisson'],[bernoulli_neg_log_likelihood,poission_neg_log_likelihood]):
    function_dict[key] = {}
    function_dict[key]['loglikelihood_loss'] = get_loglikelihood_loss(func)
    function_dict[key]['batch_loglikelihood_loss'] = batchify_over_samples_loglikelihood_loss(
                                                        function_dict[key]['loglikelihood_loss']
                                                    )
    function_dict[key]['batch_batch_loglikelihood_loss'] = batchify_over_params_loglikelihood_loss(
                                                            function_dict[key]['batch_loglikelihood_loss']
                                                        )
    function_dict[key]['value_and_grad_batch_loglikelihood_loss'] = vmap(
        function_dict[key]['batch_loglikelihood_loss'],
        (None,None,0),0
    )
    function_dict[key]['individual_grad'] = grad(function_dict[key]['loglikelihood_loss'],argnums=2)
    function_dict[key]['batch_grad'] = vmap(function_dict[key]['individual_grad'], (0, 0, None), 0)
    function_dict[key]['indiv_hessian'] = hessian(function_dict[key]['loglikelihood_loss'],argnums=2)
    function_dict[key]['batch_hessian'] = compose(
                                                partial(np.mean,axis=0),
                                                vmap(function_dict[key]['indiv_hessian'],(0,0,None),0)
                                            )
    function_dict[key]['batch_fisher_info'] = get_fisher_info_func(function_dict[key]['batch_grad'])


function_dict['bernoulli']['compute_MLE'] = compute_bernoulli_MLE
function_dict['poisson']  ['compute_MLE'] = compute_poisson_MLE

for key in ['bernoulli','poisson']:
    function_dict[key]['batch_compute_MLE'] = vmap(function_dict[key]['compute_MLE'],(0,0),0)


batch_bernoulli_loglikelihood_loss = compose(
    np.sum,
    vmap(bernoulli_loglikelihood_loss,(0,0,None),0)
)

batch_batch_bernoulli_loglikelihood_loss = vmap(batch_bernoulli_loglikelihood_loss,(None,None,0),0)

value_and_grad_batch_bernoulli_loglikelihood_loss = value_and_grad(batch_bernoulli_loglikelihood_loss,argnums=2)

indiv_grad = grad(bernoulli_loglikelihood_loss,argnums=2)

batch_grad = vmap(indiv_grad, (0, 0, None), 0)


indiv_real_hess = hessian(bernoulli_loglikelihood_loss,argnums=2)
batch_real_hess = compose(
    partial(np.mean,axis=0),
    vmap(indiv_real_hess,(0,0,None),0)
)

def main():
    pass


if __name__ == '__main__':
    main()