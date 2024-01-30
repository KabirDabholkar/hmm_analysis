import jax
import jax.numpy as np
from jax import vmap,grad,value_and_grad,hessian
from compose import compose
from functools import partial
from metrics import bernoulli_neg_log_likelihood

def compute_bernoulli_MLE(Y_heldout, posteriors,eps=1e-8):
    obs_zero = posteriors.T @ (~Y_heldout).astype(float)
    obs_one = posteriors.T @ (Y_heldout).astype(float)
    lambdas_ = obs_one / (obs_zero + obs_one + eps)
    return lambdas_

batch_compute_bernoulli_MLE = vmap(compute_bernoulli_MLE,(0,0),0)


batch_compute_bernoulli_MLE = compose(
    vmap(compute_bernoulli_MLE,(0,0),0),
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
    # print(np.isnan(rate_pred).any())
    # return rate_pred
    return bernoulli_neg_log_likelihood(rate_pred,Y_heldout)

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


def batch_hess(Y_heldout, posteriors, phi):
    grads = batch_grad(Y_heldout, posteriors, phi)
    grads = grads.reshape(grads.shape[0],-1)
    return (grads[:,:,None] @ grads[:,None,:]).mean(0)
