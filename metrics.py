import numpy as np
import logging
logger = logging.getLogger(__name__)

def self_consistency_score(model,X_lengths,window_length=10,eps=1e-3):
    X_val, lengths_val = X_lengths
    N = sum(lengths_val)

    #D_JS, D_JS_shuffled = self_consistency(model, X_val, window_length, Nsamples=Nsamples)
    D_JS, D_JS_shuffled =  self_consistency_over_sliced_data(model, X_val, lengths_val, window_length)
    model_SC = np.mean(D_JS) / (np.mean(D_JS_shuffled) + eps)
    return model_SC

def normalised_score(model,X_lengths,window_length=10):
    X_val, lengths_val = X_lengths
    N = sum(lengths_val)
    score = model.score(X_val, lengths=lengths_val)
    score /= N
    return score


def normalised_cosmoothing_score2(model,X_lengths,window_length=10):
    X_val, lengths_val = X_lengths
    X_val_heldin, X_val_heldout = X_val
    N = sum(lengths_val)
    # score = model.score(X_val, lengths=lengths_val)
    if lengths_val:
        cummu_lengths = np.concatenate([np.zeros((1), dtype=int), np.cumsum(np.array(lengths_val, dtype=int))])
        latent_prob = np.concatenate([model.predict_proba(
            X_val_heldin[cummu_lengths[i]:cummu_lengths[i+1]]
        ) for i in range(len(lengths_val))])
    else:
        latent_prob = model.predict_proba(X_val)
    n_feat = model.n_features
    pred = (latent_prob @ model.emissionprob2_)
    score = np.log(
        (pred * (X_val_heldout == np.arange(n_feat)[None,:]).astype(float)).sum(1)
    ).sum()
    score /= N
    return score


def normalised_cosmoothing_score(model,X_lengths,window_length=10):
    X_val, lengths_val = X_lengths
    N = sum(lengths_val)
    # score = model.score(X_val, lengths=lengths_val)
    if lengths_val:
        cummu_lengths = np.concatenate([np.zeros((1), dtype=int), np.cumsum(np.array(lengths_val, dtype=int))])
        latent_prob = np.concatenate([model.predict_proba(
            X_val[cummu_lengths[i]:cummu_lengths[i+1]]
        ) for i in range(len(lengths_val))])
    else:
        latent_prob = model.predict_proba(X_val)
    n_feat = model.n_features
    pred = (latent_prob @ model.emissionprob2_)
    score = np.log(
        (pred * (X_val == np.arange(n_feat)[None,:]).astype(float)).sum(1)
    ).sum()
    score /= N
    return score


def bernoulli_neg_log_likelihood(rates, spikes, zero_warning=True):
    """Calculates Poisson negative log likelihood given rates and spikes.
    formula: -log(e^(-r) / n! * r^n)
           = r - n*log(r) + log(n!)

    Parameters
    ----------
    rates : np.ndarray
        numpy array containing rate predictions
    spikes : np.ndarray
        numpy array containing true spike counts
    zero_warning : bool, optional
        Whether to print out warning about 0 rate
        predictions or not

    Returns
    -------
    float
        Total negative log-likelihood of the data
    """
    assert spikes.shape == rates.shape, \
        f"neg_log_likelihood: Rates and spikes should be of the same shape. spikes: {spikes.shape}, rates: {rates.shape}"

    if np.any(np.isnan(spikes)):
        mask = np.isnan(spikes)
        rates = rates[~mask]
        spikes = spikes[~mask]


    assert not np.any(np.isnan(rates)), \
        "neg_log_likelihood: NaN rate predictions found"

    assert np.all(rates >= 0), \
        "neg_log_likelihood: Negative rate predictions found"
    assert np.all(rates <= 1), \
        "neg_log_likelihood: predictions larger than 1.0 found"
    if (np.any(rates == 0)):
        if zero_warning:
            logger.warning("neg_log_likelihood: Zero rate predictions found. Replacing zeros with 1e-9")
        rates[rates == 0] = 1e-9

    if (np.any(rates == 1)):
        if zero_warning:
            logger.warning("neg_log_likelihood: One rate predictions found. Replacing Ones with 1.0-1e-9")
        rates[rates == 1] = 1.0-1e-9

    result = - np.concatenate([np.log(rates)[spikes==1],np.log(1-rates)[spikes==0]])
    return np.sum(result)


def bernoulli_bits_per_spike(rates, spikes):
    """Computes bits per spike of rate predictions given spikes.
    Bits per spike is equal to the difference between the log-likelihoods (in base 2)
    of the rate predictions and the null model (i.e. predicting mean firing rate of each neuron)
    divided by the total number of spikes.

    Parameters
    ----------
    rates : np.ndarray
        3d numpy array containing rate predictions
    spikes : np.ndarray
        3d numpy array containing true spike counts

    Returns
    -------
    float
        Bits per spike of rate predictions
    """
    nll_model = bernoulli_neg_log_likelihood(rates, spikes)
    nll_null = bernoulli_neg_log_likelihood(
        np.tile(np.nanmean(spikes, axis=(0, 1), keepdims=True), (spikes.shape[0], spikes.shape[1], 1)), spikes,
        zero_warning=False)
    return (nll_null - nll_model) / np.nansum(spikes) / np.log(2)
