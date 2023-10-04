import numpy as np

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
