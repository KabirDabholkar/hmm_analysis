from dynamax.hidden_markov_model import GaussianHMM
# from dynamax.hidden_markov_model import BernoulliHMM as GaussianHMM

TRUE = true = True
FALSE = false = False

DEBUG = false

NUM_TRAIN_BATCHS  = 200
NUM_TEST_BATCHS    = 1

NUM_EPOCHS          = 3 if DEBUG else 20 #Increase the pochs before addin , num_iters=NUM_EPOCHS to fit_em or num_epochs=... for fit_sgd
ITER                = 20 if DEBUG else 400 #Num of iterations per epoch
NUM_TIMESTEPS       = 10 if DEBUG else 10#100
NUM_TRIALS          = 10 if DEBUG else 100#100
STUDENTS_NUM        = 1 if DEBUG else 2 # ring = true will double the amount

SGD = DEBUG

'HMM Type and settings'
EMISSION_DIM    = 2
TRUE_NUM_STATES = 2
MIN_S_STATE     = TRUE_NUM_STATES + 2
MAX_S_STATE     = TRUE_NUM_STATES + 2
epsilon         = 0.01
HMM = GaussianHMM(TRUE_NUM_STATES, EMISSION_DIM)
# S_KEYS  = ['0', '1', '00', '01', '11'] #TODO
S_KEYS  = [i for i in range(NUM_EPOCHS)]

LEARNING_RATE = 1e-2 # 1e-3 is the default step size



###################

from sklearn.mixture import GaussianMixture


def likelihood(student_type, student_params, teacher, teacher_train_obs, test_obs):
    baseline_model = GaussianMixture(n_components=1)
    base = lambda train, test: baseline_model.fit(
        train.reshape(-1, EMISSION_DIM)
    ).score_samples(
        test.reshape(-1, EMISSION_DIM)
    ).reshape(-1, NUM_TIMESTEPS).sum(axis=1).mean(axis=0)
    evaluate_func = lambda hmm_class: vmap(hmm_class.marginal_log_prob, [None, 0], 0)  # evaluate
    ev = lambda hmm, features, test: (evaluate_func(hmm)(features, test)).mean()  # eval_true

    if DEBUG:
        def evaluate_model(hmm, model, test_data, base=False):
            #     if base:
            #         true_loss = vmap(partial(hmm.marginal_log_prob, model))(test_data).sum()
            #         # true_loss = vmap(partial(hmm._estimate_log_prob  else hmm., model))(test_data).sum()
            #     true_loss += hmm.log_prior(model)
            #     true_loss = -true_loss / test_data.size
            #     return true_loss
            # base = lambda train,test: baseline_model.fit(
            #     train.reshape(-1, EMISSION_DIM)
            #     ).score_samples(
            #         test.reshape(-1, EMISSION_DIM)
            #     )
            pass

        print(f'Base liklihood: {base(teacher_train_obs, test_obs)}')
        print(f'Teacher liklihood: {ev(HMM, teacher, test_obs)}')
        print(f'Student liklihood: {ev(student_type, student_params, test_obs)} \n\n')
        print(f'Student train liklihood: {ev(student_type, student_params, teacher_train_obs)} \n\n')

        # print(f'Teacher new eval: {evaluate_model(HMM, teacher, test_obs)} \n\n')
        # print(f'Student new eval: {evaluate_model(student_type, student_params, test_obs)} \n\n')
        # print(f'Student new eval train: {evaluate_model(student_type, student_params, teacher_train_obs)} \n\n')
        # print(f'Base new eval: {evaluate_model(baseline_model, base(teacher_train_obs, test_obs), test_obs, base=True)} \n\n')

    student_score = ev(student_type, student_params, test_obs)
    base_score = base(teacher_train_obs, test_obs)
    teacher_score = ev(HMM, teacher, test_obs)
    return float((student_score - base_score) / (teacher_score - base_score))

################


from functools import partial
from jax import vmap
import jax.numpy as jnp
import jax.random as jr
# from hmmST import likelihood
# from macros import *
import optax

data_sample = lambda model, params, key:  model.sample(params, jr.PRNGKey(key), NUM_TIMESTEPS)

fit = lambda hmm_class, params, props, emissions : hmm_class.fit_sgd(params, props, emissions, optimizer=optax.adam(LEARNING_RATE),num_epochs=ITER,batch_size=200)

def evaluate_model(hmm, model, test_data):
    true_loss = vmap(partial(hmm.marginal_log_prob, model))(test_data).sum()
    true_loss += hmm.log_prior(model)
    true_loss = -true_loss / test_data.size
    return true_loss

def norm_loss(hmm, model, test_data, true_hmm, true_model):
    return (evaluate_model(hmm, model, test_data)-2000) / (evaluate_model(true_hmm, true_model, test_data)-2000)


evaluate_func = lambda hmm_class : vmap(hmm_class.marginal_log_prob, [None, 0], 0) #evaluate
ev = lambda hmm, features, test: (evaluate_func(hmm)(features, test)).mean() #eval_true


hmm = GaussianHMM(TRUE_NUM_STATES, EMISSION_DIM)
T, _ = hmm.initialize(jr.PRNGKey(10))
S, S_props  = hmm.initialize(jr.PRNGKey(0),emission_covariances=jnp.eye(TRUE_NUM_STATES)[None])

# _, train_data  = data_sample(hmm, T, 0)
# _, test_data    = data_sample(hmm, T, 1)


# NUM_TRAIN_BATCHS = 5
# NUM_TIMESTEPS = 10

states_num, train_data = vmap(partial(hmm.sample, T, num_timesteps=NUM_TIMESTEPS))(
        jr.split(jr.PRNGKey(42), NUM_TRAIN_BATCHS))

_, test_data = \
    vmap(partial(hmm.sample, T, num_timesteps=NUM_TIMESTEPS))(
        jr.split(jr.PRNGKey(99), NUM_TEST_BATCHS))

# print(train_data.shape)
# print(test_data.shape)



for i in range(1):
    # print(f'iteration {i}, train: ', ev(hmm, S, train_data))
    # print(f'iteration {i}, test: ', ev(hmm, S, test_data))
    # print(f'iteration {i}, likelihood: ', likelihood(hmm, S, T, train_data, test_data))

    # print(f'iteration {i}, train: ', evaluate_model(hmm, T, test_data))
    # print(f'iteration {i}, test: ', loss())
    S, losses = fit(hmm, S, S_props, train_data)

print(f'iteration {i+1}, likelihood: ', likelihood(hmm, S, T, train_data, test_data))
# print(f'Teacher likelihood: ', likelihood(hmm, T, T, train_data, test_data))



import matplotlib.pyplot as plt
plt.figure()
plt.plot(losses)
plt.yscale('log')
plt.show()


plt.figure()
plt.plot(train_data[0,:,:2])
plt.show()

_, S_sample_data = vmap(partial(hmm.sample, S, num_timesteps=NUM_TIMESTEPS))(
        jr.split(jr.PRNGKey(42), NUM_TRAIN_BATCHS))

plt.figure()
plt.scatter(train_data[:20,:,0],train_data[:20,:,1],label='T')
plt.scatter(S_sample_data[:20,:,0],S_sample_data[:20,:,1],label='S')
plt.legend()
plt.show()