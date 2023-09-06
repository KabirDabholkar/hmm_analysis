from hidden_markov import hmm
import numpy as np


def main():
    states = ('s', 't')
    possible_observation = ('A', 'B')
    start_probability = np.matrix('0.5 0.5 ')
    transition_probability = np.matrix('0.6 0.4 ;  0.3 0.7 ')
    emission_probability = np.matrix('0.3 0.7 ; 0.4 0.6 ')
    ground_truth = hmm(states, possible_observation, start_probability, transition_probability, emission_probability)

    N = 100
    train_observations = [list(ground_truth.sample(T=200))[1] for _ in range(N)]
    #print(train_observations)
    quantities = [10]*N


    start_probability = np.matrix('0.5 0.5 ')
    # transition_probability = np.matrix('0.55 0.45 ;  0.2 0.8 ')
    # emission_probability = np.matrix('0.2 0.8 ; 0.3 0.7 ')
    transition_probability = np.matrix('0.6 0.4 ;  0.3 0.7 ')
    emission_probability = np.matrix('0.3 0.7 ; 0.45 0.55 ')
    model = hmm(states, possible_observation, start_probability, transition_probability, emission_probability)

    print('log_prob before training',model.log_prob(train_observations,quantities))
    model.train_hmm(train_observations,2000,quantities)
    print('log_prob after training',model.log_prob(train_observations,quantities))


    print('em_prob\n',model.em_prob)
    print('trans_prob\n',model.trans_prob)
    print('start_prob\n',model.start_prob)


if __name__ == '__main__':
    main()


