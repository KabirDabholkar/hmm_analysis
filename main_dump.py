def main():
    rs = check_random_state(546)

    model = GaussianHMM(4, init_params="")
    model.n_features = 4
    model.startprob_ = np.array([1 / 4., 1 / 4., 1 / 4., 1 / 4.])
    model.transmat_ = np.array([[0.3, 0.4, 0.2, 0.1],
                                [0.1, 0.2, 0.3, 0.4],
                                [0.5, 0.2, 0.1, 0.2],
                                [0.25, 0.25, 0.25, 0.25]])
    model.means_ = np.array([[-2.5], [0], [2.5], [5.]])
    model.covars_ = np.sqrt([[0.25], [0.25], [0.25], [0.25]])


    X, _ = model.sample(1000, random_state=rs)
    lengths = [X.shape[0]]


    aic = []
    bic = []
    lls = []
    ns = [2, 3, 4, 5, 6]
    for n in ns:
        p
        best_ll = None
        best_model = None
        for i in range(10):
            h = GaussianHMM(n, n_iter=200, tol=1e-2, random_state=rs)
            h.fit(X)
            score = h.score(X)
            if not best_ll or best_ll < best_ll:
                best_ll = score
                best_model = h
        aic.append(best_model.aic(X))
        bic.append(best_model.bic(X))
        lls.append(best_model.score(X))

    fig, ax = plt.subplots()
    ln1 = ax.plot(ns, aic, label="AIC", color="blue", marker="o")
    ln2 = ax.plot(ns, bic, label="BIC", color="green", marker="o")
    ax2 = ax.twinx()
    ln3 = ax2.plot(ns, lls, label="LL", color="orange", marker="o")

    ax.legend(handles=ax.lines + ax2.lines)
    ax.set_title("Using AIC/BIC for Model Selection")
    ax.set_ylabel("Criterion Value (lower is better)")
    ax2.set_ylabel("LL (higher is better)")
    ax.set_xlabel("Number of HMM Components")
    fig.tight_layout()

    plt.show()

def main():
    rs = check_random_state(546)

    GT = CategoricalHMM(n_components=2, init_params="")
    GT.n_features = 2
    GT.startprob_ = np.array([1 / 2., 1 / 2.])
    GT.transmat_ = np.array([[0.7, 0.3],
                                [0.2, 0.8]])

    GT.emissionprob_ = np.array([[0.3,0.7],
                                 [0.7,0.3]])
    # GT.means_ = np.array([[-2.5], [0]])
    # GT.covars_ = np.sqrt([[0.25], [0.25]])
    # print(GT.means_)

    X = [GT.sample(30)[0] for _ in range(100)]
    lengths = [x.shape[0] for x in X]
    X = np.concatenate(X)

    X_test = [GT.sample(30)[0] for _ in range(100)]
    test_lengths = [x.shape[0] for x in X_test]
    X_test = np.concatenate(X_test)

    print(lengths)


    GT_sc = GT.score(X_test,lengths=test_lengths)


    model = VariationalCategoricalHMM(n_components=10, init_params="",n_iter=100,tol=1e-2,verbose=True)
    model.n_features = 2
    # model.startprob_ = np.array([1 / 2., 1 / 2.])
    # model.transmat_ = np.array([[0.7, 0.3],
    #                             [0.2, 0.8]])
    # model.emissionprob_ = np.array([[0.4, 0.6], [0.4, 0.6]])
    # model_sc = model.score(X_test,lengths=lengths)

    model.fit(X,lengths=lengths)
    model_sc_fit = model.score(X_test,lengths=test_lengths)
    print('Ground truth score',GT_sc)
    #print('Model score',model_sc)
    print('Model score fit', model_sc_fit)
    #print(model.emissionprob_)
    #print(model.monitor_.history[1:])
    plt.plot(np.array(model.monitor_.history)[1:],label='training')
    plt.axhline(GT_sc,ls='dashed',label='ground truth',c='red')
    plt.axhline(model_sc_fit, ls='dashed',label='fit',c='green')
    plt.legend()
    plt.show()



