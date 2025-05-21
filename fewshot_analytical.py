import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
from functools import partial
from compose import compose

from sympy import symbols, Eq, solve, sqrt
mpl.rcParams['text.usetex'] = True
plt.rcParams["font.family"] = "serif"
plt.rcParams["mathtext.fontset"] = "dejavuserif"
mpl.rcParams['text.latex.preamble'] = r'\usepackage{amsmath}'

# Parameters
sigCoding = 0.3



# N_range_list = np.arange(10,31,2)  #
N_range_list = np.logspace(1,3,10).astype(int) #range(N_range_min, N_range_max+1)
# N_range_list = np.concatenate([np.arange(10,30,5),N_range_list])
N_range_list = np.unique(np.sort(N_range_list))
N_range_list = (N_range_list//2)*2
numNs = len(N_range_list)
repeats = 500 #00
N_test = 1000
theory = None
ylabel = None
ylim = None

def make_gaussian_signal(N,x_dim=1,sig_x=1):
    return np.random.normal(size=(N,x_dim)) * sig_x

def make_binary_signal(N):
    N1 = N//2
    N2 = N-N1
    # (np.random.choice(2,N)[:,None] * 2 - 1).astype(int) #.astype(float)
    return np.random.permutation(np.concatenate([np.ones(N1),-np.ones(N2)]))[:,None].astype(int)

def concatnoise_encoder(x,n_dim = 5,sigExtra=1,normalise=False):
    N = x.shape[0]
    n = np.random.randn(N, n_dim)
    if normalise:
        n = n/np.linalg.norm(n,axis=1)[:,None]
    n = n * sigExtra
    return np.concatenate([x,n],axis=-1)

def add_multivariate(x,sigmas,normalise=False):
    noise = np.random.multivariate_normal(mean=np.zeros(x.shape[-1]),cov=np.eye(x.shape[-1]),size=x.shape[:-1])
    if normalise:
        noise /= np.sqrt( np.mean( noise**2, axis=-1, keepdims=True ) )
        # noise /= np.linalg.norm(noise,axis=-1,keepdims=True)
        # noise *= np.sqrt(noise.shape[-1])
    noise *= sigmas[None,:]
    # print('noise',noise)
    # print('x',x)
    return x + noise

def flatHMMlatent(x):
    N = x.shape[0]
    return 0.5 * np.ones((N,2))

def chainHMMlatent(x,num_states=2):
    N = x.shape[0]
    # np.stack([(np.arange(2)==i)]*(N//num_states) for i in range(num_states))
    return np.arange(num_states)[None,:] == np.random.choice(num_states,size=(N,))[:,None]



def add_classdependent_multivariate(x,sigmas_a,sigmas_b,normalise=False):
    noise = np.random.multivariate_normal(mean=np.zeros(x.shape[-1]),cov=np.eye(x.shape[-1]),size=x.shape[:-1])
    if normalise:
        noise /= np.sqrt( np.mean( noise**2, axis=-1, keepdims=True ) )
        # noise /= np.linalg.norm(noise,axis=-1,keepdims=True)
        # noise *= np.sqrt(noise.shape[-1])

    noise *= sigmas_a[None,:]*(x>0).any(axis=-1,keepdims=True) + sigmas_b[None,:]*(x<0).any(axis=-1,keepdims=True)
    # print('noise',noise)
    # print('x',x)
    return x + noise

def add_noise(x, sigExtra=1):
    N = x.shape[0]
    return x + np.random.normal(size=x.shape) * sigExtra

def random_position_encoder(x,choice_range = 2):
    choice = np.random.choice(choice_range,size=x.shape[0],replace=True)
    one_hots = (np.arange(choice_range)[None]==choice[:,None]).astype(float)
    z = x * one_hots
    return z

def random_angle_encoder(x,angle_range = (0,np.pi/2)):
    angles = np.random.uniform(low=angle_range[0],high=angle_range[1],size=x.shape[0])
    z = x * np.stack([np.cos(angles),np.sin(angles)],axis=1)
    return z

def binary_angle_encoder(x,angle_range = (0,np.pi/2)):
    choice = np.random.choice(2, size=x.shape[0], replace=True)
    angles = (choice==0).astype(float) * angle_range[0] + (choice==1).astype(float) * angle_range[1]
    z = x * np.stack([np.cos(angles),np.sin(angles)],axis=1)
    # eps = 1e-5
    # z = z * (np.abs(z)>eps).astype(float)
    # z += np.random.normal(size=z.shape) * 5e-2
    return z



class LinearRegression():
    def __init__(self,threshold_output=False):
        self.a = None
        self.threshold_output = threshold_output

    def fit(self, inputs, targets):
        self.a = np.linalg.lstsq(inputs, targets)[0]
        return self.a

    def predict(self, inputs):
        preds = inputs @ self.a
        if self.threshold_output:
            preds = np.sign(preds)
        return preds


class PrototypeLearning():
    def __init__(self):
        self.a = None

    def fit(self, inputs, targets):
        xa= inputs[targets[:, 0] == 1, :].mean(axis=0)
        xb = inputs[targets[:, 0] == -1].mean(axis=0)
        self.a = xa - xb
        self.a = self.a[:,None]
        self.bias = self.a.T @ (xa+xb) / 2
        return self.a

    def predict(self, inputs):
        return np.sign(inputs @ self.a - self.bias)


class BinomialMLE():
    def __init__(self,eps=1e-4):
        self.B = None
        self.eps = eps

    def fit(self,inputs,targets):
        # print(inputs.shape,targets.shape)
        numerator = ((targets == 1) * inputs).sum(0)
        denominator = inputs.sum(0)
        self.B = (numerator / (denominator + self.eps))

    def predict(self,inputs):
        return inputs @ self.B[:,None]

### select settings


# encoder_type = "concatnoise"
# title = None
# loss = lambda pred,target: np.mean((pred - target)**2)
# signal_generator = partial(make_gaussian_signal,x_dim = 1,sig_x=1)
# ModelClass = LinearRegression
# sigExtra = [0.1, 1]
# encoders = [
#     partial(concatnoise_encoder,n_dim = 2,sigExtra=sigExtra[j])
#     for j in range(len(choice_range))
# ]
# labels=[
#     r'$\sigma_\text{ext}(z)=$' + f'{sigExtra[j]:.1f}'
#     for j in range(len(choice_range))
# ]

# encoder_type = "randposition"
# regression_type = 'LinearRegression' #+ 'Threshold'
# title = None
# loss = lambda pred,target: np.mean((pred - target)**2)
# signal_generator = partial(make_gaussian_signal,x_dim = 1,sig_x=1)
# ModelClass = LinearRegression
# choice_range = [1, 2]
# encoders = [
#     compose(
#         # partial(add_noise,sigExtra=0.05),
#         partial(random_position_encoder,choice_range=choice_range[j]),
#     )
#     for j in range(len(choice_range))
# ]
# labels=[
#     r'$\text{dim}(z)=$' + f'{choice_range[j]:d}'
#     for j in range(len(choice_range))
# ]

# encoder_type = "randangle"
# title = None
# loss = lambda pred,target: np.mean((pred - target)**2)
# signal_generator = partial(make_gaussian_signal,x_dim = 1,sig_x=1)
# ModelClass = LinearRegression
# angle_ranges = np.arange(0.1,0.51,0.1)
# angle_ranges_list = list(np.stack([np.zeros_like(angle_ranges),angle_ranges * np.pi],axis=1))
# encoders = [
#     partial(random_angle_encoder,angle_range=angle_range)
#     for angle_range in angle_ranges_list
# ]
# labels=[
#     rf'$\theta\sim Unif(0,{angle:.1f}\pi)$'
#     for angle in angle_ranges
# ]

# encoder_type = "binaryangle"
# title = None
# regression_type = 'LinearRegression' #+ 'Threshold'
# loss = lambda pred,target: np.mean((pred - target)**2)
# signal_generator = partial(make_gaussian_signal,x_dim = 1,sig_x=1)
# ModelClass = LinearRegression
# angle_ranges = np.arange(0.0,0.51,0.1)
# angle_ranges_list = list(np.stack([np.zeros_like(angle_ranges),angle_ranges * np.pi],axis=1) + np.pi*0.2)
# encoders = [
#     compose(
#         partial(add_noise, sigExtra=0.001),
#         partial(binary_angle_encoder,angle_range=angle_range)
#     )
#     for angle_range in angle_ranges_list
# ]
# labels=[
#     rf'$\theta=0$ or $\theta={angle:.1f}\pi$'
#     for angle in angle_ranges
# ]


#########

encoder_type = "binaryclassification_21dim"
sigCoding = 0.0
repeats = 20
title = None
ylabel = r"Average $k$-shot"+"\n"+"classification error $\epsilon$"
# ModelClass = partial(LinearRegression) # ,threshold_output=True)
# regression_type = 'LinearRegression' #+ 'Threshold'
ModelClass = PrototypeLearning
regression_type = 'PrototypeLearning'
# title = regression_type + "\n" + r"$x \sim \text{Rademacher}(0.5); z:=[x+n,m_1,\dots,m_M]^T;\\ n\sim \mathcal N(0,0.5^2); m_i\sim \mathcal N(0,\sigma_m^2)$"
loss = lambda pred,target: np.mean(pred != target)
# loss = lambda pred, target: np.mean((pred - target)**2)
signal_generator = make_binary_signal
sigExtras = [3,5,10] #[0.5, 0.75, 1, 1.5, 2]
sigSignal = 0.0
M = 20

encoders = [
    compose(
        # partial(concatnoise_encoder, n_dim=M, sigExtra=sigExtra,normalise=True),
        # partial(add_noise, sigExtra=0.5)
        # partial(add_multivariate,
        #         sigmas=np.array([0] + [sigExtra] * M
        #                         # + list(sigExtra * np.arange(1,M+1).astype(float)**(-0.5) )
        #                         )),#,normalise=True),

        # partial(add_classdependent_multivariate,
        #         sigmas_a=np.array([0.0] + [sigExtra] * (M//2) + [0.0] * (M//2)),
        #         sigmas_b=np.array([0.0] + [0.0] * (M//2) + [sigExtra] * (M//2))
        #     ),  # ,normalise=True),
        # lambda x: x @ (np.arange(M+1)==0)[None].astype(float)/np.sqrt(2)

        partial(add_classdependent_multivariate,
                sigmas_a=np.array([0.0] + [sigExtra] * (M) + [0.0] * (M))/np.sqrt(M),
                sigmas_b=np.array([0.0] + [0.0] * (M) + [sigExtra] * (M))/np.sqrt(M)
            ),  # ,normalise=True),
        lambda x: x @ (np.arange(2*M+1)==0)[None].astype(float)/np.sqrt(2)
    )
    for sigExtra in sigExtras
]
labels=[
    # rf'$\sigma_{ext}={sigExtra:.2f}$'
    rf'$\sigma_{{\text{{ext}}}}={sigExtra:.1f}$'
    for sigExtra in sigExtras
]

theory_k = np.logspace(1,3,100)

# from scipy.stats import norm
#
# def H(x):
#     return norm.sf(x)  # Survival function (1 - CDF)

from scipy.special import erfc

def H(x):
    return 1/2*erfc(x/np.sqrt(2))

from numpy import random
def few_shot_err(N, D, P, m, Ra, Rb, theta=np.pi / 2):
    # Subspaces
    U = random.randn(N, 2 * (D + 1))
    U, _ = np.linalg.qr(U)
    Ua, Ub, x0a, x0b = np.split(U, (D, 2 * D, 2 * D + 1), axis=-1)

    # Center-subspace overlaps
    x0a = np.sin(theta) * x0a + np.cos(theta) * Ua.sum(-1, keepdims=True) / np.sqrt(D)
    x0a /= np.sqrt(2)
    x0b = np.sin(theta) * x0b + np.cos(theta) * Ub.sum(-1, keepdims=True) / np.sqrt(D)
    x0b /= np.sqrt(2)



    # Training examples
    sa = random.randn(D, P, m) / np.sqrt(D)
    #     sa /= np.linalg.norm(sa,axis=0)
    sb = random.randn(D, P, m) / np.sqrt(D)
    #     sb /= np.linalg.norm(sb,axis=0)
    Xatrain = x0a + Ra * Ua @ sa.mean(-1)
    Xbtrain = x0b + Rb * Ub @ sb.mean(-1)
    Xtrain = np.stack([Xatrain, Xbtrain])

    print('xatrain.shape', Xatrain.shape)
    print('norm delta_x', np.linalg.norm(x0a - x0b))

    # Testing examples
    ssa = random.randn(D, P) / np.sqrt(D)
    #     ssa /= np.linalg.norm(ssa,axis=0)
    ssb = random.randn(D, P) / np.sqrt(D)
    #     ssb /= np.linalg.norm(ssb,axis=0)
    Xatest = x0a + Ra * Ua @ ssa
    Xbtest = x0b + Rb * Ub @ ssb

    print('xatest.shape', Xatest.shape)
    print('x0a.shape', x0a.shape)
    print('norm (Xatest-x0a)', np.linalg.norm(Xatest - x0a,axis=(0)).mean())

    # Evaluate
    da = ((Xatest - Xtrain) ** 2).sum(1)
    #     db = ((Xbtest - Xtrain)**2).sum(1)
    erra = (da.argmin(0) != 0).mean()
    #     errb = (db.argmin(0)!=1).mean()
    #     err = (erra+errb)/2

    return erra

def SNR_func(signal, bias, D, overlap, m):
    return 1/2*(signal + bias/m) / np.sqrt(1/D/m + overlap*(1+1/m) + 1/D/m**2)

theory = []
for sigext in sigExtras:
    # radii = np.array([0]
    #                  + [sigext] * M
    #                  #+ list(sigext * np.arange(1,M+1).astype(float)**(-0.5))
    # )
    # radii_a = radii
    # radii_b = radii
    # radii_a = np.array([0.0] + [sigext] * (M // 2) + [0.0] * (M // 2))
    # radii_b = np.array([0.0] + [0.0] * (M // 2) + [sigext] * (M // 2))
    radii_a = np.array([0.0] + [sigext] * (M) + [0.0] * (M))
    radii_b = np.array([0.0] + [0.0] * (M) + [sigext] * (M))
    delta_x = np.zeros(2*M + 1)
    P = len(radii_a)
    # print(radii)
    delta_x[0] = 2/np.sqrt(2)
    R = np.sqrt(np.mean(radii_a**2))
    delta_x /= R
    # delta_x /= np.sqrt(P)
    D = np.sum(radii_a**2)**2/np.sum(radii_a**4)
    print('D',D)
    U_a = np.diag(radii_a)/R
    U_b = np.diag(radii_b)/R
    m = theory_k /2
    signal = 0.5 * np.linalg.norm(delta_x)**2  # /P
    print('norm U_ax', np.linalg.norm(U_a @ delta_x))
    print('shape U_a.T U_b', (U_a.T@U_b).shape)
    noise_squared = 1/D/m + np.linalg.norm(U_a @ delta_x)**2 + np.linalg.norm(U_b @ delta_x)**2 * 1/m + np.linalg.norm(U_a.T@U_b)**2/m #  + 2*1/D/2/m**2*(1-1/m * (D/(M+1)))
    SNR = signal/np.sqrt(noise_squared)
    # SNR = compute_snr(theory_k,M+1,sigma2=sigext)

    # # from sorcher notebook
    bias = 0
    overlap = 0
    signals = (1 / R) ** 2
    SNR = SNR_func(signals,bias,M,overlap,m)

    print('R', R**2)
    print('formula', 0.5*sigext**2)
    ## Sorscher simplified
    SNR = 1 / sigext**2 / np.sqrt(1/M/m + 1/M/m**2)

    theory.append((theory_k, H(SNR)))
#
#
# ##############
#
# encoder_type = "concatnoise"
# loss = lambda pred,target: np.mean((pred - target)**2)
# signal_generator = partial(make_gaussian_signal,x_dim = 1,sig_x=1)
# ModelClass = LinearRegression
# repeats = 50
# M = 50
# regression_type = f'LinearRegression_M{M}'
# ylabel  = None
# ylim  = (0,2)
# # title = regression_type + "\n" + r"$x \sim \mathcal N(0,1); z:=[x+n,m_1,\dots,m_M]^T;\\ n\sim \mathcal N(0,0.1^2); m_i\sim \mathcal N(0,\sigma_m^2)$"
# title = None
# sigExtra = [0.1, 1, 2]
# encoders = [
#     compose(
#         partial(concatnoise_encoder,n_dim = M,sigExtra=sigExtra[j]),
#         partial(add_noise, sigExtra=0.0) #0.1
#     )
#     for j in range(len(sigExtra))
# ]
# labels=[
#     rf'$\sigma_{{\text{{ext}}}}={sigExtra[j]:.1f}$'
#     for j in range(len(sigExtra))
# ]
#
#
# theory_k = np.logspace(1,2,100)
# gamma = (M+1)/theory_k
#
# theory = [
#     (
#         theory_k,
#         (sigCoding**2 * gamma / (1-gamma)) * (gamma<1).astype(float)
#         + ( 1+1/(gamma-1) ) / ( 1+1/((gamma-1)*sigExtra[j]**2) )**2 * (gamma>1).astype(float)  # bias
#         + (sigCoding**2 / (gamma-1)) * (gamma>1).astype(float) # variance
#     ) for j in range(len(sigExtra))
# ]


######

#
# encoder_type = "HMM_examples"
# repeats = 1000
# sigCoding = 0
# title = None
# ylabel = r"Average $k$-shot Loglikelihood"
# ModelClass = BinomialMLE
# regression_type = 'BinomialMLE'
# # title = regression_type + "\n" + r"$x \sim \text{Rademacher}(0.5); z:=[x+n,m_1,\dots,m_M]^T;\\ n\sim \mathcal N(0,0.5^2); m_i\sim \mathcal N(0,\sigma_m^2)$"
# loss = lambda pred,target: (np.log(pred[target[:,0]==1]).sum() + np.log(1-pred[target[:,0]==0]).sum())/pred.shape[0]
# # loss = lambda pred, target: np.mean((pred - target)**2)
# signal_generator = lambda N: np.random.choice(2,size=(N,1))
# encoders = [flatHMMlatent,partial(chainHMMlatent,num_states=2),partial(chainHMMlatent,num_states=4)]
# labels = [r'Flat latent $\xi$',r'Chain latent 2 $\mu$',r'Chain latent 4 $\mu$']
# Bstar = 0.5
# Ltrue = Bstar * np.log(Bstar) + (1-Bstar) * np.log(1-Bstar)
# theory_k = np.logspace(1,3,100)
# var = [1/theory_k/2 * Bstar * (1-Bstar),1/theory_k * Bstar * (1-Bstar)]
#
# theory = [
#     (theory_k,Ltrue - 0.5 / theory_k),
#     (theory_k,Ltrue - 2 * 0.5 / theory_k),
#     (theory_k,Ltrue - 4 * 0.5 / theory_k),
# ]
# ylim = None


############# End of config ######################


# Initialize result storage
num_options = len(encoders)
res = np.zeros((numNs, num_options, repeats))
res_ana = np.zeros((numNs, num_options, repeats))
sig_a_all = np.zeros((numNs, num_options, 2,2))
res_ana_intermediate = np.zeros((numNs, num_options))
res_ana_ana = np.zeros((numNs, num_options))
trace_term = np.zeros((numNs, num_options, repeats))
all_a = np.zeros((numNs,num_options,repeats,2))


B = np.array([1,0])[:,None]



for j,encoder in enumerate(encoders):
    for iN,N in enumerate(N_range_list):
        for i in range(repeats):
            model = ModelClass()

            x = signal_generator(N)
            x_target = x + sigCoding * np.random.randn(N, x.shape[-1])
            z = encoder(x)

            if iN==0 and i==0:
                print(x.shape,z.shape,x_target.shape)

            # print(x.shape, z.shape, x_target.shapw)
            # print(x.shape,z.shape)
            model.fit(z,x_target)

            # testing
            x = signal_generator(N_test)
            z = encoder(x)

            x_target = x # sigCoding * np.random.randn(N_test, x.shape[-1])
            xhat = model.predict(z)  # Calculate estimated x
            res[iN, j, i] = loss(xhat,x_target)
            z[...,0] = 0
            # print('z.shape',z.shape)
            # print('norm (Xatest-x0a)', np.linalg.norm(z, axis=(-1)).mean())

        # res[iN, j, :] = few_shot_err(100,M,repeats, N_range_list[iN]*2, sigExtras[j], sigExtras[j])


            # sigma_n = np.diag(np.array([sigCoding**2,sigExtra[j]**2]))
            # res_ana[iN, j, i] = np.mean( (a[0,0]-1)**2 + (a.T @ sigma_n @ a)[0,0]) #np.trace( w.T @ w @  ) )
            # trace_term[iN, j, i] = (a.T @ sigma_n @ a)[0,0]
            # all_a[iN,j,i,:] = model.a.flatten()

        # mean_a = np.mean(all_a[iN,j],axis=0)
        # sig_a = np.cov((all_a[iN,j] - mean_a[None,:]).T)
        #
        # sig_a_all[iN,j] = sig_a
        #
        # sigma_n = np.diag(np.array([sigCoding**2,sigExtra[j]**2]))
        #
        # res_ana_intermediate[iN,j] = (
        #     (mean_a[0]-1)**2 + sig_a[0,0]
        #     + np.trace(sig_a @ sigma_n)
        #     + mean_a.T @ sigma_n @ mean_a
        #   )
        # # res_ana_intermediate[N-1,j] = (
        # #     np.trace(sig_a @ B @ B.T)
        # #     + mean_a @ B @ B.T @ mean_a
        # #     + 1 - 2 * mean_a.T @ B
        # #     + np.trace(sig_a @ sigma_n)
        # #     + mean_a.T @ sigma_n @ mean_a
        # #   )[0]
        #
        # res_ana_ana[iN,j] = (
        #     (1-mean_a[0])**2
        #     + sig_a[0,0] * (1+sigCoding**2)
        #     + sig_a[1,1] * sigExtra[j]**2
        #     + mean_a[0]**2 * sigCoding**2
        #     #+ mean_a[1]**2 * sigExtra[j]**2
        # )



# Plotting
# iN = 1
# fig,axes = plt.subplots(2, 2, figsize=(10, 8), sharex=True,sharey=True)
# for j,ax in enumerate(axes.flatten()):
#
#     ax = axes.flatten()[j]
#     ax.set_title(labels[j])
#     r = np.linspace(-10,10)
#     a = all_a[iN,j,:,:]
#     theta = np.array([np.arctan2(a_select[1],a_select[0])+np.pi/2 for a_select in a])
#     ax.plot(np.outer(r,np.cos(theta)),np.outer(r,np.sin(theta)),lw=1,alpha=0.3)
#
#     y_radius = sigExtras[j]
#     ovaltheta = np.linspace(0, 2 * np.pi, 100)
#
#     # Oval centered at (-1, 0)
#     x1 = -1 + 0.5 * np.cos(ovaltheta)
#     y1 = y_radius * np.sin(ovaltheta)
#
#     # Oval centered at (1, 0)
#     x2 = 1 + 0.5 * np.cos(ovaltheta)
#     y2 = y_radius * np.sin(ovaltheta)
#     ax.plot(x1, y1,ls='dashed',c='black')
#     ax.plot(x2, y2,ls='dashed',c='black')
#
# plt.tight_layout()
# plt.savefig(f"plots/fewshot_analytical_{encoder_type}_{regression_type}_2Dplot.png",dpi=300)
# plt.close()


plt.figure(figsize=np.array((4, 3))*0.8)

# First subplot
# Generate graded shades of blue
import matplotlib.cm as cm
colors = cm.Blues(np.linspace(0.4, 0.9, num_options))  # Adjust range as needed for desired shades

for j in range(num_options):
    mean_res = np.nanmean(res[:, j, :], axis=-1)
    # Plot with shades of blue
    plt.plot(N_range_list, mean_res, 'o', color=colors[j], label=labels[j], alpha=1, markersize=5)  # Mean over the third dimension
    # plt.errorbar(N_range_list, mean_res, yerr=np.std(res[:, j, :], axis=-1) / np.sqrt(repeats),
    #              marker='o', color=colors[j], lw=0)  # Uncomment for error bars
    if theory is not None:
        plt.plot(*theory[j], ls='dashed', color=colors[j])  # Plot theory lines with matching shade of blue
    # plt.plot(N_range_list, res_ana_ana[:, j], ls='dashed', color=colors[j])  # Uncomment for additional line if needed

plt.legend()
# plt.plot(np.mean(res_ana[1:, :, :], axis=2),ls='dashed',c='red')  # Mean over the third dimension
# plt.title('Mean of Residuals (2:end)')
plt.xlabel('k')
plt.ylabel(r'Average $k$-shot MSE')
if ylabel is not None:
    plt.ylabel(ylabel)
# plt.xlim(0,)
if ylim is not None:
    plt.ylim(*ylim)
plt.xscale('log')

if title is not None:
    plt.title(title)

# Second subplot
# plt.subplot(2, 1, 2)
# plt.plot(np.mean(res[1:10, :, :], axis=2))  # Mean for N=2:10
# plt.title('Mean of Residuals (2:10)')
# plt.xlabel('N')
# plt.ylabel('Residual Variance')
# plt.yscale('log')

plt.tight_layout()
plt.savefig(f"plots/fewshot_analytical_{encoder_type}_MSE_{regression_type}.png",dpi=300)
plt.savefig(f"plots/fewshot_analytical_{encoder_type}_MSE_{regression_type}.pdf")
plt.close()

#
# plt.figure(figsize=(5, 4))
# plt.subplot(1, 1, 1)
# for j in range(2):
#     plt.plot(N_range_list,sig_a_all[:,j,1,1],'o',c=f'C{j}',label=r'$\sigma_{ext}=$' + f'{sigExtra[j]:.1f}')
#     # plt.plot(N_range_list, 10/N_range_list**2/sigExtra[j]**(-2), c=f'C{j}', ls='dashed')
# plt.legend()
# plt.xlabel('k')
# plt.ylabel(r'$\text{Var}(a_2)$')
# # plt.xlim(0,)
# plt.ylim(0,)
# # plt.xscale('log')
#
# plt.tight_layout()
# # plt.show()
# plt.savefig("plots/fewshot_analytical_variance_a2.png",dpi=300)
# plt.close()
#
#
# plt.figure(figsize=(5, 4))
# plt.subplot(1, 1, 1)
# for j in range(2):
#     plt.plot(N_range_list,np.mean(all_a[:,j,:,0],axis=1),'o',c=f'C{j}',label=r'$\sigma_{ext}=$' + f'{sigExtra[j]:.1f}')
#     # plt.plot(N_range_list, 10/N_range_list**2/sigExtra[j]**(-2), c=f'C{j}', ls='dashed')
# plt.legend()
# plt.xlabel('k')
# plt.ylabel(r'$\langle a_1\rangle$')
# # plt.xlim(0,)
# # plt.ylim(0,)
# # plt.xscale('log')
#
# plt.tight_layout()
# # plt.show()
# plt.savefig("plots/fewshot_analytical_mean_a1.png",dpi=300)
# plt.close()




