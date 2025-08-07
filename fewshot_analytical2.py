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

class ExperimentConfig:
    """Configuration class for different experiment settings"""
    def __init__(self, name, encoder_type, sigCoding, repeats, title, ylabel, 
                 ModelClass, regression_type, loss, signal_generator, encoders, 
                 labels, theory=None, ylim=None, **kwargs):
        self.name = name
        self.encoder_type = encoder_type
        self.sigCoding = sigCoding
        self.repeats = repeats
        self.title = title
        self.ylabel = ylabel
        self.ModelClass = ModelClass
        self.regression_type = regression_type
        self.loss = loss
        self.signal_generator = signal_generator
        self.encoders = encoders
        self.labels = labels
        self.theory = theory
        self.ylim = ylim
        # Store any additional parameters
        for key, value in kwargs.items():
            setattr(self, key, value)

### select settings

# Define all three configurations
from scipy.special import erfc

def H(x):
    return 1/2*erfc(x/np.sqrt(2))

def SNR_func(signal, bias, D, overlap, m):
    return 1/2*(signal + bias/m) / np.sqrt(1/D/m + overlap*(1+1/m) + 1/D/m**2)

# Binary Classification Configuration
def create_binary_classification_config():
    sigExtras = [3, 5, 10]
    M = 20
    
    encoders = [
        compose(
            partial(add_classdependent_multivariate,
                    sigmas_a=np.array([0.0] + [sigExtra] * (M) + [0.0] * (M))/np.sqrt(M),
                    sigmas_b=np.array([0.0] + [0.0] * (M) + [sigExtra] * (M))/np.sqrt(M)
                ),
            lambda x: x @ (np.arange(2*M+1)==0)[None].astype(float)/np.sqrt(2)
        )
        for sigExtra in sigExtras
    ]
    
    labels = [rf'$\sigma_{{\text{{ext}}}}={sigExtra:.1f}$' for sigExtra in sigExtras]
    
    theory_k = np.logspace(1, 3, 100)
    theory = []
    for sigext in sigExtras:
        # Use the proper theory calculation from the original file
        radii_a = np.array([0.0] + [sigext] * (M) + [0.0] * (M))
        radii_b = np.array([0.0] + [0.0] * (M) + [sigext] * (M))
        delta_x = np.zeros(2*M + 1)
        delta_x[0] = 2/np.sqrt(2)
        R = np.sqrt(np.mean(radii_a**2))
        delta_x /= R
        D = np.sum(radii_a**2)**2/np.sum(radii_a**4)
        U_a = np.diag(radii_a)/R
        U_b = np.diag(radii_b)/R
        m = theory_k / 2
        signal = 0.5 * np.linalg.norm(delta_x)**2
        noise_squared = 1/D/m + np.linalg.norm(U_a @ delta_x)**2 + np.linalg.norm(U_b @ delta_x)**2 * 1/m + np.linalg.norm(U_a.T@U_b)**2/m
        SNR = signal/np.sqrt(noise_squared)
        
        # Also try the SNR_func approach
        bias = 0
        overlap = 0
        signals = (1 / R) ** 2
        SNR_alt = SNR_func(signals, bias, M, overlap, m)
        
        # Use the more accurate SNR calculation
        SNR = SNR_alt
        
        theory.append((theory_k, H(SNR)))
    
    return ExperimentConfig(
        name="Binary Classification",
        encoder_type="binaryclassification_21dim",
        sigCoding=0.0,
        repeats=20,
        title=None,
        ylabel=r"Average $k$-shot" + "\n" + "classification error $\epsilon$",
        ModelClass=PrototypeLearning,
        regression_type='PrototypeLearning',
        loss=lambda pred, target: np.mean(pred != target),
        signal_generator=make_binary_signal,
        encoders=encoders,
        labels=labels,
        theory=theory,
        ylim=None
    )

# Linear Regression Configuration
def create_linear_regression_config():
    M = 50
    sigExtra = [0.1, 1, 2]
    sigCoding = 0.3
    
    encoders = [
        compose(
            partial(concatnoise_encoder, n_dim=M, sigExtra=sigExtra[j]),
            partial(add_noise, sigExtra=0.1)
        )
        for j in range(len(sigExtra))
    ]
    
    labels = [rf'$\sigma_{{\text{{ext}}}}={sigExtra[j]:.1f}$' for j in range(len(sigExtra))]
    
    theory_k = np.logspace(1, 2, 100)
    gamma = (M+1)/theory_k
    
    theory = [
        (
            theory_k,
            (sigCoding**2 * gamma / (1-gamma)) * (gamma<1).astype(float)
            + ( 1+1/(gamma-1) ) / ( 1+1/((gamma-1)*sigExtra[j]**2) )**2 * (gamma>1).astype(float)
            + (sigCoding**2 / (gamma-1)) * (gamma>1).astype(float)
        ) for j in range(len(sigExtra))
    ]
    
    return ExperimentConfig(
        name="Linear Regression",
        encoder_type="concatnoise",
        sigCoding=sigCoding,
        repeats=50,
        title=None,
        ylabel=None,
        ModelClass=LinearRegression,
        regression_type=f'LinearRegression_M{M}',
        loss=lambda pred, target: np.mean((pred - target)**2),
        signal_generator=partial(make_gaussian_signal, x_dim=1, sig_x=1),
        encoders=encoders,
        labels=labels,
        theory=theory,
        ylim=(0, 2)
    )

# HMM Configuration
def create_hmm_config():
    encoders = [flatHMMlatent, partial(chainHMMlatent, num_states=2), partial(chainHMMlatent, num_states=4)]
    labels = [r'Flat latent $\xi$', r'Chain latent 2 $\mu$', r'Chain latent 4 $\mu$']
    
    Bstar = 0.5
    Ltrue = Bstar * np.log(Bstar) + (1-Bstar) * np.log(1-Bstar)
    theory_k = np.logspace(1, 3, 100)
    
    theory = [
        (theory_k, Ltrue - 0.5 / theory_k),
        (theory_k, Ltrue - 2 * 0.5 / theory_k),
        (theory_k, Ltrue - 4 * 0.5 / theory_k),
    ]
    
    return ExperimentConfig(
        name="HMM",
        encoder_type="HMM_examples",
        sigCoding=0,
        repeats=1000,
        title=None,
        ylabel=r"Average $k$-shot" + "\n" + r"Loglikelihood",
        ModelClass=BinomialMLE,
        regression_type='BinomialMLE',
        loss=lambda pred, target: (np.log(pred[target[:,0]==1]).sum() + np.log(1-pred[target[:,0]==0]).sum())/pred.shape[0],
        signal_generator=lambda N: np.random.choice(2, size=(N, 1)),
        encoders=encoders,
        labels=labels,
        theory=theory,
        ylim=None
    )

# Create all configurations
configs = [
    create_binary_classification_config(),
    create_linear_regression_config(),
    create_hmm_config()
]

# Select which configuration to use (for backward compatibility)
# Uncomment the line below to use a specific configuration
# config = configs[2]  # HMM configuration (current active one)

# For combined plot, we'll use all configurations

def run_experiment(config):
    """Run a single experiment configuration"""
    # Initialize result storage for this config
    num_options = len(config.encoders)
    res = np.zeros((numNs, num_options, config.repeats))
    
    for j, encoder in enumerate(config.encoders):
        for iN, N in enumerate(N_range_list):
            for i in range(config.repeats):
                model = config.ModelClass()

                x = config.signal_generator(N)
                x_target = x + config.sigCoding * np.random.randn(N, x.shape[-1])
                z = encoder(x)

                if iN == 0 and i == 0:
                    print(f"{config.name}: x.shape={x.shape}, z.shape={z.shape}, x_target.shape={x_target.shape}")

                model.fit(z, x_target)

                # testing
                x = config.signal_generator(N_test)
                z = encoder(x)
                x_target = x
                xhat = model.predict(z)
                res[iN, j, i] = config.loss(xhat, x_target)
    
    return res

# Create combined subplot
fig, axes = plt.subplots(1, 3, figsize=(12, 4))

# Generate colors for each configuration
import matplotlib.cm as cm
color_maps = [cm.Blues, cm.Greens, cm.Reds]

for config_idx, config in enumerate(configs):
    print(f"Running experiment: {config.name}")
    
    # Run the experiment
    res = run_experiment(config)
    
    # Plot results
    ax = axes[config_idx]
    colors = color_maps[config_idx](np.linspace(0.4, 0.9, len(config.encoders)))
    
    for j in range(len(config.encoders)):
        mean_res = np.nanmean(res[:, j, :], axis=-1)
        ax.plot(N_range_list, mean_res, 'o', color=colors[j], 
                label=config.labels[j], alpha=1, markersize=5)
        
        if config.theory is not None:
            ax.plot(*config.theory[j], ls='dashed', color=colors[j])
    
    ax.legend(framealpha=0.4, fontsize=8)
    ax.set_xlabel('k')
    if config.ylabel is not None:
        ax.set_ylabel(config.ylabel)
    if config.ylim is not None:
        ax.set_ylim(*config.ylim)
    ax.set_xscale('log')
    ax.set_title(config.name)

plt.tight_layout()
plt.savefig("plots/fewshot_analytical_combined_subplots.png", dpi=300)
plt.savefig("plots/fewshot_analytical_combined_subplots.pdf")
plt.close()

# Also run individual experiments for backward compatibility
# Uncomment the line below to run a single configuration
# config = configs[2]  # HMM configuration
# res = run_experiment(config)
# 
# # Plot individual result
# plt.figure(figsize=np.array((4, 3))*0.65)
# colors = cm.Blues(np.linspace(0.4, 0.9, len(config.encoders)))
# 
# for j in range(len(config.encoders)):
#     mean_res = np.nanmean(res[:, j, :], axis=-1)
#     plt.plot(N_range_list, mean_res, 'o', color=colors[j], 
#              label=config.labels[j], alpha=1, markersize=5)
#     if config.theory is not None:
#         plt.plot(*config.theory[j], ls='dashed', color=colors[j])
# 
# plt.legend(framealpha=0.4, fontsize=9)
# plt.xlabel('k')
# if config.ylabel is not None:
#     plt.ylabel(config.ylabel)
# if config.ylim is not None:
#     plt.ylim(*config.ylim)
# plt.xscale('log')
# if config.title is not None:
#     plt.title(config.title)
# 
# plt.tight_layout()
# plt.savefig(f"plots/fewshot_analytical_{config.encoder_type}_MSE_{config.regression_type}.png", dpi=300)
# plt.savefig(f"plots/fewshot_analytical_{config.encoder_type}_MSE_{config.regression_type}.pdf")
# plt.close()




