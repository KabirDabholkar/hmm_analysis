from omegaconf import DictConfig, OmegaConf
import hydra
from hydra.utils import instantiate

import matplotlib as mpl
mpl.rcParams['text.usetex'] = True

CONFIG_PATH = "configs"
# CONFIG_NAME = "config"
CONFIG_NAME = "config_cohmm"

@hydra.main(version_base='1.3', config_path=CONFIG_PATH, config_name=CONFIG_NAME)
def main(cfg):
    model = instantiate(cfg.model)
    # model.transmat_ = model.transmat_()
    print(
        model.transmat_,
        model.lambdas_,
        model.startprob_
    )
    print(model.sample(10))
    return

if __name__ == '__main__':
    OmegaConf.register_new_resolver("eval", eval)
    OmegaConf.register_new_resolver("ind", lambda a, i: a[i])
    OmegaConf.register_new_resolver("listmul", lambda l, i: [l] * i)
    OmegaConf.register_new_resolver("getattr", getattr)
    OmegaConf.register_new_resolver("setattrs", lambda target, attributes, values:  [setattr(target,attr,val) for attr,val in zip(attributes,values)])

    main()
