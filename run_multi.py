import shutil
from datetime import datetime
from pathlib import Path

from ray import tune
from ray.tune import CLIReporter
from ray.tune.schedulers import FIFOScheduler
from ray.tune.suggest.basic_variant import BasicVariantGenerator

from run_single import load_config_with_overrides_and_run


mandatory_overrides = {
    "student_subpath": "multirun_lr${dynamax.fit_kwargs.optimizer.learning_rate}",
}

# def load_config_with_overrides_and_run(config_path="configs/config_cohmm.yaml",overrides={}):
#     print(config_path)

# def load_config_with_overrides_and_run(config=None, config_path="configs/config_cohmm.yaml", overrides={}):
#     # config_path = kwargs.get("config_path", "configs/config_cohmm.yaml")
#     print(config_path)

tune.run(
    tune.with_parameters(
        load_config_with_overrides_and_run,
        config_path="configs/config_cohmm.yaml",
    ),
    # partial(load_config_with_overrides_and_run,config_path="configs/config_cohmm.yaml"),
    num_samples=1,
    config={
        **mandatory_overrides,
        # "dynamax.fit_kwargs.optimizer.learning_rate": tune.loguniform(1e-4, 1e-1),
    },
    resources_per_trial=dict(cpu=1, gpu=0),
)
