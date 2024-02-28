import shutil
from datetime import datetime
from pathlib import Path

import os


import ray
# from ray.tune import TuneConfig
from ray import tune
from ray.tune import CLIReporter
from ray.tune.schedulers import FIFOScheduler
from ray.tune.suggest.basic_variant import BasicVariantGenerator

from run_single import load_config_with_overrides_and_run

os.environ["XLA_FLAGS"] = ("--xla_cpu_multi_thread_eigen=false "
                           "intra_op_parallelism_threads=1")

ray.init()
print(
    'available_resources',ray.available_resources()
)

mandatory_overrides = {
    "student_subpath": "multirun/${dynamax.name}",
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
    num_samples=2,
    config={
        **mandatory_overrides,
        "dynamax.fit_kwargs.optimizer.learning_rate": tune.loguniform(1e-4, 1e-1).quantized(1e-4),
        "dynamax.fit_kwargs.batch_size": tune.choice([10,50,100,300,500,1000]),
        "dynamax.fit_kwargs.num_epochs": tune.randint(3,300),
    },
    # resources_per_trial=dict(cpu=1, gpu=0),
    # num_cpus_per_trial=1,
    max_concurrent_trials = 1,
)
