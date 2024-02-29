import shutil
from datetime import datetime
from pathlib import Path

import os
import numpy as np

import ray
# from ray.tune import TuneConfig
from ray import tune
from ray.tune import CLIReporter
from ray.tune.schedulers import FIFOScheduler, ASHAScheduler
from ray.tune.search.hyperopt import HyperOptSearch
from ray.tune.suggest.basic_variant import BasicVariantGenerator

from run_single import load_config_with_overrides_and_run

# os.environ["XLA_FLAGS"] = ("--xla_cpu_multi_thread_eigen=false "
#                            "intra_op_parallelism_threads=1")


mandatory_overrides = {
    "student_subpath": "multirun/${dynamax.name}_ncomp${student.n_components}",
}

# def load_config_with_overrides_and_run(overrides={},config_path="configs/config_cohmm.yaml"):
#     print(config_path)
#     to_report =  {'original co-smoothing':np.random.uniform(1)}
#     tune.report(**to_report)

# def load_config_with_overrides_and_run(config=None, config_path="configs/config_cohmm.yaml", overrides={}):
#     # config_path = kwargs.get("config_path", "configs/config_cohmm.yaml")
#     print(config_path)

def train_func(overrides={}, config_path="configs/config_cohmm.yaml"):
    id_num = int(tune.get_trial_id()[-5:])
    overrides = {
        **overrides,
        'student_index':id_num,
        'numpy_seed.seed': id_num,
    }
    results = load_config_with_overrides_and_run(overrides=overrides,config_path=config_path)
    tune.report(**results)

train_func_partial = tune.with_parameters(
        train_func,
        config_path="configs/config_cohmm.yaml",
    )

search_space = {
        **mandatory_overrides,
        "dynamax.fit_kwargs.optimizer.learning_rate": tune.loguniform(1e-4, 1e-1).quantized(1e-4),
        "dynamax.fit_kwargs.batch_size" : tune.choice([5,10,50,100,250,500,750,1000]),
        "dynamax.fit_kwargs.num_epochs" : tune.randint(3,300),
        "student.n_components"          : tune.randint(1,17),
    }

results = tune.run(
    train_func_partial,
    num_samples=1,
    metric='original co-smoothing',
    mode='max',
    config=search_space,
    progress_reporter=CLIReporter(
        metric_columns=['original co-smoothing'],
        sort_by_metric=True,
    ),
    search_alg=BasicVariantGenerator(random_state=0),
    resources_per_trial=dict(cpu=1, gpu=0),
    # max_concurrent_trials = 1,
)



print(results.get_best_config(metric='original co-smoothing',mode='max'),results.best_result)
