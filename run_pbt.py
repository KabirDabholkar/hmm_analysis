import os
import shutil
from datetime import datetime
from pathlib import Path

from ray import tune
from ray.tune import CLIReporter
from ray.tune.search.basic_variant import BasicVariantGenerator

from lfads_torch.extensions.tune import (
    BinaryTournamentPBT,
    HyperParam,
    ImprovementRatioStopper,
)
# from lfads_torch.run_model import run_model
from run_multi import train_func_partial as run_model

mandatory_overrides = {
    "student_subpath": "multirun/${dynamax.name}_ncomp${student.n_components}",
}

HYPERPARAM_SPACE = {
    "dynamax.fit_kwargs.optimizer.learning_rate":  HyperParam(1e-4, 1e-1, explore_wt=0.8),
}


init_space = {name: tune.sample_from(hp.init) for name, hp in HYPERPARAM_SPACE.items()}

# Run the hyperparameter search
metric = "original co-smoothing"
num_trials = 1 #20
perturbation_interval = 25
burn_in_period = 80 + 25
analysis = tune.run(
    tune.with_parameters(
        run_model,
        config_path="../configs/config_cohmm.yaml",
    ),
    metric=metric,
    mode="max",
    stop=ImprovementRatioStopper(
        num_trials=num_trials,
        perturbation_interval=perturbation_interval,
        burn_in_period=burn_in_period,
        metric=metric,
        patience=4,
        min_improvement_ratio=5e-4,
    ),
    config={**mandatory_overrides, **init_space},
    resources_per_trial=dict(cpu=1, gpu=0),
    num_samples=num_trials,
    search_alg=BasicVariantGenerator(random_state=0),
    scheduler=BinaryTournamentPBT(
        perturbation_interval=perturbation_interval,
        burn_in_period=burn_in_period,
        hyperparam_mutations=HYPERPARAM_SPACE,
    ),
    keep_checkpoints_num=1,
    verbose=1,
    progress_reporter=CLIReporter(
        metric_columns=[metric, "cur_epoch"],
        sort_by_metric=True,
    ),
    trial_dirname_creator=lambda trial: str(trial),
)

run_model(
    overrides=mandatory_overrides,
    config_path="../configs/main_cohmm.yaml",
)
