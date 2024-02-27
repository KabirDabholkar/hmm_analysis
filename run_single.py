import os
import shutil
from datetime import datetime
from pathlib import Path
import hydra
from omegaconf import OmegaConf
from lfads_torch.utils import flatten

from main_cohmm import main

# ---------- OPTIONS -----------
PROJECT_STR = "lfads-torch-example"
DATASET_STR = "nlb_mc_maze"
RUN_TAG = datetime.now().strftime("%y%m%d") + "_exampleSingle"
RUN_DIR = Path("/snel/share/runs") / PROJECT_STR / DATASET_STR / RUN_TAG
OVERWRITE = True
# ------------------------------

# Overwrite the directory if necessary
# if RUN_DIR.exists() and OVERWRITE:
#     shutil.rmtree(RUN_DIR)
# RUN_DIR.mkdir(parents=True)
# Copy this script into the run directory
# shutil.copyfile(__file__, RUN_DIR / Path(__file__).name)
# Switch to the `RUN_DIR` and train the model
# os.chdir(RUN_DIR)
def load_config_with_overrides_and_run(
        overrides={},
        config_path="configs/config_cohmm.yaml",
    ):
    config_path = Path(config_path)
    overrides = [f"{k}={v}" for k, v in flatten(overrides).items()]
    print('overrides',overrides)
    print()
    with hydra.initialize(
        config_path=config_path.parent,
        job_name="run_model",
        version_base="1.1",
    ):
        print('config path', config_path.name)
        config = hydra.compose(config_name=config_path.name, overrides=overrides)
    main(config)
    # print(OmegaConf.to_yaml(config))

if __name__ == '__main__':
    load_config_with_overrides_and_run(
        # overrides={
        #     "datamodule": DATASET_STR,
        #     "model": DATASET_STR,
        # },
        # overrides={},
        config_path="configs/config_cohmm.yaml",
    )
