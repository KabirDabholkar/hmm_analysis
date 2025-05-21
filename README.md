Clone the repository.
```sh
git clone --recurse-submodules https://github.com/KabirDabholkar/hmm_analysis.git
```

```sh
conda create -n hmm_analysis python=3.9
conda activate hmm_analysis
pip install -r requirements.txt
pip install -e hmmlearn
pip install -e config-utils
```

Modify `path_to_save_files` in `configs/path_to_save_files.yaml`. This is where all results and model data will be stored.

As a starting point run, run `python main_cohmm.py`.
This script is loading a config file `configs/config_simply.yaml`.
By default, it trains and evaluates a single student model in the student-teacher setting. 
Uncommenting the following line 
```yaml
  - hydra: multirun_cohmm
```
in `configs/config_simply.yaml` and running `python main_cohmm.py` will train many models of various sizes.
Results are stored in a directory 'all_models_validated_v2', by default.
