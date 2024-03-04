```sh
conda create -n hmm_analysis python=3.9
conda activate hmm_analysis
pip install -r requirements.txt
pip install -e hmmlearn
pip install -e config-utils
```

Modify `configs/path_to_save_files.yaml`.