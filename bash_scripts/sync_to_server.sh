#cd ..
base_directory='hmm_analysis'
rsync -ahP *.py kabird@132.68.139.145:~/$base_directory/
rsync -ahP hmmlearn kabird@132.68.139.145:~/$base_directory/
rsync -ahP configs kabird@132.68.139.145:~/$base_directory/
rsync -ahP plots kabird@132.68.139.145:~/$base_directory/
rsync -ahP src kabird@132.68.139.145:~/$base_directory/
rsync -ahP all_models_validated kabird@132.68.139.145:~/$base_directory/
# rsync -arhP configs --exclude configs/which_system  kabird@132.68.139.145:~/self_consistent_SAE/  #
#rsync -arhP models kabird@132.68.139.145:~/self_consistent_SAE/
#rsync -arhP dataloaders kabird@132.68.139.145:~/self_consistent_SAE/
#rsync -ahrP /Users/kabir/Documents/datasets/HFSPsuspension kabird@132.68.139.145:~/datasets/
#rsync -ahrP /Users/kabir/Documents/datasets/random_input_driven kabird@132.68.139.145:~/datasets/
#rsync -ahP *.sh kabird@132.68.139.145:~/self_consistent_SAE/
