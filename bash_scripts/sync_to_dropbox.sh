#!/bin/bash
PROJECT_HOME="/Users/kabir/Documents/code/nlb-lightning/"
DROPBOX_HOME="/Users/kabir/Dropbox/nlb-lightning/"

rsync -ahP "$PROJECT_HOME"*.py "$DROPBOX_HOME"
rsync -ahP "$PROJECT_HOME"*.txt "$DROPBOX_HOME"
rsync -arhP "$PROJECT_HOME"scripts_hydra "$DROPBOX_HOME"
rsync -arhP "$PROJECT_HOME"nlb_lightning "$DROPBOX_HOME"
rsync -arhP "$PROJECT_HOME"simulate_data --exclude simulate_data/configs/which_system   "$DROPBOX_HOME"
rsync -arhP "$PROJECT_HOME"configs --exclude configs/which_system "$DROPBOX_HOME"  #
rsync -arhP "$PROJECT_HOME"assets       "$DROPBOX_HOME"
rsync -arhP "$PROJECT_HOME"analysis     "$DROPBOX_HOME"
rsync -arhP "$PROJECT_HOME"nlb_tools    "$DROPBOX_HOME"


#rsync -ahrP /Users/kabir/Documents/datasets/HFSPsuspension kabird@132.68.139.145:~/datasets/
#rsync -ahrP /Users/kabir/Documents/datasets/random_input_driven kabird@132.68.139.145:~/datasets/

#rsync -ahP *.sh ~/Dropbox/self_consistent_SAE/

#rm -r ~/Dropbox/remote/commands/*
#rsync "$PROJECT_HOME"bash_scripts/run_train.sh ~/Dropbox/remote/commands/