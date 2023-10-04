rm -r checkpoints
rsync -ahP kabird@132.68.139.145:~/self_consistent_SAE/checkpoints\* ../
rsync -ahP kabird@132.68.139.145:~/self_consistent_SAE/metrics\* ../
#rsync -ahP kabird@132.68.139.145:~/keim_data/metrics\* .
#rsync -ahP kabird@132.68.139.145:~/keim_data/monitor\* .