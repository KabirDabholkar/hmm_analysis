#./sync_from_dropbox.sh
cd ..
python3 gather_metrics.py
cd plotting
python3 plot_metrics.py