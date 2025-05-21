import os
from pathlib import Path
# Define the colors to be replaced
colors = [
    # {'color': '#ffb000', 'replace': '#fe6100'},
    {'color': '#ffb000', 'replace': '#ffb000'}, # training
    {'color': '#009e73', 'replace': '#dc267f'}, # evaluation and Q_s
    # {'color': '#3097ff', 'replace': '#648fff'}
    {'color': '#3097ff', 'replace': '#785ef0'} # decoding distances

]


def modify_svgs(file_paths,modifier="_modified"):
    new_files = []
    for f in file_paths:
        print(f)
        with open(f, 'r') as fh:
            data = fh.read()

        # Replace the colors
        for c in colors:
            data = data.replace(c['color'], c['replace'])

        # Define the new file name with '_modified' suffix
        base, ext = os.path.splitext(f)
        new_file = f"{base}{modifier}{ext}"

        # Write the modified content to the new file
        with open(new_file, 'w') as fh:
            fh.write(data)
        new_files.append(new_file)
    return new_files


base_path = Path("/Users/kabir/Documents/reports/predict_explain_paper/consolidated_figures")
base_path2 = Path("/Users/kabir/Documents/reports/predict_explain_paper/ICLR_illustrations")


# Example usage: pass a list of file paths to modify
file_paths = [
    base_path / "fig1modified2.svg",
    base_path / "hmm_fewshot_modified.svg",
    base_path2 / "fig3.drawio.svg",
    base_path / "lfads_and_STNDT_kshot_and_cros_colsums_modified.svg",
    base_path / "cross_decoding_modified.svg",
    base_path / "hmm_examples_fig_modified.svg"
    # Add more file paths as needed
]
file_paths = [str(f) for f in file_paths]

new_file_paths = modify_svgs(file_paths,modifier="_newcolors")

for new_f in new_file_paths:
    base,_ = os.path.splitext(new_f)
    os.system(f"cat {base}.svg | inkscape --pipe --export-filename={base}.pdf")
