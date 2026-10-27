import os
import sys
import subprocess
import argparse
import pandas as pd

# Usage:
#   python scripts/run_tsfm_on_eva.py moment s 512
#   python scripts/run_tsfm_on_eva.py chronos l 1024
#   python scripts/run_tsfm_on_eva.py time-moe s 512
#   python scripts/run_tsfm_on_eva.py times_fm l 1024

# Read all script files in the 'tsfms' directory into a list.
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
script_files = [f for f in os.listdir(os.path.join(project_root, "tsfms")) if f.endswith(".py") and f != "helper.py"]

parser = argparse.ArgumentParser(
    description='Run a time series foundation model on the evaluation dataset.',
    formatter_class=argparse.RawDescriptionHelpFormatter,
    epilog="""
script_name : script name under tsfms/
model_size : 's' or 'l'
window_size: positive integer
--s / --save-reconstruction: whether to save reconstruction results (optional flag only for MOMENT)
""")
parser.add_argument('script_name', type=str, choices=script_files, help="Name of the script to run")
parser.add_argument('model_size', type=str, choices=['s', 'l'], help="Model size: 's' or 'l'")
parser.add_argument('window_size', type=int, help='Positive integer for window size')
parser.add_argument("-s" ,"--save-reconstruction", action='store_true', help="Whether to save reconstruction results (only for MOMENT)")
args = parser.parse_args()

if args.window_size <= 0:
    parser.error("window_size must be a positive integer.")

script_name = args.script_name
model_size = args.model_size
window_size = str(args.window_size)

project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
file_list_path = os.path.join(project_root, "data", "File_List", "TSB-AD-U-Eva.csv")
data_dir = os.path.join(project_root, "data", "TSB-AD-U")
model_script = os.path.join(project_root, "tsfms", script_name)

if not os.path.exists(model_script):
    print(f"Error: Model script not found at {model_script}")
    sys.exit(1)

files_df = pd.read_csv(file_list_path)
files = files_df["file_name"].dropna().tolist()

for i, file_name in enumerate(files, 1):
    full_path = os.path.join(data_dir, file_name)
    if not os.path.exists(full_path):
        print(f"[{i}/{len(files)}] missing: {file_name}")
        continue

    print(f"[{i}/{len(files)}] processing: {file_name} with {script_name}")
    cmd = [sys.executable, model_script, model_size, window_size, file_name]
    if args.save_reconstruction and script_name == 'moment_zs.py':
        cmd.append("--save-reconstruction")
    subprocess.run(cmd, check=False)