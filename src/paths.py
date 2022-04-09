# Get paths to the data and results directories.
import os 
src_path = os.path.dirname(os.path.realpath(__file__))
prj_path = os.path.dirname(src_path)
data_dir = os.path.join(prj_path, 'data')
results_dir = os.path.join(prj_path, 'results')
