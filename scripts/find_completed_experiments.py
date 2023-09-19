
import os
import glob

def find_completed_files(pattern):
    # Recursively find all directories matching `pattern`
    experiment_directories = [f for f in glob.glob(os.path.join(pattern, '**'), recursive=True) if os.path.isdir(f)]
    
    for experiment_dir in experiment_directories:
        if not os.path.isdir(experiment_dir): continue
        dat_files = glob.glob(os.path.join(experiment_dir, '**', '*.dat'), recursive=True)
        if dat_files:
            for dat_file in dat_files:
                print(dat_file)
        else:
            print(f"No .dat files found beneath {experiment_dir}")

# Replace 'directory_path' with the path to the root directory where you want to start the search.
directory_path = 'data/results/Comcast_*background*'
find_completed_files(directory_path)

