base_dir = "./"

train_files_dir = f"{base_dir}/train_files"
supplemental_files_dir = f"{base_dir}/supplemental_files"

HORIZON = 1
WINDOW_SIZE = 14
N_EPOCHS = 5000
N_NEURONS = 128
N_STACKS = 10
N_LAYERS = 4
INPUT_SIZE = WINDOW_SIZE * HORIZON
THETA_SIZE = INPUT_SIZE + HORIZON
