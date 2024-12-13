import os
from config_SimPy import *

ACTION_SPACE = [0, 1, 2, 3, 4, 5]

BUFFER_SIZE = 100000,
BATCH_SIZE = 64,
LEARNING_RATE = 0.01,
GAMMA = 0.95

# Find minimum Delta
PRODUCT_OUTGOING_CORRECTION = 0
for key in P:
    PRODUCT_OUTGOING_CORRECTION = max(P[key]["PRODUCTION_RATE"] *
                                      max(P[key]['QNTY_FOR_INPUT_ITEM']), INVEN_LEVEL_MAX)
# maximum production

# Training
'''
N_TRAIN_EPISODES: Number of training episodes (Default=1000)
EVAL_INTERVAL: Interval for evaluation and printing results (Default=10)
'''
N_TRAIN_EPISODES = 1000
EVAL_INTERVAL = 10

# Evaluation
'''
N_EVAL_EPISODES: Number of evaluation episodes (Default=100) 
'''
N_EVAL_EPISODES = 10

# Saved Model
SAVE_MODEL = False
SAVED_MODEL_PATH = os.path.join(parent_dir, "Saved_Model")
SAVED_MODEL_NAME = "PPO_MODEL_test_val"

# Load Model
LOAD_MODEL = False
LOAD_MODEL_NAME = "PPO_MODEL_SIM500"
