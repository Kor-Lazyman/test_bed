import os
from config_SimPy import *

# Action space constants
ACTION_MIN = 0
ACTION_MAX = 5  # ACTION_SPACE = [0, 1, 2, 3, 4, 5]

# State dimension constants
'''
On-hand inventory level for each item: len(I)
In-transition inventory level for each material: MAT_COUNT
Remaining demand: 1 MAA
'''
STATE_DIM = len(I) + MAT_COUNT + 1


BUFFER_SIZE = 3000
BATCH_SIZE = 64  # Batch size for training (unit: episodes)
LEARNING_RATE = 0.01
GAMMA = 0.95
N_STEPS = 100
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
N_TRAIN_EPISODES = 10000
EVAL_INTERVAL = 10

# Evaluation
'''
N_EVAL_EPISODES: Number of evaluation episodes (Default=100) 
'''
N_EVAL_EPISODES = 10

# Configuration for model loading/saving
LOAD_MODEL = False  # Set to True to load a saved model
MODEL_DIR = "saved_models"
if not os.path.exists(MODEL_DIR):
    os.makedirs(MODEL_DIR)
MODEL_PATH = os.path.join(MODEL_DIR, "maac_best_model.pt")  # 불러올 모델 경로

# Debuging
UPDATE_COUNT = 1
UPDATE_COUNT_FORMER = 0