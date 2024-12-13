import time
from GymWrapper import *
from GymEnvironment import *
from config_SimPy import *
from config_MARL import *

# Start timing the computation
start_time = time.time()

# Create environment
env = InventoryManagementEnv()

# Initialize wrapper
wrapper = GymWrapper(
    env=env,
    n_agents=MAT_COUNT,
    action_dim=len(ACTION_SPACE),  # 0-5 units order quantity
    buffer_size=BUFFER_SIZE,
    batch_size=BATCH_SIZE,
    lr=LEARNING_RATE,
    gamma=GAMMA
)

# Train MAAC
wrapper.train(N_TRAIN_EPISODES, EVAL_INTERVAL)
# trained_maac = wrapper.train(N_TRAIN_EPISODES, EVAL_INTERVAL)
training_end_time = time.time()
'''
# model = build_model()
# # Train the model
# model.learn(total_timesteps=SIM_TIME * N_EPISODES)
# model.save(os.path.join(SAVED_MODEL_PATH, SAVED_MODEL_NAME))
# print(f"{SAVED_MODEL_NAME} is saved successfully")
'''

# Evaluate
wrapper.evaluate(N_EVAL_EPISODES)
'''
# Evaluate the trained model
mean_reward, std_reward = gw.evaluate_model(model, env, N_EVAL_EPISODES)
print(
    f"Mean reward over {N_EVAL_EPISODES} episodes: {mean_reward:.2f} +/- {std_reward:.2f}")
'''

# Calculate computation time and print it
end_time = time.time()
print(f"Computation time: {(end_time - start_time)/60:.2f} minutes \n",
      f"Training time: {(training_end_time - start_time)/60:.2f} minutes \n ",
      f"Test time:{(end_time - training_end_time)/60:.2f} minutes")
