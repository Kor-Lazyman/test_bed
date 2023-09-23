from InventoryMgtEnv import GymInterface
from config import *
import numpy as np
import optuna
import optuna.visualization as vis
from stable_baselines3.common.evaluation import evaluate_policy
import time
if RL_ALGORITHM == "DQN":
    from stable_baselines3 import DQN
elif RL_ALGORITHM == "DDPG":
    from stable_baselines3 import DDPG
elif RL_ALGORITHM == "PPO":
    from stable_baselines3 import PPO

# Create environment
env = GymInterface()


def evaluate_model(model, env, num_episodes):
    all_rewards = []
    for _ in range(num_episodes):
        obs = env.reset()
        episode_reward = 0
        done = False
        while not done:
            action, _ = model.predict(obs)
            obs, reward, done, _ = env.step(action)
            episode_reward += reward
        all_rewards.append(episode_reward)
    mean_reward = np.mean(all_rewards)
    std_reward = np.std(all_rewards)
    return mean_reward, std_reward


def tuning_hyperparam(trial):
    # Initialize the environment
    env.reset()
    # Define search space for hyperparameters
    learning_rate = trial.suggest_loguniform('learning_rate', 1e-5, 1)
    gamma = trial.suggest_float('gamma', 0.9, 0.9999, log=True)
    batch_size = trial.suggest_categorical(
        'batch_size', [16, 32, 64, 128, 256])
    # Define the RL model
    if RL_ALGORITHM == "DQN":
        model = DQN("MlpPolicy", env, learning_rate=learning_rate,
                    gamma=gamma, batch_size=batch_size, verbose=0)
    elif RL_ALGORITHM == "DDPG":
        model = DDPG("MlpPolicy", env, learning_rate=learning_rate,
                     gamma=gamma, batch_size=batch_size, verbose=0)
    elif RL_ALGORITHM == "PPO":
        model = PPO("MlpPolicy", env, learning_rate=learning_rate,
                    gamma=gamma, batch_size=batch_size, verbose=0)
    # Train the model
    model.learn(total_timesteps=SIM_TIME*N_EPISODES)
    # Evaluate the model
    eval_env = GymInterface()
    mean_reward, _ = evaluate_policy(
        model, eval_env, n_eval_episodes=N_EVAL_EPISODES)

    return -mean_reward  # Minimize the negative of mean reward


start_time = time.time()

if OPTIMIZE_HYPERPARAMETERS:
    study = optuna.create_study()
    study.optimize(tuning_hyperparam, n_trials=N_TRIALS)

    # Print the result
    best_params = study.best_params
    print("Best hyperparameters:", study.best_params)
    # Visualize hyperparameter optimization process
    vis.plot_optimization_history(study).show()
    vis.plot_parallel_coordinate(study).show()
    vis.plot_slice(study).show()
    vis.plot_contour(study, params=['learning_rate', 'gamma']).show()

else:
    if RL_ALGORITHM == "DQN":
        # model = DQN("MlpPolicy", env, verbose=0)
        model = DQN("MlpPolicy", env, learning_rate=BEST_PARAMS['learning_rate'], gamma=BEST_PARAMS['gamma'],
                    batch_size=BEST_PARAMS['batch_size'], verbose=0, tensorboard_log=TENSORFLOW_LOGS)
    elif RL_ALGORITHM == "DDPG":
        model = DDPG("MlpPolicy", env, learning_rate=BEST_PARAMS['learning_rate'], gamma=BEST_PARAMS['gamma'],
                     batch_size=BEST_PARAMS['batch_size'], verbose=0, tensorboard_log=TENSORFLOW_LOGS)
    elif RL_ALGORITHM == "PPO":
        model = PPO("MlpPolicy", env, learning_rate=BEST_PARAMS['learning_rate'], gamma=BEST_PARAMS['gamma'],
                    batch_size=BEST_PARAMS['batch_size'], verbose=0, tensorboard_log=TENSORFLOW_LOGS)

    model.learn(total_timesteps=SIM_TIME*N_EPISODES)  # Time steps = days
    env.render()

    # 학습 후 모델 평가
    mean_reward, std_reward = evaluate_model(model, env, N_EVAL_EPISODES)
    print(
        f"Mean reward over {N_EVAL_EPISODES} episodes: {mean_reward:.2f} +/- {std_reward:.2f}")

    # Optimal policy
    if RL_ALGORITHM == "DQN":
        optimal_actions_matrix = np.zeros(
            (INVEN_LEVEL_MAX + 1, INVEN_LEVEL_MAX + 1), dtype=int)
        for i in range(INVEN_LEVEL_MAX + 1):
            for j in range(INVEN_LEVEL_MAX + 1):
                if STATE_DEMAND:
                    state = np.array([i, j, I[0]['DEMAND_QUANTITY']])
                    action, _ = model.predict(state)
                    optimal_actions_matrix[i, j] = action
                else:
                    state = np.array([i, j])
                    action, _ = model.predict(state)
                    optimal_actions_matrix[i, j] = action

        # Print the optimal actions matrix
        print("Optimal Actions Matrix:")
        # print("Demand quantity: ", I[0]['DEMAND_QUANTITY'])
        print(optimal_actions_matrix)

end_time = time.time()
print(f"Computation time: {(end_time - start_time)/3600:.2f} hours")

'''
#모델 저장 및 로드 (선택적)
model.save("dqn_inventory")
loaded_model = DQN.load("dqn_inventory")
'''

# TensorBoard 실행:
# tensorboard --logdir="C:/tensorboard_logs/"
# http://localhost:6006/
