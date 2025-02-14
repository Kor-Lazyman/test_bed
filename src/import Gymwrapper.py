import GymWrapper as gw 
from GymEnvironment import *
import optuna
import optuna.visualization as vis

def tuning_hyperparam(trial):
    # Create environment
    env = InventoryManagementEnv()
    env.reset()
    learning_rate = trial.suggest_loguniform('learning_rate', 1e-5, 1)
    gamma = trial.suggest_float('gamma', 0.9, 0.9999, log=True)
    batch_size = trial.suggest_categorical(
        'batch_size', [16, 32, 64, 128, 256])
    buffer_size = trial.suggest_integral("Buffer_size",2,10000,log=True)
    
    # Initialize wrapper
    model = gw.GymWrapper(
    env=env,
    n_agents=MAT_COUNT,
    action_dim=ACTION_MAX,  # 0-5 units order quantity
    state_dim=STATE_DIM,
    buffer_size=buffer_size,
    batch_size=batch_size,
    lr=learning_rate,
    gamma=gamma
    )

    model.train(N_TRAIN_EPISODES, EVAL_INTERVAL)
    avg_reward= model.evaluate(N_EVAL_EPISODES)

    return -avg_reward  # Minimize the negative of mean reward

def run_optuna():
    # study = optuna.create_study( )
    study = optuna.create_study(direction='minimize')
    study.optimize(tuning_hyperparam, n_trials=50)

    # Print the result
    best_params = study.best_params
    print("Best hyperparameters:", best_params)
    # Visualize hyperparameter optimization process
    vis.plot_optimization_history(study).show()
    vis.plot_parallel_coordinate(study).show()
    vis.plot_slice(study).show()
    vis.plot_contour(study, params=['learning_rate', 'gamma']).show()
