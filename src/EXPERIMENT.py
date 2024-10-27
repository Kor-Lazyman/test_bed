import GymWrapper as gw
import time
import HyperparamTuning as ht  # Module for hyperparameter tuning
from GymWrapper import GymInterface
from config_SimPy import *
from config_RL import *
from stable_baselines3 import PPO
from log_SimPy import *
from log_RL import *
from torch.utils.tensorboard import SummaryWriter
import pandas as pd

experiment_result = {}

current_dir = os.path.dirname(__file__)
parent_dir = os.path.dirname(current_dir)
meta_model_path = os.path.join(SAVED_MODEL_PATH, 'MAML_PPO_AP3_E5_O1000')
def build_model(env):
    model = PPO("MlpPolicy", env, verbose=0, learning_rate=0.0001,batch_size=20,n_steps=SIM_TIME)
    return model

def load_model(env):
    meta_model = PPO("MlpPolicy", env, verbose=0, learning_rate=0.0001,batch_size=20,n_steps=SIM_TIME)
    meta_saved_model = PPO.load(meta_model_path)
    meta_model.policy.load_state_dict(
    meta_saved_model.policy.state_dict())
    return meta_model

def make_scenario():
    demand_scenario = []
    leadtime_scenario = []
    for mean in range(13, 16):
        demand_dict = {}
        demand_dict["Dist_Type"] = "GAUSSIAN"
        demand_dict["mean"] = mean
        for std in range(2):
            demand_dict["std"] = std
            demand_scenario.append(demand_dict)
    for mean in range(3):
        leadtime_dict = {}
        leadtime_dict["Dist_Type"] = "GAUSSIAN"
        leadtime_dict["mean"] = mean
        for std in range(2):
            leadtime_dict["std"] = std
            leadtime_scenario.append(leadtime_dict)  

    return demand_scenario, leadtime_scenario
demand_scenario, leadtime_scenario = make_scenario()

case_num = 1
for demand_scenario_dict in demand_scenario:
    for leadtime_scenario_dict in leadtime_scenario:
        rl_env = GymInterface()
        rl_env.scenario["DEMAND"] = demand_scenario_dict
        rl_env.scenario['LEADTIME'] = leadtime_scenario_dict
        rl_log_path = os.path.join(EXPERIMENT_LOGS,f'RANDOM_case_{case_num}')
        os.makedirs(rl_log_path)
        rl_env.writer = SummaryWriter(rl_log_path)
        
        rl_model = build_model(rl_env)
        rl_model.learn(total_timesteps = SIM_TIME*10)
        rl_model.save(os.path.join(SAVED_MODEL_PATH, f'RANDOM_SAVED_CASE_{case_num}'))
        rl_mean_reward, rl_std_reward = gw.evaluate_model(rl_model, rl_env, N_EVAL_EPISODES)

        meta_env = GymInterface()
        meta_env.scenario["DEMAND"] = demand_scenario_dict
        meta_env.scenario['LEADTIME'] = leadtime_scenario_dict
        meta_log_path = os.path.join(EXPERIMENT_LOGS,f'META_case_{case_num}')
        os.makedirs(meta_log_path)
        meta_env.writer = SummaryWriter(meta_log_path)
        
        meta_model = load_model(meta_env)
        meta_model.learn(total_timesteps= SIM_TIME * 10)
        meta_model.save(os.path.join(SAVED_MODEL_PATH, f'META_SAVED_CASE_{case_num}'))
        meta_mean_reward, meta_std_reward = gw.evaluate_model(meta_model, meta_env, N_EVAL_EPISODES)

        experiment_result[f'case {case_num}'] = [rl_mean_reward, meta_mean_reward]
        case_num += 1
df = pd.DataFrame(experiment_result)
df.to_csv("./Experiment_Result.csv")