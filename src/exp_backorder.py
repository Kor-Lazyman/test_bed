import GymWrapper as gw
import time
import HyperparamTuning as ht  # Module for hyperparameter tuning
from GymWrapper import GymInterface
from config_SimPy import *
from config_RL import *
from stable_baselines3 import DQN, DDPG, PPO
from log_SimPy import *
from log_RL import *
from torch.utils.tensorboard import SummaryWriter
import pandas as pd

# Function to build the model based on the specified reinforcement learning algorithm


def build_model(env):
    if RL_ALGORITHM == "DQN":
        model = DQN("MlpPolicy", env, verbose=0)
        # model = DQN("MlpPolicy", env, learning_rate=BEST_PARAMS['learning_rate'], gamma=BEST_PARAMS['gamma'],
        #             batch_size=BEST_PARAMS['batch_size'], verbose=0)
    elif RL_ALGORITHM == "DDPG":
        model = DQN("MlpPolicy", env, verbose=0)
        # model = DDPG("MlpPolicy", env, learning_rate=BEST_PARAMS['learning_rate'], gamma=BEST_PARAMS['gamma'],
        #              batch_size=BEST_PARAMS['batch_size'], verbose=0)
    elif RL_ALGORITHM == "PPO":
        model = PPO("MlpPolicy", env, verbose=0, learning_rate=0.0001,batch_size=20,n_steps=SIM_TIME)
        # model = PPO("MlpPolicy", env, learning_rate=BEST_PARAMS['learning_rate'], gamma=BEST_PARAMS['gamma'],
        #             batch_size=BEST_PARAMS['batch_size'], n_steps=SIM_TIME, verbose=0)
        print(env.observation_space)
    return model


'''
def export_report(inventoryList):
    for x in range(len(inventoryList)):
        for report in DAILY_REPORTS:
            export_Daily_Report.append(report[x])
    daily_reports = pd.DataFrame(export_Daily_Report)
    daily_reports.columns = ["Day", "Name", "Type",
                         "Start", "Income", "Outcome", "End"]
    daily_reports.to_csv("./Daily_Report.csv")
'''


# Start timing the computation
start_time = time.time()



demand_scenario = []
leadtime_scenario = []

for demand_mean in range(10,15):
    temp = {}
    temp["Dist_Type"] = "UNIFORM"
    temp['min'] = demand_mean - 2
    temp['max'] = demand_mean + 2
    demand_scenario.append(temp)

for leadtime_mean in range(2,4):
    temp = {}
    temp["Dist_Type"] = 'UNIFORM'
    temp['min'] = leadtime_mean - 1
    temp['max'] = leadtime_mean + 1
    leadtime_scenario.append(temp)

scenarios = []
pair = []
report = {}
for demand in demand_scenario:
    for leadtime in leadtime_scenario:
        # Define the scenario
        scenarios.append([demand, leadtime])

reward_report = {}
case_num = 0
for demand, leadtime in scenarios:    
    case_num += 1
    env = GymInterface()
    model = build_model(env)
    
    env.writer = SummaryWriter(log_dir = os.path.join(TENSORFLOW_LOGS, f"Case{case_num}"))
    reward_report[f"DEMAND{demand}, LEADTIME{leadtime}"] = []
    env.scenario = {"DEMAND" : demand, "LEADTIME": leadtime}
    
    model.learn(total_timesteps = SIM_TIME * N_EPISODES)
    mean_reward, std_reward = gw.evaluate_model(model, env, N_EVAL_EPISODES)
    reward_report[f"DEMAND{demand}, LEADTIME{leadtime}"].append(mean_reward)
df = pd.DataFrame(reward_report)
df.to_csv("./exp_backorder.csv")