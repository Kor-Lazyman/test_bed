import gym
from gym import spaces
import numpy as np
from config_SimPy import *
from config_RL import *
import environment as env
from log_SimPy import *
from log_RL import *
import random
from torch.utils.tensorboard import SummaryWriter

class GymInterface(gym.Env):
    def __init__(self):
        self.shortages = 0
        self.writer = SummaryWriter(log_dir=TENSORFLOW_LOGS)
        os = []
        super(GymInterface, self).__init__()
        # Action space, observation space
        if RL_ALGORITHM == "DQN":
            # Define action space
            self.action_space = spaces.Discrete(len(ACTION_SPACE))
            # Define observation space:
            os = []
            for _ in range(len(I)):
                os.append(INVEN_LEVEL_MAX+1)
                os.append(DEMAND_QTY_MAX+1+DELTA_MIN)
                os.append(DEMAND_QTY_MAX+1)
            self.observation_space = spaces.MultiDiscrete(os)
        elif RL_ALGORITHM == "DDPG":
            # Define action space
            actionSpace = []
            for i in range(len(I)):
                if I[i]["TYPE"] == "Material":
                    actionSpace.append(len(ACTION_SPACE))
            self.action_space = spaces.MultiDiscrete(actionSpace)
         
            os=[102 for _ in range(len(I)*2+1)]
            self.observation_space = spaces.MultiDiscrete(os)
            print(os)

        elif RL_ALGORITHM == "PPO":
            # Define action space
            actionSpace = []
            for i in range(len(I)):
                if I[i]["TYPE"] == "Material":
                    actionSpace.append(len(ACTION_SPACE))
            self.action_space = spaces.MultiDiscrete(actionSpace)
         
            os=[102 for _ in range(len(I)*2+1)]
            self.observation_space = spaces.MultiDiscrete(os)
            print(os)
        self.total_reward_over_episode = []
        self.total_reward = 0
        self.num_episode = 1

        # For functions that only work when testing the model
        self.model_test=False 
        #Record the cumulative value of each cost
        self.cost_ratio={
                        'Holding cost': 0,
                        'Process cost': 0,
                        'Delivery cost': 0,
                        'Order cost': 0,
                        'Shortage cost': 0
                            }
    def reset(self):
        self.cost_ratio={
                        'Holding cost': 0,
                        'Process cost': 0,
                        'Delivery cost': 0,
                        'Order cost': 0,
                        'Shortage cost': 0
                        }
        # Initialize the simulation environment
        print("\nEpisode: ", self.num_episode)
        self.simpy_env, self.inventoryList, self.procurementList, self.productionList, self.sales, self.customer, self.providerList, self.daily_events = env.create_env(
            I, P, DAILY_EVENTS)
        env.simpy_event_processes(self.simpy_env, self.inventoryList, self.procurementList,
                                  self.productionList, self.sales, self.customer, self.providerList, self.daily_events, I)
        env.update_daily_report(self.inventoryList)
        
        #print("==========Reset==========")
        self.shortages = 0
        state_real=self.get_current_state()
        return self.correct_state_for_SB3(state_real)

    def step(self, action):
        
        # Update the action of the agent
        if RL_ALGORITHM == "DQN":
            I[1]["LOT_SIZE_ORDER"] = action


        elif RL_ALGORITHM == "DDPG":
            i = 0
            for _ in range(len(I)):
                if I[_]["TYPE"] == "Material":
                    I[_]["LOT_SIZE_ORDER"] = action[i]
                    i += 1
        elif RL_ALGORITHM == "PPO":
            i = 0
            for _ in range(len(I)):
                if I[_]["TYPE"] == "Material":
                    I[_]["LOT_SIZE_ORDER"] = action[i]
                    #I[_]["LOT_SIZE_ORDER"] = ORDER_QTY
                    i += 1

        # Capture the current state of the environment
        # current_state = env.cap_current_state(self.inventoryList)
        # Run the simulation for 24 hours (until the next day)
        #Action append
        STATE_ACTION_REPORT_REAL[-1].append(action)
        STATE_ACTION_REPORT_CORRECTION[-1].append(action)

        self.simpy_env.run(until=self.simpy_env.now + 24)
        env.update_daily_report(self.inventoryList)
        # Capture the next state of the environment
        next_state_real=self.get_current_state()
        next_state_corr = self.correct_state_for_SB3(next_state_real)
        # Calculate the total cost of the day
        env.Cost.update_cost_log(self.inventoryList)
        if PRINT_SIM:
            cost=dict(DAILY_COST_REPORT)

        if self.model_test:
            for key in DAILY_COST_REPORT.keys():
                self.cost_ratio[key]+=DAILY_COST_REPORT[key]
        
        env.Cost.clear_cost()

        reward = -COST_LOG[-1]
        self.total_reward += reward
        self.shortages += self.sales.num_shortages
        self.sales.num_shortages = 0
        
        if PRINT_SIM:
            # Print the simulation log every 24 hours (1 day)
            print(f"\nDay {(self.simpy_env.now+1) // 24}:")
            if RL_ALGORITHM == "DQN":
                print(f"[Order Quantity for {I[1]['NAME']}] ", action)
            else:
                i = 0
                for _ in range(len(I)):
                    if I[_]["TYPE"] == "Raw Material":
                        print(
                            f"[Order Quantity for {I[_]['NAME']}] ", action[i])
                        i += 1
            for log in self.daily_events:
                print(log)
            print("[Daily Total Cost] ", -reward)
            for _ in cost.keys():
                print(_,cost[_])
            print("Total cost: ", -self.total_reward)
            print("[STATE for the next round] ", next_state_real)

                
        self.daily_events.clear()

        # Check if the simulation is done
        done = self.simpy_env.now >= SIM_TIME * 24  # 예: SIM_TIME일 이후에 종료
        if done == True:
            self.writer.add_scalar(
                "reward", self.total_reward, global_step=self.num_episode)
            self.writer.add_scalar(
                "shortage", self.shortages, global_step=self.num_episode)

            print("Total reward: ", self.total_reward)
            self.total_reward_over_episode.append(self.total_reward)
            self.total_reward = 0
            self.num_episode += 1

        info = {}  # 추가 정보 (필요에 따라 사용)
        
        return next_state_corr, reward, done, info

    def get_current_state(self):
        #Make State for RL
        temp=[]
        #Update STATE_ACTION_REPORT_REAL
        for id in range(len(I)):
            #ID means Item_ID, 7 means to the length of the report for one item
            temp.append(DAILY_REPORTS[-1][(id)*7+6])#append On_Hand_inventory
            temp.append(DAILY_REPORTS[-1][(id)*7+4]-DAILY_REPORTS[-1][(id)*7+5])#append changes in inventory
        temp.append(I[0]["DEMAND_QUANTITY"]-DAILY_REPORTS[-1][6])#append remaining demand
        STATE_ACTION_REPORT_REAL.append(temp)
        return STATE_ACTION_REPORT_REAL[-1]
    
    #Min-Max Normalization    
    def correct_state_for_SB3(self,state):
        #Update STATE_ACTION_REPORT_CORRECTION.append(state_corrected)
        state_corrected=[]
        for id in range(len(I)):
            state_corrected.append(round((state[id*2]/INVEN_LEVEL_MAX)*100))#normalization Onhand inventory
            state_corrected.append(round(((state[id*2+1]-(-DELTA_MIN))/(ACTION_SPACE[-1]-(-DELTA_MIN)))*100))#normalization changes in inventory
        state_corrected.append(round(((state[-1]+INVEN_LEVEL_MAX)/(I[0]['DEMAND_QUANTITY']+INVEN_LEVEL_MAX))*100))#normalization remaining demand
        STATE_ACTION_REPORT_CORRECTION.append(state_corrected)
        return STATE_ACTION_REPORT_CORRECTION[-1]
    
    def render(self, mode='human'):
        pass

    def close(self):
        # 필요한 경우, 여기서 리소스를 정리
        pass
