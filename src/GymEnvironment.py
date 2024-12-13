
import gym
from gym import spaces
import numpy as np
from config_SimPy import *
from config_MARL import *
from environment import *
from MAAC import *
from log_SimPy import *
from log_MARL import *


class InventoryManagementEnv(gym.Env):
    """
    Gym environment for multi-agent inventory management system
    Handles the simulation of inventory management with multiple procurement agents
    """

    def __init__(self):
        super(InventoryManagementEnv, self).__init__()
        self.scenario = {"DEMAND": DEMAND_SCENARIO,
                         "LEADTIME": LEADTIME_SCENARIO}
        self.shortages = 0
        self.total_reward_over_episode = []
        self.total_reward = 0
        self.cur_episode = 1  # Current episode
        self.cur_outer_loop = 1  # Current outer loop
        self.cur_inner_loop = 1  # Current inner loop
        self.scenario_batch_size = 99999  # Initialize the scenario batch size
        self.current_day = 0  # Initialize the current day
        # Record the cumulative value of each cost
        for key in DAILY_COST.keys():
            LOG_TOTAL_COST_COMP[key] = 0

        # Define action space
        """
        Action space is a MultiDiscrete space where each agent can choose an order quantity
        """
        actionSpace = []
        for i in range(len(I)):
            if I[i]["TYPE"] == "Material":
                actionSpace.append(len(ACTION_SPACE))
        self.action_space = spaces.MultiDiscrete(actionSpace)

        # Define observation space for both local and global states
        """
        Observation space is a Dict containing local and global observations
        Local observation: On-hand inventory level for each material agent, in-transit inventory level for each material agent
        Global observation: WIP inventory levels, product inventory level, demand-inventory difference
        """
        # 각 material agent의 local observation
        local_obs_dims = []
        for _ in range(MAT_COUNT):
            local_obs_dims.extend([
                # on_hand inventory (0 to INVEN_LEVEL_MAX)
                INVEN_LEVEL_MAX + 1,
                # in_transit inventory (need larger range due to potential accumulation)
                INVEN_LEVEL_MAX * 2 + 1
            ])

        # Global observation (WIP levels, product inventory, demand-inventory difference)
        global_obs_dims = []
        # WIP inventories
        wip_count = sum(1 for id in I.keys() if I[id]["TYPE"] == "WIP")
        global_obs_dims.extend([INVEN_LEVEL_MAX + 1] *
                               wip_count)  # WIP inventory levels
        global_obs_dims.append(INVEN_LEVEL_MAX + 1)  # Product inventory level
        # Demand-inventory difference (can be negative?)
        global_obs_dims.append(INVEN_LEVEL_MAX * 2 + 1)

        self.observation_space = spaces.Dict({
            'local_obs': spaces.MultiDiscrete(local_obs_dims),
            'global_obs': spaces.MultiDiscrete(global_obs_dims)
        })

        # Initialize simulation environment
        self.reset()

    def reset(self):
        """
        Reset the environment to initial state

        Returns:
            observations: Initial state of the environment
        """
        # Initialize the total reward for the episode
        for key in DAILY_COST.keys():
            LOG_TOTAL_COST_COMP[key] = 0

        # Create new SimPy environment and components
        self.simpy_env, self.inventory_list, self.procurement_list, self.production_list, self.sales, self.customer, self.supplier_list, self.daily_events = create_env(
            I, P, LOG_DAILY_EVENTS)

        # Initialize simulation processes
        scenario = {
            "DEMAND": DEMAND_SCENARIO,
            "LEADTIME": LEADTIME_SCENARIO
        }
        simpy_event_processes(
            self.simpy_env, self.inventory_list, self.procurement_list,
            self.production_list, self.sales, self.customer, self.supplier_list,
            self.daily_events, I, scenario
        )
        update_daily_report(self.inventory_list)

        self.current_day = 0

        return self._get_observations()

    def step(self, actions):
        """
        Execute one time step (1 day) in the environment

        Args:
            actions: Array of order quantities for each material agent

        Returns:
            observations: Dict containing local and global observations
            reward: Negative total cost for the day
            done: Whether the episode has ended
            info: Additional information for debugging
        """
        # Set order quantities for each material agent
        for i, action in enumerate(actions):
            I[self.procurement_list[i].item_id]["LOT_SIZE_ORDER"] = int(action)

        # Run simulation for one day
        STATE_ACTION_REPORT_REAL[-1].append(actions)
        self.simpy_env.run(until=(self.current_day + 1) * 24)
        self.current_day += 1
        update_daily_report(self.inventory_list)

        # Get next observations
        observations = self._get_observations()

        # Calculate reward (a negative value of the daily total cost)
        reward = -Cost.update_cost_log(self.inventory_list)
        # Update LOG_TOTAL_COST_COMP
        for key in DAILY_COST.keys():
            LOG_TOTAL_COST_COMP[key] += DAILY_COST[key]
        Cost.clear_cost()
        self.total_reward += reward
        self.shortages += self.sales.num_shortages
        self.sales.num_shortages = 0

        # Check if episode is done
        done = self.current_day >= SIM_TIME

        # Additional info for debugging
        info = {
            'Day': self.current_day,
            'Daily cost': -reward,
            'Total cost': -self.total_reward,
            'inventory_levels': {f"Material_{i}": inv.on_hand_inventory
                                 for i, inv in enumerate(self.inventory_list)
                                 if I[inv.item_id]['TYPE'] == 'Material'}
        }

        return observations, reward, done, info

    def _get_observations(self):
        """
        Construct observation dictionary containing local and global states

        Returns:
            Dict containing local and global observations
        """
        # Get local observations for each material agent
        local_obs = np.zeros((MAT_COUNT, 2), dtype=np.float32)
        for i, proc in enumerate(self.procurement_list):
            local_obs[i] = [
                self.inventory_list[proc.item_id].on_hand_inventory,
                self.inventory_list[proc.item_id].in_transition_inventory
            ]

        # Get global observations (WIP levels, product inventory, demand-inventory)
        wip_levels = [inv.on_hand_inventory for inv in self.inventory_list
                      if I[inv.item_id]['TYPE'] == 'WIP']
        product_inventory = self.inventory_list[0].on_hand_inventory
        demand_gap = I[0]['DEMAND_QUANTITY'] - product_inventory

        global_obs = np.array(wip_levels + [product_inventory, demand_gap],
                              dtype=np.float32)

        return {
            'local_obs': local_obs,
            'global_obs': global_obs
        }

    def render(self, mode='human'):
        """
        Render the environment's current state
        Currently just prints basic information
        """
        if mode == 'human':
            print(f"\nDay: {self.current_day}")
            print("\nInventory Levels:")
            for inv in self.inventory_list:
                print(f"{I[inv.item_id]['NAME']}: {inv.on_hand_inventory} "
                      f"(In Transit: {inv.in_transition_inventory})")

    def close(self):
        """Clean up environment resources"""
        pass
