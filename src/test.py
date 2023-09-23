import random
#### Items #####################################################################
# ID: Index of the element in the dictionary
# TYPE: Product, Raw Material, WIP;
# NAME: Item's name or model;
# CUST_ORDER_CYCLE: Customer ordering cycle [days]
# MANU_ORDER_CYCLE: Manufacturer ordering cycle to providers [days]
# DEMAND_QUANTITY: Demand quantity for the final product [units]
# DELIVERY_TIME_TO_CUST: Delivery time to the customer [days]
# DELIVERY_TIME_FROM_SUP: Delivery time from a supplier [days]
# SUP_LEAD_TIME: The total processing time for a supplier to process and deliver the manufacturer's order [days]
# REMOVE## LOT_SIZE_ORDER: Lot-size for the order of raw materials (Q) [units] -> THIS IS AN AGENT ACTION THAT IS UPDATED EVERY 24 HOURS
# HOLD_COST: Holding cost of the items [$/unit*day]
# PURCHASE_COST: Holding cost of the raw materials [$/unit]
# SETUP_COST_PRO: Setup cost for the delivery of the products to the customer [$/delivery]
# SETUP_COST_RAW: Setup cost for the ordering of the raw materials to a supplier [$/order]
# DELIVERY_COST: Delivery cost of the products [$/unit]
# DUE_DATE: Term of customer order to delivered [days]
# BACKORDER_COST: Backorder cost of products or WIP [$/unit]

#### Processes #####################################################################
# ID: Index of the element in the dictionary
# PRODUCTION_RATE [units/day]
# INPUT_TYPE_LIST: List of types of input materials or WIPs
# QNTY_FOR_INPUT_ITEM: Quantity of input materials or WIPs [units]
# OUTPUT: Output WIP or Product
# PROCESS_COST: Processing cost of the process [$/day]
# PROCESS_STOP_COST: Penalty cost for stopping the process [$/unit]


# Scenario #1: a single process
# Scenario 1-1: deterministic
I = {0: {"ID": 0, "TYPE": "Product",      "NAME": "PRODUCT",
         "CUST_ORDER_CYCLE": 1,
         "DEMAND_QUANTITY": 1,
         "HOLD_COST": 0,
         "SETUP_COST_PRO": 5,
         "DELIVERY_COST": 5,
         "DUE_DATE": 0,
         "BACKORDER_COST": 30},
     1: {"ID": 1, "TYPE": "Raw Material", "NAME": "RAW MATERIAL 1",
         "MANU_ORDER_CYCLE": 1,
         "SUP_LEAD_TIME": 1,
         "HOLD_COST": 0,
         "PURCHASE_COST": 0,
         "SETUP_COST_RAW": 1}}
# Scenario 1-2: stochastic
# I = {0: {"ID": 0, "TYPE": "Product",      "NAME": "PRODUCT",
#          "CUST_ORDER_CYCLE": 1,
#          "DEMAND_QUANTITY": random.randint(0, 5),
#          "HOLD_COST": 5,
#          "SETUP_COST_PRO": 5,
#          "DELIVERY_COST": 5,
#          "DUE_DATE": 0,
#          "BACKORDER_COST": 20},
#      1: {"ID": 1, "TYPE": "Raw Material", "NAME": "RAW MATERIAL 1",
#          "MANU_ORDER_CYCLE": 1,
#          "SUP_LEAD_TIME": random.randint(1, 5),
#          "HOLD_COST": 1,
#          "PURCHASE_COST": 2,
#          "SETUP_COST_RAW": 20}}

P = {0: {"ID": 0, "PRODUCTION_RATE": 1, "INPUT_TYPE_LIST": [I[1]], "QNTY_FOR_INPUT_ITEM": [
    1], "OUTPUT": I[0], "PROCESS_COST": 0, "PROCESS_STOP_COST": 2}}

'''
# Scenario 2
I = {0: {"ID": 0, "TYPE": "Product",      "NAME": "PRODUCT",          "INIT_LEVEL": 30, "CUST_ORDER_CYCLE": 7, "DEMAND_QUANTITY": 21,                                           "HOLD_COST": 5, "SHORTAGE_COST": 10,                     "SETUP_COST_PRO": 50, "DELIVERY_COST": 10, "DUE_DATE": 2, "BACKORDER_COST": 5},
     1: {"ID": 1, "TYPE": "Raw Material", "NAME": "RAW MATERIAL 1.1", "INIT_LEVEL": 30, "MANU_ORDER_CYCLE": 7,                        "SUP_LEAD_TIME": 7, "LOT_SIZE_ORDER": 21, "HOLD_COST": 1, "SHORTAGE_COST": 2, "PURCHASE_COST": 3,  "SETUP_COST_RAW": 20},
     2: {"ID": 2, "TYPE": "Raw Material", "NAME": "RAW MATERIAL 2.1", "INIT_LEVEL": 30, "MANU_ORDER_CYCLE": 7,                        "SUP_LEAD_TIME": 7, "LOT_SIZE_ORDER": 21, "HOLD_COST": 1, "SHORTAGE_COST": 2, "PURCHASE_COST": 3,  "SETUP_COST_RAW": 20},
     3: {"ID": 3, "TYPE": "Raw Material", "NAME": "RAW MATERIAL 2.2", "INIT_LEVEL": 30, "MANU_ORDER_CYCLE": 7,                        "SUP_LEAD_TIME": 7, "LOT_SIZE_ORDER": 21, "HOLD_COST": 1, "SHORTAGE_COST": 2, "PURCHASE_COST": 3,  "SETUP_COST_RAW": 20},
     4: {"ID": 4, "TYPE": "WIP",          "NAME": "WIP 1",            "INIT_LEVEL": 30,                                                                                         "HOLD_COST": 2, "SHORTAGE_COST": 2}}

P = {0: {"ID": 0, "PRODUCTION_RATE": 3, "INPUT_LIST": [I[1]]            , "INPUT_USE_COUNT": [1]    , "OUTPUT": I[4], "PROCESS_COST": 5, "PRO_STOP_COST": 2},
     1: {"ID": 1, "PRODUCTION_RATE": 2, "INPUT_LIST": [I[2], I[3], I[4]], "INPUT_USE_COUNT": [1,1,1], "OUTPUT": I[0], "PROCESS_COST": 6, "PRO_STOP_COST": 3}}
'''
# RL algorithms
RL_ALGORITHM = "DQN"  # "DP", "DQN", "DDPG", "PPO", "SAC"
ACTION_SPACE = [0, 1, 2]

# State space
# if this is not 0, the length of state space of demand quantity is not identical to INVEN_LEVEL_MAX
INVEN_LEVEL_MIN = 0
INVEN_LEVEL_MAX = 10  # Capacity limit of the inventory [units]
# DEMAND_QTY_MIN = 0  # if this is not 0, the length of state space of demand quantity is not identical to DEMAND_QTY_MAX
# DEMAND_QTY_MAX = 2

# Print logs
PRINT_SIM_EVENTS = False
PRINT_DQN = True

COST_VALID = False
VISUAL = False
DAILY_EVENTS = []
# Cost model
# If False, the total cost is calculated based on the inventory level for every 24 hours.
# Otherwise, the total cost is accumulated every hour.
HOURLY_COST_MODEL = True

# Simulation
EPISODES = 1000
SIM_TIME = 100  # [days] per episode
INIT_LEVEL = 5  # Initial inventory level [units]

batch_size = 32
# hyper parameter DQN
DISCOUNT_FACTOR = 0.98  # gamma
epsilon_greedy = 1.0
epsilon_min = 0.01
epsilon_decay = 0.99995
learning_rate = 0.001
max_memory_size = 2000
target_update_frequency = 1
