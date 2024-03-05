from config import *
import environment as env

# Create environment
simpy_env, inventoryList, procurementList, productionList, sales, customer, supplierList, daily_events = env.create_env(
    I, P, DAILY_EVENTS)
env.simpy_event_processes(simpy_env, inventoryList, procurementList,
                          productionList, sales, customer, supplierList, daily_events, I)
# total_reward = 0

if PRINT_SIM_EVENTS:
    print(f"============= Initial Inventory Status =============")
    ############## PLEASE CODE HERE ##########

    print(f"============= SimPy Simulation Begins =============")

for x in range(SIM_TIME):
    simpy_env.run(until=simpy_env.now+24)
    # daily_total_cost = env.cal_daily_cost(inventoryList, procurementList, productionList, sales)
    if PRINT_SIM_EVENTS:
        # Print the simulation log every 24 hours (1 day)
        print(f"\nDay {(simpy_env.now+1) // 24} Report:")
        for log in daily_events:
            print(log)
        # print("[Daily Total Cost] ", daily_total_cost)
    daily_events.clear()
    # reward = -daily_total_cost
    # total_reward += reward
# print(total_reward)
