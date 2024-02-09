from config import *
import environment as env


# Create environment
simpy_env, inventoryList, procurementList, productionList, sales, customer, providerList, daily_events = env.create_env(
    I, P, DAILY_EVENTS)
env.simpy_event_processes(simpy_env, inventoryList, procurementList,
                          productionList, sales, customer, providerList, daily_events, I)
total_reward = 0
for x in range(SIM_TIME):
    simpy_env.run(until=simpy_env.now+24)
    daily_total_cost = env.cal_daily_cost_ACC(
        inventoryList, procurementList, productionList, sales)

    if PRINT_SIM_EVENTS:

        # Print the simulation log every 24 hours (1 day)
        print(f"\nDay {(simpy_env.now+1) // 24}:")
        print("===============Order Phase===============")
        if (x) % I[0]["CUST_ORDER_CYCLE"] == 0 or x == 0:
            I[0]['DEMAND_QUANTITY'] += random.randint(
                DEMAND_QTY_MIN, DEMAND_QTY_MAX)
            ORDER_HISTORY.append(I[0]['DEMAND_QUANTITY'])
            print(
                f"{simpy_env.now-24}: Customer order of {I[0]['NAME']}                                : {I[0]['DEMAND_QUANTITY']} units ")

        for y in range(1, len(I)):

            if (x) % I[y]["MANU_ORDER_CYCLE"] == 0 or x == 0:
                I[y]["LOT_SIZE_ORDER"] = random.randint(0, 5)
                print(
                    f"{simpy_env.now-24}: {I[y]['NAME']} is ordered by procurement                 : {I[y]['LOT_SIZE_ORDER']}  units  ")
        for log in daily_events:
            print(log)
        print("[Daily Total Cost] ", daily_total_cost)

    daily_events.clear()
    reward = -daily_total_cost
    total_reward += reward
print(total_reward)
