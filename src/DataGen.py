import environment as env
import time
from config import *
import csv

start_time = time.time()

RM = []
for _ in range(len(I)):
    if I[_]["TYPE"] == "Raw Material":
        RM.append(I[_])

# Create or clear the CSV file and write the header
with open('simulation_data.csv', mode='w', newline='') as file:
    writer = csv.writer(file)
    writer.writerow(['Day', 'Demand', 'DailyCost'] +
                    [f'OrderQty_{r["NAME"]}' for r in RM] + [f'InvLev_{I[_]["NAME"]}' for _ in I])

# Initialize the simulation environment
daily_events = []
total_reward = 0  # 리워드 초기화
simpy_env, inventoryList, procurementList, productionList, sales, customer, providerList, daily_events = env.create_env(
    I, P, daily_events)
env.simpy_event_processes(simpy_env, inventoryList, procurementList,
                          productionList, sales, customer, providerList, daily_events, I)

# # Print the list of items and processes
# print("\nItem list")
# for i in I.keys():
#     print(f"ITEM {i}: {I[i]['NAME']}")
# print("\nProcess list")
# for i in P.keys():
#     print(f"Output of PROCESS {i}: {P[i]['OUTPUT']['NAME']}")
# print("Number of Inventories: ", len(inventoryList))
# print("Number of Providers: ", len(providerList))

total_cost = 0
previous_state = env.cap_current_state(inventoryList)

for r in RM:
    r["LOT_SIZE_ORDER"] = random.randint(ACTION_SPACE[0], ACTION_SPACE[-1])
I[0]['DEMAND_QUANTITY'] = random.randint(DEMAND_QTY_MIN, DEMAND_QTY_MAX)

for i in range(SIM_TIME*24):  # i: hourly time step
    simpy_env.run(until=i+1)  # Run the simulation until the next hour

    if (i+1) % 24 == 0:  # Daily time step
        current_state = env.cap_current_state(inventoryList)

        # Calculate the total cost of the day
        daily_total_cost = env.cal_daily_cost_ACC(
            inventoryList, procurementList, productionList, sales)

        # Print the simulation log every 24 hours (1 day)
        if PRINT_SIM_EVENTS:
            print(f"\nDay {(i+1) // 24}:")
            for log in daily_events:
                print(log)
            print("[Daily Total Cost] ", daily_total_cost)
        daily_events.clear()

        # Append data to CSV
        with open('simulation_data.csv', mode='a', newline='') as file:
            writer = csv.writer(file)
            writer.writerow([(i+1) // 24, I[0]['DEMAND_QUANTITY'], daily_total_cost] + [r["LOT_SIZE_ORDER"] for r in RM] + [
                            inven.current_level for inven in inventoryList])

        for r in RM:
            r["LOT_SIZE_ORDER"] = random.randint(
                ACTION_SPACE[0], ACTION_SPACE[-1])
        I[0]['DEMAND_QUANTITY'] = random.randint(
            DEMAND_QTY_MIN, DEMAND_QTY_MAX)


end_time = time.time()
print(f"Computation time: {(end_time - start_time):.2f} senconds")
