from config_SimPy import *
from log import *
import environment as env
import pandas as pd
import Visualization
# Create environment
simpy_env, inventoryList, procurementList, productionList, sales, customer, supplierList, daily_events, daily_reports = env.create_env(
    I, P, DAILY_EVENTS, DAILY_REPORTS)
env.simpy_event_processes(simpy_env, inventoryList, procurementList,
                          productionList, sales, customer, supplierList, daily_events, daily_reports, I)
# total_reward = 0


if PRINT_SIM_EVENTS:
    print(f"============= Initial Inventory Status =============")
    for inventory in inventoryList:
        print(
            f"Day 1 - {I[inventory.item_id]['NAME']} Inventory: {inventory.on_hand_inventory} units")

    print(f"============= SimPy Simulation Begins =============")

for x in range(SIM_TIME):
    daily_events.append(f"\nDay {(simpy_env.now) // 24+1} Report:")
    simpy_env.run(until=simpy_env.now+24)
    # daily_total_cost = env.cal_daily_cost(inventoryList, procurementList, productionList, sales)
    if PRINT_SIM_EVENTS:
        # Print the simulation log every 24 hours (1 day)
        for log in daily_events:
            print(log)
        # print("[Daily Total Cost] ", daily_total_cost)
    daily_events.clear()

    env.update_daily_report(inventoryList)
    if PRINT_SIM_REPORT:
        for id in range(len(inventoryList)):
            print(DAILY_REPORTS[x][id])

    env.Cost.update_cost_log(simpy_env, inventoryList)
    if PRINT_SIM_COST:
        for key in DAILY_COST_REPORT.keys():
            print(f"{key}: {DAILY_COST_REPORT[key]}")
        print(sum(COST_LOG))
    env.Cost.clear_cost()
    # reward = -daily_total_cost
    # total_reward += reward

export_Daily_Report = []
for x in range(len(inventoryList)):
    for report in DAILY_REPORTS:
        export_Daily_Report.append(report[x])
if VISUALIAZTION != False:
    Visualization.visualization(export_Daily_Report)
daily_reports = pd.DataFrame(export_Daily_Report)
daily_reports.columns = ["Day", "Name", "Type",
                         "Start", "Income", "Outcome", "End"]
daily_reports.to_csv("./Daily_Report.csv")

# print(total_reward)
