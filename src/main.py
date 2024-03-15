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
    ############## PLEASE CODE HERE ##########\
    for inventory in inventoryList:
        print(f"Day 0 - {I[inventory.item_id]['NAME']} Inventory: {inventory.on_hand_inventory} units")

    print(f"============= SimPy Simulation Begins =============")

for x in range(SIM_TIME):
    for inventory in inventoryList:
        inventory.reset_daily_records()

    simpy_env.run(until=simpy_env.now + 24)  # SimPy 환경을 24시간(하루) 동안 실행

    # 여기에 최종 인벤토리 상태를 업데이트하는 코드를 추가
    for inventory in inventoryList:
        inventory.daily_final = inventory.on_hand_inventory  # 하루가 끝날 때 최종 상태 업데이트

    if PRINT_LOG_DAILY_REPORT:  # 일일 인벤토리 보고서 출력
        print(f"\nDay {(simpy_env.now + 1) // 24} Inventory Report:")
        print("NAME : START_INVEN / INCOMING / OUTGOING / END_INVEN")
        for inventory in inventoryList:
            name = I[inventory.item_id]['NAME']
            start_inven = inventory.daily_final - inventory.daily_in + inventory.daily_out  # 시작 재고 계산
            incoming = inventory.daily_in
            outgoing = inventory.daily_out
            end_inven = inventory.daily_final
            print(f"{name} : {start_inven} / {incoming} / {outgoing} / {end_inven}")

    if PRINT_SIM_EVENTS:
        print(f"\nDay {(simpy_env.now + 1) // 24} - Report")
        for inventory in inventoryList:
            print(f"Day {(simpy_env.now + 1) // 24} - {I[inventory.item_id]['NAME']} Inventory: In: {inventory.daily_in}, Out: {inventory.daily_out}, Final: {inventory.daily_final} units")
        for log in daily_events:
            print(log)
        daily_events.clear()
    # reward = -daily_total_cost
    # total_reward += reward
# print(total_reward)
