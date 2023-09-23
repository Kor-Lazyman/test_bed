import environment as env
import numpy as np
import random
from visualization import *
from config import *
from DQN import *


def print_policy(agent):
    # 임의로 상태 정의
    state_space = []
    for i in range(CAPA_LIMIT+1):
        for j in range(CAPA_LIMIT+1):
            state_space.append([i, j])
    optimal_policy = np.zeros([CAPA_LIMIT+1, CAPA_LIMIT+1])
    for state in state_space:
        optimal_policy[state[0], state[1]] = np.argmax(
            agent.get_policy(state))

    print(optimal_policy)


def main():
    # Initialize the simulation environment
    daily_events = []
    total_reward = 0  # 리워드 초기화
    simpy_env, inventoryList, procurementList, productionList, sales, customer, providerList, daily_events = env.create_env(
        I, P, daily_events)

    # Initialize the DQN agent
    state_size = len(inventoryList)  # Number of inventories
    agent = DQNAgent(state_size, ACTION_SPACE, DISCOUNT_FACTOR,
                     epsilon_greedy, epsilon_min, epsilon_decay,
                     learning_rate, max_memory_size, target_update_frequency)
    episode_done = False
    hist_total_reward, hist_loss = [], []

    env.simpy_event_processes(simpy_env, inventoryList, procurementList,
                              productionList, sales, customer, providerList, daily_events, I)

    # Print the list of items and processes
    print("\nItem list")
    for i in I.keys():
        print(f"ITEM {i}: {I[i]['NAME']}")
    print("\nProcess list")
    for i in P.keys():
        print(f"Output of PROCESS {i}: {P[i]['OUTPUT']['NAME']}")
    print("Number of Inventories: ", len(inventoryList))
    print("Number of Providers: ", len(providerList))

    for episode in range(EPISODES):
        total_cost = 0
        losses = []
        previous_state = env.cap_current_state(inventoryList)

        for i in range(SIM_TIME*24):  # i: hourly time step
            simpy_env.run(until=i+1)  # Run the simulation until the next hour

            if (i+1) % 24 == 0:  # Daily time step
                current_state = env.cap_current_state(inventoryList)
                agent_action = agent.choose_action(current_state)
                I[1]["LOT_SIZE_ORDER"] = agent_action

                # Calculate the total cost of the day
                if HOURLY_COST_MODEL:
                    daily_total_cost = env.cal_daily_cost_ACC(
                        inventoryList, procurementList, productionList, sales)
                else:
                    s = []
                    for _ in range(len(inventoryList)):
                        s.append(inventoryList[_].current_level)
                    daily_total_cost = env.cal_daily_cost_DESC(
                        s[0], s[1], agent_action)
                total_cost += daily_total_cost

                # Print the simulation log every 24 hours (1 day)
                if PRINT_SIM_EVENTS:
                    print(f"\nDay {(i+1) // 24}:")
                    for log in daily_events:
                        print(log)
                    print("[Daily Total Cost] ", daily_total_cost)
                daily_events.clear()

                # DOES IT NESSESSARY TO HAVE episode_done ?
                reward = -daily_total_cost
                agent.remember(Transition(
                    previous_state, agent_action, reward, current_state, episode_done))
                current_state = previous_state

                if len(agent.memory) == agent.max_memory_size:
                    loss = agent.replay(batch_size)
                    losses.append(loss)

                total_reward += reward

        if isinstance(total_reward, torch.Tensor):
            total_reward = int(total_reward)
        if isinstance(total_cost, torch.Tensor):
            total_cost = int(total_cost)

        print("\n[Total Cost] ", total_cost)
        print(f'Episode: {episode}/{EPISODES}, Total Reward: {total_reward}, Eps: {agent.epsilon:.2f}, Loss: {np.mean(losses):.5f}, Memory: {len(agent.memory)}')
        hist_total_reward.append(total_reward)
        hist_loss.append(np.mean(losses))
        if episode > 20:
            if episode % 10 == 0:
                print_policy(agent)

        # Initialize the simulation environment
        total_reward = 0
        simpy_env, inventoryList, procurementList, productionList, sales, customer, providerList, daily_events = env.create_env(
            I, P, daily_events)

    print("Optimal action for the state:")
    print_policy(agent)

    print(hist_total_reward)
    print(hist_loss)
    visualization.plot_hist_and_loss(hist_total_reward, hist_loss)

    '''
    # Visualize the data trackers of the inventory level and cost over time
    for i in I.keys():
        inventory_visualization = visualization.visualization(
            inventoryList[i], I[i]['NAME'])
        inventory_visualization.inventory_level_graph()
        inventory_visualization.inventory_cost_graph()
        # calculate_inventory_cost()
    '''
    '''
    if SPECIFIC_HOLDING_COST:
        print(EventHoldingCost)
    
    #visualization
    if VISUAL :
        cost_list=[]#inventory_cost by id   id -> day 순으로 리스트 생성  전체 id 별로 저장되어 있는 list
        level_list=[]#inventory_level by id
        item_name_list=[]
        total_cost_per_day = env.cal_cost(inventoryList, productionList, procurementList,sales)
        total_cost_list = total_cost_per_day
        for i in I.keys():
            temp1=[]
            temp2=[]
            inventory_visualization = visualization.visualization(
                inventoryList[i])
            temp1,temp2=inventory_visualization.return_list()
            level_list.append(temp1)
            cost_list.append(temp2)
            item_name_list.append(I[i]['NAME'])
        inventory_visualization = visualization.visualization(None) # 필요하지 않으므로 None
        inventory_visualization.plot_inventory_graphs(level_list, cost_list,total_cost_list,item_name_list)
    '''


if __name__ == "__main__":
    main()
