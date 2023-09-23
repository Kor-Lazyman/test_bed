import numpy as np
import pprint
from config import *
import environment as env


# 임의로 상태 정의
state_space = []
for i in range(INVEN_LEVEL_MAX+1):
    for j in range(INVEN_LEVEL_MAX+1):
        state_space.append([i, j])
state_size = len(state_space)

# 상태 전이 확률, 보상, 감쇠인자 정의
tran_prob = 1

# 초기 정책 설정
policy = {}
for state in state_space:
    prob_per_action = 1.0 / len(ACTION_SPACE)  # 가능한 모든 행동에 대해 같은 확률 부여
    # policy[tuple(state)] = {tuple(action): prob_per_action for action in action_space}  # 확률 부여
    policy[tuple(state)] = {action: prob_per_action for action in ACTION_SPACE}
# 초기 정책 설정 확인
# print(policy[tuple([0, 0])])

# 초기 가치 함수 초기화 (예: 모든 상태에 대해 0으로 초기화)
value_function = np.zeros([INVEN_LEVEL_MAX+1, INVEN_LEVEL_MAX+1])
# 초기 가치 함수 초기화 확인
# print(value_function)

# 반복 횟수
num_iterations = 1000
stop_condition_delta = 0.1


def calculate_next_state(state, action):
    s1, s2 = state
    # (s_prime1, s_prime2) = (s1 + P[0]['PRODUCTION_RATE'] - I[0]['DEMAND_QUANTITY'], s2 - P[0]['PRODUCTION_RATE'] + action)
    # +(생산량), -(고객 주문량)
    s_prime1 = max(
        0, min(INVEN_LEVEL_MAX, s1 + P[0]['PRODUCTION_RATE'] - I[0]['DEMAND_QUANTITY']))
    # -(생산량), +(내 주문량)
    s_prime2 = max(0, min(INVEN_LEVEL_MAX, s2 -
                   P[0]['PRODUCTION_RATE'] + action))

    return (s_prime1, s_prime2)

# Policy Iteration


def policy_iteration():
    global value_function, policy
    for _ in range(num_iterations):
        new_value_function = np.zeros_like(value_function)
        new_policy = {}

        # Policy Evaluation
        while True:
            delta = 0
            for state in state_space:
                v = value_function[state[0], state[1]]
                expected_value = 0
                for action, action_prob in policy[tuple(state)].items():
                    s_prime1, s_prime2 = calculate_next_state(state, action)
                    HoldingCost = s_prime1 * \
                        I[0]['HOLD_COST'] + s_prime2 * I[1]['HOLD_COST']
                    ProductionCost = P[0]['PROCESS_COST']
                    ProcurementCost = I[1]['PURCHASE_COST'] * action
                    reward = -(HoldingCost + ProductionCost + ProcurementCost)
                    next_value = value_function[s_prime1, s_prime2]
                    expected_value += action_prob * \
                        (reward + DISCOUNT_FACTOR * tran_prob * next_value)
                value_function[state[0], state[1]] = expected_value
                delta = max(delta, abs(v - value_function[state[0], state[1]]))
            if delta < stop_condition_delta:  # Stop condition
                break

        # Policy Improvement
        policy_stable = True
        for state in state_space:
            old_action = max(policy[tuple(state)],
                             key=policy[tuple(state)].get)
            expected_values = []
            for action in ACTION_SPACE:
                s_prime1, s_prime2 = calculate_next_state(state, action)
                HoldingCost = s_prime1 * \
                    I[0]['HOLD_COST'] + s_prime2 * I[1]['HOLD_COST']
                ProductionCost = P[0]['PROCESS_COST']
                ProcurementCost = I[1]['PURCHASE_COST'] * action
                reward = -(HoldingCost + ProductionCost + ProcurementCost)
                next_value = value_function[s_prime1, s_prime2]
                expected_values.append(
                    reward + DISCOUNT_FACTOR * tran_prob * next_value)
            best_action_index = np.argmax(expected_values)
            best_action = ACTION_SPACE[best_action_index]
            for action in ACTION_SPACE:
                policy[tuple(state)
                       ][action] = 1.0 if action == best_action else 0.0
            if old_action != best_action:
                policy_stable = False

        if policy_stable:
            break

    print("Value Function : ")
    pprint.pprint(value_function)
    print("Policy : ")
    pprint.pprint(policy)


def value_iteration():
    global value_function  # Use the global value function defined outside the function

    # Repeat the value iteration for the specified number of iterations
    for _ in range(num_iterations):
        # Create a new value function with zeros (initialization)
        new_value_function = np.zeros_like(value_function)

        # Iterate over all possible states
        for state in state_space:
            max_value = -np.inf  # Set an initial max value as negative infinity

            # Policy Evaluation
            # Evaluate each possible action from the current state
            for action in ACTION_SPACE:
                action_value = 0  # Initialize the action value
                # Calculate the next states for given state and action
                s_prime1, s_prime2 = calculate_next_state(state, action)

                daily_total_cost = env.cal_daily_cost_DESC(
                    state[0], state[1], action)
                reward = -(daily_total_cost)

                # Fetch the value for the next states from the existing value function
                next_value = value_function[s_prime1, s_prime2]
                # Apply the Bellman optimality equation to update the action value
                action_value += (reward + DISCOUNT_FACTOR *
                                 tran_prob * next_value)
                # Update the maximum action value if the current action value is greater
                max_value = max(max_value, action_value)

            # Update the new value function for the current state with the computed max value
            new_value_function[state[0], state[1]] = max_value

        # Loop stop condition: check for the accuracy of estimation
        max_diff = max([abs(new_value_function[s[0], s[1]] - value_function[s[0], s[1]])
                       for s in state_space])
        if max_diff < stop_condition_delta:
            break

        # Update the global value function with the newly computed value function (rounded to 3 decimal places)
        value_function = np.round(new_value_function, 3)

        if _ % 100 == 0:  # Print the value function every 100 iterations
            print("Iteration: "+str(_))
            # Set numpy print options to display values up to 3 decimal places
            np.set_printoptions(precision=0, suppress=True, linewidth=100)
            # Print the computed value function
            print(value_function)


def extract_policy():
    global value_function
    optimal_policy = np.zeros_like(value_function)

    for state in state_space:
        max_value = -np.inf
        best_action = None

        for action in ACTION_SPACE:
            action_value = 0  # Initialize the action value
            s_prime1, s_prime2 = calculate_next_state(state, action)
            daily_total_cost = env.cal_daily_cost_DESC(
                state[0], state[1], action)
            reward = -(daily_total_cost)
            next_value = value_function[s_prime1, s_prime2]

            action_value = reward + DISCOUNT_FACTOR * tran_prob * next_value

            # If the current action value is greater than the max value, update the max value and best action
            if action_value > max_value:
                max_value = action_value
                best_action = action

        optimal_policy[state[0], state[1]] = best_action

    print(optimal_policy)


# Policy iteration
# policy_iteration()

# Value iteration

print("VALUE ITERATION")
value_iteration()
print("\nOPTIMAL POLICY")
extract_policy()
