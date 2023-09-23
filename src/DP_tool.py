import numpy as np
import mdptoolbox
from config import *
import environment as env
np.set_printoptions(threshold=np.inf)


# 상태 공간 및 행동 공간 정의
state_space = [(i, j) for i in range(INVEN_LEVEL_MAX+1)
               for j in range(INVEN_LEVEL_MAX+1)]
state_size = len(state_space)

# 전이 확률 및 보상 행렬 초기화
P_matrix = np.zeros((len(ACTION_SPACE), state_size, state_size))
R_matrix = np.zeros((len(ACTION_SPACE), state_size, state_size))


def calculate_next_state(state, action):
    s1, s2 = state
    if s2 > 0:  # if production is possible
        s_prime1 = max(0, min(INVEN_LEVEL_MAX, s1 +
                       P[0]['PRODUCTION_RATE'] - I[0]['DEMAND_QUANTITY']))
        s_prime2 = max(0, min(INVEN_LEVEL_MAX, s2 -
                       P[0]['PRODUCTION_RATE'] + action))
    else:  # if production is impossible
        s_prime1 = max(0, min(INVEN_LEVEL_MAX, s1 - I[0]['DEMAND_QUANTITY']))
        s_prime2 = max(0, min(INVEN_LEVEL_MAX, action))
    return (s_prime1, s_prime2)


# 전이 확률 및 보상 행렬 채우기
for state_idx, state in enumerate(state_space):
    for action_idx, action in enumerate(ACTION_SPACE):
        next_state = calculate_next_state(state, action)
        next_state_idx = state_space.index(next_state)

        # 전이 확률
        P_matrix[action_idx][state_idx][next_state_idx] = 1

        # 보상
        daily_total_cost = env.cal_daily_cost_DESC(state[0], state[1], action)
        R_matrix[action_idx][state_idx][next_state_idx] = -daily_total_cost

# Value Iteration 수행
vi = mdptoolbox.mdp.ValueIteration(P_matrix, R_matrix, DISCOUNT_FACTOR)
vi.run()

optimal_value_function_2D = np.array(vi.V).reshape(
    (INVEN_LEVEL_MAX+1, INVEN_LEVEL_MAX+1)).astype(int)
optimal_policy_action = [ACTION_SPACE[action_idx] for action_idx in vi.policy]
optimal_policy_2D = np.array(optimal_policy_action).reshape(
    (INVEN_LEVEL_MAX+1, INVEN_LEVEL_MAX+1))

print("Optimal Value Function:")
print(optimal_value_function_2D)
print("\nOptimal Policy:")
print(optimal_policy_2D)
print("\nP_matrix:")
print(P_matrix)
print("\nR_matrix:")
print(R_matrix)
