import time
from GymWrapper import *
from GymEnvironment import *
from config_SimPy import *
from config_MARL import *
import multiprocessing
import pandas as pd
# Simulate Code for multi_processing

'''
simulate: 멀티 프로세싱에서 작동될 함수
    *변수*
    env: startmap은 generator를 인식 할 수 없기에 simulate에서 선언
    manager: multi_processing에서 데이터 공유를 할 때 사용하는 데이터 type, 각 프로세스 별 replaybuffer가 저장되어 있음음
    model: model을 manager에 저장할 경우 부모 프로세스에서 tensor를 찾을 수 없기에 부모 프로세스에서 입력

    *기능*
    시뮬레이션 진행 후 Process N의 replaybuffer 업데이트 
'''
def simulate(manager, epsilon, process_id, model):
    temp_buffer = manager[f'process{process_id}'] #manager에서 replaybuffer 추출
    env = InventoryManagementEnv()
    state = env.reset()
    done = False
    action_distribution = [[] for _ in range(NUM_AGENTS)]
    rewards = []
    while not done:
        actions = []
        # Select actions for each agent (optional: with probs)
        for i in range(NUM_AGENTS):
            action = model.select_action(state, i, epsilon)
            actions.append(action)
            action_distribution[i].append(action)

        next_state, reward, done, info = env.step(actions)
        
        # Store in replay buffer
        temp_buffer.push((state, actions, reward, next_state, done))
        env.total_reward += reward
        state = next_state
        rewards.append(reward)

    manager[f'process{process_id}'] = temp_buffer #manager 업데이트
    return env.total_reward


'''
multiprocessing.Manager(): multi_process에서 process간의 공유 가능한 데이터 저장 형식, dict와 list로 선언 가능
Process_args: 멀티 프로세싱을 할 때, 함수에 들어가는 입력값들
multiprocessing.Pool(processes): processes에서 지정한 만큼의 프로세스에 코어를 배정
starmap: pool에 들어갈 함수와 인자를 설정 및 실행
'''
if __name__ == "__main__":
    test_times = []
    for num_processes in range(1, NUM_PROCESS+1):
        print('='*20,f"Experiment processes-{num_processes}",'='*20)
        temp = []
        for run in range(1):
            # Start timing the computation
            start_time = time.time()

            # Create environment
            env = InventoryManagementEnv()
            # Initialize wrapper
            wrapper = GymWrapper(
                env=env,
                num_agents=MAT_COUNT,
                joint_action_space_size=JOINT_ACTION_SPACE_SIZE,
                multi_state_space_size=MULTI_STATE_SPACE_SIZE,
                buffer_size=BUFFER_SIZE,
                batch_size=BATCH_SIZE,
                lr_actor=LEARNING_RATE_ACTOR,
                lr_critic=LEARNING_RATE_CRITIC,
                gamma=GAMMA,
                tau=TAU,
                num_heads=NUM_HEADS,
                hidden_dim=HIDDEN_DIM
            )
            if LOAD_MODEL:
                # Load the saved model and evaluate
                print(f"Loading model from {MODEL_PATH}")
                try:
                    wrapper.load_model(MODEL_PATH)
                    print("Model loaded successfully")

                    # Evaluate the loaded model
                    training_end_time = time.time()
                    wrapper.evaluate(N_EVAL_EPISODES)
                except FileNotFoundError:
                    print(f"No saved model found at {MODEL_PATH}")
                    exit()
            else:
                manager = multiprocessing.Manager().dict() #process 데이터 공유 설정정
                for x in range(1, num_processes+1):
                    manager[f'process{x}'] = ReplayBuffer(
                    buffer_size=BUFFER_SIZE)

                # Train new model
                print("Starting training of new model...")
                for episode in range(N_TRAIN_EPISODES//num_processes + 1):
                    # Indicators for logging
                    critic_loss_val = 0.0
                    actor_losses_val = [0.0]*NUM_AGENTS
                    td_error_val = 0.0
                    mean_q_val = 0.0
                    std_q_val = 0.0
                    policy_entropy_val = 0.0
                    param_norms_val = (0.0, [0.0]*NUM_AGENTS)

                    if EPSILON_DECAY_TYPE == 'linear':
                        fraction = min(float(episode) / float(N_TRAIN_EPISODES), 1.0)
                        epsilon = EPSILON_START + fraction * \
                            (EPSILON_END - EPSILON_START)
                    elif EPSILON_DECAY_TYPE == 'exponential':
                        epsilon = EPSILON_END + \
                            (EPSILON_START - EPSILON_END) * (DECAY_RATE ** episode)
                        
                    process_args = [(manager, epsilon, i, wrapper.maac) for i in range(1,num_processes+1)]
                    with multiprocessing.Pool(processes = num_processes) as pool:
                            result = pool.starmap(simulate, process_args)  # 병렬 실행

                    if len(manager['process1'].buffer)>=BATCH_SIZE:
                        for process_id in range(1, num_processes+1):
                            wrapper.maac.update(BATCH_SIZE, manager[f'process{x}'])
                    wrapper.logger.log_training_info(
                        episode=episode,
                        episode_reward=wrapper.env.total_reward,
                        critic_loss=critic_loss_val,
                        actor_losses=actor_losses_val,
                        epsilon=epsilon,
                        q_values=(mean_q_val, std_q_val),
                        policy_entropy=policy_entropy_val,
                        kl_divergence=None,
                        td_error=td_error_val,
                        episode_length=episode,
                        param_norms=param_norms_val
                    )
                                # Print info every eval_interval
                    if episode % EVAL_INTERVAL == 0:
                        print(
                            f"Episode {episode} | Epsilon {epsilon} | Average Total Reward {sum(result)/num_processes:.3f}")
                        print("-"*50)
                training_end_time = time.time()

                # Evaluate the trained model
                print("\nStarting evaluation...")
                wrapper.evaluate(N_EVAL_EPISODES)

            # Calculate computation time and print it
            end_time = time.time()
            computation_time = (end_time - start_time) / 60
            temp.append(computation_time)
            print("\nTime Analysis:")
            print(f"Total computation time: {(end_time - start_time)/60:.2f} minutes")
            if not LOAD_MODEL:
                print(f"Training time: {(training_end_time - start_time)/60:.2f} minutes")
            print(f"Evaluation time: {(end_time - training_end_time)/60:.2f} minutes")

        test_times.append(temp)
            # tensorboard --logdir=runs
    df = pd.DataFrame(test_times)
    df.to_csv('test_times.csv')