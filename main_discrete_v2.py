import gym
import numpy as np
import time
import sys
import argparse
from math import pi
import matplotlib.pyplot as plt
import random

def main():
    parser = get_paras()
    args = parser.parse_args()
    env = gym.make('RobotInterception-v1')
    total_reward = []
    state = env.reset()
    position_size = env.img_height + 1
    error_size = env.img_width + 1
    state_size = position_size * error_size
    action_size = 3
    Q1 = np.zeros([position_size, action_size])
    Q2 = np.zeros([error_size, action_size])
    # Q_idx = np.zeros([position_size, error_size], dtype=int)
    #
    # i = 0
    # for p in range(position_size):
    #     for e in range(error_size):
    #         Q_idx[p, e] = i
    #         i += 1

    # action_dict = {
    #     0: np.array([0, 0]),
    #     1: np.array([0, 1]),
    #     2: np.array([0, 2]),
    #     3: np.array([1, 0]),
    #     4: np.array([1, 1]),
    #     5: np.array([1, 2]),
    #     6: np.array([2, 0]),
    #     7: np.array([2, 1]),
    #     8: np.array([2, 2])
    # }

    for episode in range(args.max_episode+1):
        done = False
        state = env.reset()
        reward = 0
        reward_sum = 0
        reward_sum_pos = 0
        reward_sum_err = 0
        i = 1
        init_state = state
        action_sum = 0
        info = False
        while not done and i <= args.max_iterations:

            position = state[3]
            error = state[2]
            # print(yaw)
            action = get_action(env, Q1, Q2, state, args.epsilon)
            if abs(error) >= 5:
                action[0] = 1
            # action[0] = 0
            # print(type(action_idx))
            # action = action_dict[action_idx]
            next_state, reward, done, info = env.step(action)
            next_position = next_state[3]
            next_error = next_state[2]
            # try:
            #     next_state_idx = Q_idx[next_position, next_error+160]
            # except:
            #     print(f"Step {i}:")
            #     print(f"Current position of robot: {env.robot}")
            #     print(f"Current position of target: {env.target}")
            #     print(f"Current action: {action}")
            #     print(f"Current position reward: {reward_pos}; rotation reward: {reward_error}; total reward: {reward}")
            #     # print(f"total reward is {reward_sum}, step reward is {reward}, action is {action}, error is {error}, "
            #     print(f"State is {state}, next State is {next_state}.\n")
            if abs(next_error) < abs(error):
                reward_error = 1/i  #(abs(error)-abs(next_error))/i
                if abs(error) <= 3:
                    reward_error = 10
            else:
                reward_error = -1
            if position < next_position:
                reward_pos = 1/i
                if position > 0.9 * env.img_height:
                    reward_pos = 10
            else:
                reward_pos = -1

            reward = reward_pos + reward_error
            # print(error, next_error)
            Q1[position, action[0]] += args.alpha * (reward_pos + args.gamma * np.max(Q1[next_position]) - Q1[position, action[0]])

            Q2[error+160, action[1]] += args.alpha * (reward_error + args.gamma * np.max(Q2[next_error+160]) - Q2[error+160, action[1]])
            if args.debug_print and episode >=  args.max_episode:
                print(f"Step {i}:")
                print(f"Current position of robot: {env.robot}")
                print(f"Current position of target: {env.target}")
                print(f"Current action: {action}")
                print(f"Current position reward: {reward_pos}; rotation reward: {reward_error}; total reward: {reward}")
                # print(f"total reward is {reward_sum}, step reward is {reward}, action is {action}, error is {error}, "
                print(f"State is {state}, next State is {next_state}.\n")

            reward_sum += reward
            reward_sum_pos += reward_pos
            reward_sum_err += reward_error

            # action_sum += action
            if args.visualization:
                env.render()
                time.sleep(0.01)

            state = next_state
            i += 1

            # print(f"total reward is {reward_sum}, step reward is {reward}, action is {action}, done is {done}.")
        reward_avg = [reward_sum_pos, reward_sum_err]
        total_reward.append(reward_sum)


        if episode % 50 == 0:
            # print()
            print(f"Episode: {episode}:")
            print(f"Sum reward: {reward_avg}; Final state: {state}; Initial state: {init_state};")
            print(f"Final step reward: {[reward_pos, reward_error]}; Times of iters: {i}")

def get_action(env, Q1, Q2, state, epsilon):
    if random.random() <= epsilon:
        action = env.action_space.sample()
    else:
        action1 = np.argmax(Q1[state[3]])
        action2 = np.argmax(Q2[state[2]+160])
        action = np.array([action1, action2])
    return action

def get_paras():
    parser = argparse.ArgumentParser()
    parser.add_argument('--max_episode', dest='max_episode', default=1000, type=int,
                        help='the maximum number of episodes')
    parser.add_argument('--max_iterations', dest='max_iterations', default=1000, type=int,
                        help='the maximum number of iteration per episode')
    parser.add_argument('--alpha', default=0.01, dest='alpha', type=np.float32,
                        help='the learning rate')
    parser.add_argument('--epsilon', default=0.2, dest='epsilon', type=np.float32,
                        help='the possibility of exploring in epsilon-greedy action choosing algorithm')
    parser.add_argument('--gamma', default=0.9, dest='gamma', type=np.float32,
                        help='the learning rate')
    parser.add_argument('--stop_reward', default=200.0, dest='stop_reward', type=np.float32,
                        help='the cumulative reward early stop condition')
    parser.add_argument('--render_mode', default='human', dest='render_mode', type=str,
                        help='the mode of renderer in gym package')
    parser.add_argument('--debug_print', dest='debug_print', default=False, type=bool,
                        help='To print the state of system every iteration, for debugging only.')
    parser.add_argument('--visualization', dest='visualization', default=False, type=bool,
                        help='Visualization for environment.')
    return parser

if __name__ == '__main__':
    main()


