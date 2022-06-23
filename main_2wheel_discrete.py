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
    env = gym.make('RobotInterception-v2')
    total_reward = []
    state = env.reset()
    position_size = env.img_height + 1
    error_size = env.img_width + 1
    state_size = position_size * error_size
    action_size = 9
    Q = np.zeros([state_size, action_size])
    Q_idx = np.zeros([position_size, error_size], dtype=int)

    i = 0
    for p in range(position_size):
        for e in range(error_size):
            Q_idx[p, e] = i
            i += 1

    action_dict = {
        0: np.array([0, 0]),
        1: np.array([0, 1]),
        2: np.array([0, 2]),
        3: np.array([1, 0]),
        4: np.array([1, 1]),
        5: np.array([1, 2]),
        6: np.array([2, 0]),
        7: np.array([2, 1]),
        8: np.array([2, 2])
    }

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
        while not done and i <= args.max_iterations:

            position = state[3]
            error = state[2]
            state_idx = Q_idx[position, error+160]
            # print(yaw)
            action, action_idx = get_action(env, Q, state_idx, action_dict, args.epsilon)
            # print(type(action_idx))
            # action = action_dict[action_idx]
            next_state, reward, done, info = env.step(action)
            next_position = next_state[3]
            next_error = next_state[2]
            try:
                next_state_idx = Q_idx[next_position, next_error+160]
            except:
                print(f"Step {i}:")
                print(f"Current position of robot: {env.robot}")
                print(f"Current position of target: {env.target}")
                print(f"Current action: {action}")
                print(f"Current position reward: {reward_pos}; rotation reward: {reward_error}; total reward: {reward}")
                # print(f"total reward is {reward_sum}, step reward is {reward}, action is {action}, error is {error}, "
                print(f"State is {state}, next State is {next_state}.\n")
            if abs(next_error) < abs(error):
                reward_error = 1/i  #(abs(error)-abs(next_error))/i
                if abs(error) <= 3:
                    reward_error = 10
            else:
                reward_error = 0
            if position < next_position:
                reward_pos = 1/i
                if position > 0.9 * env.img_height:
                    reward_pos = 10
            else:
                reward_pos = 0

            reward = reward_pos + reward_error
            # print(error, next_error)
            Q[state_idx, action_idx] += args.alpha * (reward + args.gamma * np.max(Q[next_state_idx]) - Q[state_idx, action_idx])
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

def get_action(env, Q, state_idx, action_dict, epsilon):
    if random.random() <= epsilon:
        action = env.action_space.sample()
        action_idx = 3 * (action[0]) + (action[1])
    else:
        action_idx = np.argmax(Q[state_idx])
        action = action_dict[action_idx]
    return action, action_idx

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
    return parser

if __name__ == '__main__':
    main()


