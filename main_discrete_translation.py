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
    env = gym.make('RobotInterception-v3')
    total_reward = []
    state = env.reset()
    position_size = env.img_height + 1
    error_size = env.img_width + 1
    state_size = env.img_height + 1
    action_size = env.action_space.n
    Q = np.zeros([state_size, action_size])

    i = 0

    for episode in range(args.max_episode+1):
        done = False
        state = env.reset()
        reward = 0
        reward_sum = 0
        reward_sum_pos = 0
        i = 1
        init_state = state
        action_sum = 0
        while not done and i <= args.max_iterations:

            position = state[3]
            action = get_action(env, Q, args.epsilon, position)

            next_state, reward, done, info = env.step(action)
            next_position = next_state[3]

            if position < next_position:
                reward = (next_position - position)/i
                if position > 0.9 * env.img_height:
                    reward = 10
            else:
                reward_pos = -1


            Q[position, action] += args.alpha * (reward + args.gamma * np.max(Q[next_position]) - Q[position, action])
            if args.debug_print and episode >=  args.max_episode:
                print(f"Step {i}:")
                print(f"Current position of robot: {env.robot}")
                print(f"Current position of target: {env.target}")
                print(f"Current action: {action}")
                print(f"Current position reward: {reward_pos}; total reward: {reward}")
                # print(f"total reward is {reward_sum}, step reward is {reward}, action is {action}, error is {error}, "
                print(f"State is {state}, next State is {next_state}.\n")

            reward_sum += reward

            # action_sum += action

            state = next_state
            i += 1

            # print(f"total reward is {reward_sum}, step reward is {reward}, action is {action}, done is {done}.")
        reward_avg = reward_sum
        total_reward.append(reward_sum)


        if episode % 50 == 0:
            # print()
            print(f"Episode: {episode}:")
            print(f"Sum reward: {reward_avg}; Final state: {state}; Initial state: {init_state};")
            print(f"Final step reward: {reward_pos}; Times of iters: {i}")

def get_action(env, Q, epsilon, state):
    if random.random() <= epsilon:
        action = env.action_space.sample()
    else:
        action = np.argmax(Q[state])
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
    return parser

if __name__ == '__main__':
    main()


