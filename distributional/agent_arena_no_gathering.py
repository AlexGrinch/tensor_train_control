import os
import time
import numpy as np
import torch
import torch.nn as nn
from IPython import display
import matplotlib.pyplot as plt

from trainers import Trainer
from normal_buffer import ReplayBuffer


class AgentArena:

    def __init__(
        self, train_env, num_actions, trainers,
        save_path="rl_models", model_name="model"
    ):

        self.env = train_env
        self.num_actions = num_actions
        self.trainers = trainers

        self.path = save_path + "/" + model_name
        if not os.path.exists(self.path):
            os.makedirs(self.path)

    def set_parameters(
        self, replay_memory_size=100000, discount_factor=0.99,
        max_episode_length=100, num_samples_per_transition=400
    ):

        self.gamma = discount_factor
        self.rep_buffer = ReplayBuffer(replay_memory_size)
        self.max_ep_length = max_episode_length

        # fill the buffer with all possible transitions with repetitions
        for i in range(self.env.w):
            for j in range(self.env.h):
                valid, s = self.env.set_pos((i, j))
                if valid:
                    for k in range(num_samples_per_transition):
                        for a in range(self.num_actions):
                            s_, r, done = self.env.step(a)
                            self.rep_buffer.push_transition(
                                (s, a, r, s_, done))
                            _, s = self.env.set_pos((i, j))

        # list of unique states to track q-values during training
        test_states = []
        for i in range(self.env.w):
            for j in range(self.env.h):
                valid, s = self.env.set_pos((i, j))
                if valid:
                    test_states.append(s)
        self.test_states = np.array(test_states)

    def start(
        self,
        batch_size=32,
        agent_update_freq=4,
        target_update_freq=5000,
        max_num_frames=1000000,
        test_freq=1000
    ):

        self.batch_size = batch_size
        test_rewards = [[] for i in range(len(self.trainers))]
        test_num_steps = [[] for i in range(len(self.trainers))]
        test_q_values = [[] for i in range(len(self.trainers))]
        frame_count = 0
        episode_count = 0

        while frame_count < max_num_frames:
            frame_count += 1

            if frame_count % agent_update_freq == 0:
                batch = self.rep_buffer.get_batch(batch_size)
                for t in self.trainers:
                    t.train(batch)

            if frame_count % target_update_freq == 0:
                for t in self.trainers:
                    t.update_target()

            if frame_count % test_freq == 0:
                for idx, t in enumerate(self.trainers):
                    rew, num_steps = self.evaluate_episode(t)
                    test_rewards[idx].append(rew)
                    test_num_steps[idx].append(num_steps)
                    q_values = t.get_q_values(self.test_states)
                    test_q_values[idx].append(q_values)
                np.savez(
                    self.path+"/learning_curve.npz",
                    test=test_rewards,
                    test_num_steps=test_num_steps,
                    q_vals=test_q_values
                )

            episode_count += 1

    def evaluate_episode(self, trainer):
        ep_reward = 0
        last_obs = self.env.reset()
        for time_step in range(self.max_ep_length):
            recent_obs = np.array(last_obs).astype(np.uint8)
            action = trainer.get_greedy_action(recent_obs)
            obs, reward, done = self.env.step(action)[:3]
            ep_reward += reward
            if done:
                break
            last_obs = obs
        return ep_reward, time_step + 1
