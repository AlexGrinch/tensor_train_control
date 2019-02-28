import os
import time

from IPython import display
import matplotlib.pyplot as plt

from trainers import *
from normal_buffer import ReplayBuffer


class AgentArena:
    
    def __init__(
            self, train_env, test_env, 
            num_actions, trainers,
            save_path="rl_models", model_name="model"):
        self.env = train_env
        self.test_env = test_env
        self.num_actions = num_actions
        self.trainers = trainers

        self.path = save_path + "/" + model_name
        if not os.path.exists(self.path):
            os.makedirs(self.path)
            
    def set_parameters(
            self, replay_memory_size=100000, replay_start_size=10000,
            init_eps=1, final_eps=0.02, annealing_steps=100000,
            discount_factor=0.99, max_episode_length=2000, frame_history_len=1):

        self.rep_buffer = ReplayBuffer(replay_memory_size)
        frame_count = 0
        
        transitions = []
        for i in range(7):
            for j in range(10):
                valid, s = self.env.set_pos((i, j))
                if valid:
                    for k in range(400):
                        for a in range(4):
                            s_, r, done = self.env.step(a)
                            self.rep_buffer.push_transition(
                                (s, a, r, s_, done))
                            valid, s = self.env.set_pos((i, j))

        # define epsilon decrement schedule for exploration
        self.eps = init_eps
        self.final_eps = final_eps
        self.eps_drop = (init_eps - final_eps) / annealing_steps

        self.gamma = discount_factor
        self.max_ep_length = max_episode_length
        
        test_states = []
        for i in range(7):
            for j in range(10):
                valid, s = self.env.set_pos((i, j))
                if valid:
                    test_states.append(s)
        self.test_states = np.array(test_states)
        
    def start(
            self,
            gpu_id=0,
            batch_size=32,
            exploration="e-greedy",
            agent_update_freq=4,
            target_update_freq=5000,
            max_num_frames=1000000,
            performance_print_freq=500,
            test_freq=1000):

        config = self.gpu_config(gpu_id)
        self.batch_size = batch_size
        train_rewards = []
        test_rewards = [[] for i in range(len(self.trainers))]
        test_q_values = [[] for i in range(len(self.trainers))]
        frame_counts = []
        frame_count = 0
        
        with tf.Session(config=config) as sess:
            sess.run(tf.global_variables_initializer())
            episode_count = 0
            while frame_count < max_num_frames:
                
                train_ep_reward = 0
                last_obs = self.env.reset()
                
                for time_step in range(self.max_ep_length):
                    
                    action = np.random.randint(self.num_actions)
                    obs, reward, done = self.env.step(action)[:3]
                    frame_count += 1
                    train_ep_reward += reward
                    self.eps = max(self.final_eps, self.eps - self.eps_drop)
                    
                    if frame_count % agent_update_freq == 0:
                        batch = self.rep_buffer.get_batch(batch_size)
                        for t in self.trainers:
                            t.train(sess, batch)

                    if frame_count % target_update_freq == 0:
                        for t in self.trainers:
                            t.update_target(sess)
                            
                    if frame_count % test_freq == 0:
                        for idx, t in enumerate(self.trainers):
                            rew = self.evaluate_episode(sess, t)
                            test_rewards[idx].append(rew)
                            q_values = t.get_q_values(sess, self.test_states)
                            test_q_values[idx].append(q_values)
                        np.savez(
                            self.path+"/learning_curve.npz", train=train_rewards,
                            frame=frame_counts, test=test_rewards, q_vals=test_q_values)
                            
                    if done:
                        break
                    last_obs = obs
                    
                episode_count += 1
                train_rewards.append(train_ep_reward)
                frame_counts.append(frame_count)
                
                if episode_count % performance_print_freq == 0:
                    avg_reward = np.mean(train_rewards[-performance_print_freq:])
                    print("frame count:", frame_count)
                    print("average reward:", avg_reward)
                    print("epsilon:", round(self.eps, 3))
                    print("-------------------------------")
                    
    def evaluate_episode(self, sess, trainer):
        ep_reward = 0
        last_obs = self.test_env.reset()
        for time_step in range(self.max_ep_length):
            recent_obs = np.array(last_obs).astype(np.uint8)
            action = trainer.get_greedy_action(sess, [recent_obs])
            obs, reward, done = self.test_env.step(action)[:3]
            ep_reward += reward
            if done:
                break
            last_obs = obs
        return ep_reward

    def gpu_config(self, gpu_id):
        if (gpu_id == -1):
            config = tf.ConfigProto()
        else:
            os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"  
            os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
            config = tf.ConfigProto()
            config.gpu_options.allow_growth = True
            config.intra_op_parallelism_threads = 1
            config.inter_op_parallelism_threads= 1
        return config
