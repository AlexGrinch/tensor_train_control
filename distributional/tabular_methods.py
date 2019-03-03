import matplotlib.pyplot as plt
import numpy as np


class QTable:

    def __init__(self, env, gamma=0.99):
        self.w, self.h = env.w, env.h
        self.q = np.zeros((self.w, self.h, 4))
        self.pi = np.ones((self.w, self.h, 4)) / 4
        self.env = env
        self.env.reset()
        self.alpha = 0.3
        self.gamma = gamma
        self.end_states = set([
            (self.w - 1, 0),
            (self.env.x_wall + 1, self.env.y_hole2 + 1),
            (self.env.x_wall + 1, self.env.y_hole2 - 1)
        ])

    def update_xya(self, x, y, a):
        td_target = 0
        for act in range(4):
            if (act == a):
                param = 1.0 - 0.75 * self.env.stochasticity
            else:
                param = 0.25 * self.env.stochasticity

            self.env.set_pos((x, y))
            x_, y_ = self.env.move(act)
            self.env.set_pos((x_, y_))
            reward, done = self.env.get_reward_done()

            if (x, y) not in self.end_states:
                q_ = (self.q[x_, y_]*self.pi[x_, y_]).sum()
                td_target += param * (reward + self.gamma * q_ * (1 - done))

        self.q[x,y,a] = self.q[x,y,a] + self.alpha * (td_target - self.q[x,y,a])
            
    def update_field(self, num_times=10):
        q_old = self.q.copy()
        for i in range(num_times):
            for x in range(self.w):
                for y in range(self.h):
                    for a in range(4):
                        if (x == self.env.x_wall):
                            if (y == self.env.y_hole or y == self.env.y_hole2):
                                self.update_xya(x, y, a)
                        else:
                            self.update_xya(x, y, a)
        res = np.linalg.norm(q_old - self.q)
        return res
            
    def update_policy(self):
        new_pi = np.zeros((self.w, self.h, 4))
        best_acts = np.argmax(self.q, axis=2)
        for a in range(4):
            indices = np.where(best_acts == a)
            new_pi[:,:,a][indices] = 1
        self.pi = new_pi
        
    def run_policy_iteration(self, pol_impr_times=100, pol_eval_times=10, tol=1e-6):
        for i in range(pol_impr_times):
            res = self.update_field(pol_eval_times)
            self.update_policy()
            if (res < tol): break
                
    def plot_q_values(self, figsize=(15, 5)):
        fig, ax = plt.subplots(1, 4, figsize=figsize)
        for a in range(4):
            img = np.rot90(self.q[:,:,a])
            ax[a].imshow(img, cmap='gray', vmin=-1, vmax=1)
            
    def q_learning(self, max_num_iter=100, max_ep_len=100, lr=0.1, state=(0, 0)):
        self.q = np.zeros((self.w, self.h, 4))
        eps = 1
        eps_fin = 0.1
        delta_eps = (eps - eps_fin) / 1000
        
        x_eval, y_eval = state
        a_eval = 0
        td_values = []
        
        for it in range(max_num_iter):
            eps = max(eps, eps_fin)
            x, y = self.env.reset()
            td_values.append(self.q[x_eval,y_eval,a_eval])
            for step in range(max_ep_len):
                if np.random.rand() < eps:
                    a = np.random.randint(4)
                else:
                    a = np.argmax(self.q[x,y])
                (x_, y_), reward, done = self.env.step(a)
                a_ = np.argmax(self.q[x_,y_])
                td_error = reward + self.gamma*self.q[x_,y_,a_]*(1-done) - self.q[x,y,a]
                self.q[x,y,a] = self.q[x,y,a] + lr * td_error
                x, y = x_, y_
                if done: break
            eps = eps - delta_eps
            
            
            if it % 2000 == 0:
                lr = lr * 0.8
        
        return td_values

    def qrq_learning(self, num_atoms=32, max_num_iter=100, max_ep_len=100, lr=0.1,
                     state=(0, 0)):
        self.z = np.zeros((self.w, self.h, 4, num_atoms))
        self.q = np.zeros((self.w, self.h, 4))
        eps = 1
        eps_fin = 0.1
        delta_eps = (eps - eps_fin) / 1000
        
        tau_min = 1 / (2 * num_atoms) 
        tau_max = 1 - tau_min
        taus = np.linspace(start=tau_min, stop=tau_max, num=num_atoms)
        
        x_eval, y_eval = state
        a_eval = 0
        
        hists = []
        qrtd_values = []
        
        for it in range(max_num_iter):
            eps = max(eps, eps_fin)
            x, y = self.env.reset()
            qrtd_values.append(self.z[x_eval, y_eval, a_eval].mean())
            for step in range(max_ep_len):
                if np.random.rand() < eps:
                    a = np.random.randint(4)
                else:
                    a = np.argmax(self.z[x,y].mean(axis=-1), axis=-1)
                (x_, y_), reward, done = self.env.step(a)
                a_ = np.argmax(self.z[x_,y_].mean(axis=-1), axis=-1)
                
                delta = (reward + self.gamma * self.z[x_,y_,a_][:,None] * \
                         (1-done) < self.z[x,y,a][None,:]).mean(axis=0)
                qrtd_error = taus - delta
                self.z[x,y,a] = self.z[x,y,a] + lr * qrtd_error
                
                x, y = x_, y_
                if done: break
            self.q = np.mean(self.z, axis=-1)
            eps = eps - delta_eps   

            if it % 100 == 0:
                hists.append(self.z[0,0,0].copy())
                
            if it % 2000 == 0:
                lr = lr * 0.8
        return hists, np.array(qrtd_values)