import numpy as np
import copy
import torch
import torch.nn as nn

from .utils import *


class StateActionNetwork(nn.Module):
    def __init__(
        self,
        num_actions,
        state_shape,
        convs=[[32, 4, 2], [64, 2, 1]],
        hiddens=[128],
        num_atoms=1
    ):
        super(StateActionNetwork, self).__init__()
        self._num_actions = num_actions
        self._num_atoms = num_atoms

        # convolutional part of the network
        conv_net = []
        in_channels = state_shape[-1]
        for conv in convs:
            out_channels, kernel_size, stride = conv
            conv_net.append(
                nn.Conv2d(in_channels, out_channels, kernel_size, stride)
            )
            conv_net.append(nn.ReLU())
            in_channels = out_channels
        self.conv_net = nn.Sequential(*conv_net)

        # calculate number of features after flattened convolutions
        hws = state_shape[:-1]
        for i in range(len(convs)):
            out_channels, kernel_size, stride = convs[i]
            for j in range(len(hws)):
                hws[j] = (hws[j] - kernel_size) // stride + 1
        in_features = np.prod(hws) * out_channels

        # fully connected part of the network
        fc_net = []
        for hidden in hiddens:
            fc_net.append(nn.Linear(in_features, hidden))
            fc_net.append(nn.ReLU())
            in_features = hidden
        self.fc_net = nn.Sequential(*fc_net)

        # head of the network
        self.head_net = nn.Linear(hidden, num_actions * num_atoms, bias=False)

    def forward(self, x):
        batch_size = x.shape[0]
        x = self.conv_net(x)
        x = x.view(batch_size, -1)
        x = self.fc_net(x)
        x = self.head_net(x)
        x = x.view(-1, self._num_actions, self._num_atoms)
        return x


class Trainer:
    def __init__(
        self,
        num_actions,
        state_shape,
        convs=[[32, 4, 2], [64, 2, 1]],
        hiddens=[128],
        num_atoms=1,
        values_range=(-10., 10.),
        distribution=None,
        optimizer=torch.optim.Adam,
        optimizer_params={"lr": 2.5e-4, "eps": 0.01/32},
        huber_loss_delta=10.,
        gamma=0.99
    ):

        self.gamma = gamma
        self.num_atoms = num_atoms
        self.distribution = distribution

        self._device = torch.device("cpu")

        self.agent_net = StateActionNetwork(
            num_actions, state_shape, convs, hiddens, num_atoms
        ).to(self._device)
        self.target_net = copy.deepcopy(self.agent_net).to(self._device)
        self.optimizer = optimizer(
            self.agent_net.parameters(), **optimizer_params
        )
        self.criterion = HuberLoss(huber_loss_delta)

        self._loss_fn = self._base_loss
        if self.distribution == "quantile":
            tau_min = 1 / (2 * self.num_atoms)
            tau_max = 1 - tau_min
            self.tau = torch.linspace(
                start=tau_min, end=tau_max, steps=self.num_atoms
            ).to(self._device)
            self._loss_fn = self._quantile_loss
        elif self.distribution == "categorical":
            self.v_min, self.v_max = values_range
            self.delta_z = (self.v_max - self.v_min) / (self.num_atoms - 1)
            self.z = torch.linspace(
                start=self.v_min, end=self.v_max, steps=self.num_atoms
            ).to(self._device)
            self._loss_fn = self._categorical_loss

    def _to_tensor(self, *args, **kwargs):
        return torch.from_numpy(*args, **kwargs).to(self._device)

    def train(self, batch):
        states_t = self._to_tensor(batch.s).permute(0, 3, 1, 2).float()
        actions_t = self._to_tensor(batch.a).long()
        rewards = self._to_tensor(batch.r).float()
        states_tp1 = self._to_tensor(batch.s_).permute(0, 3, 1, 2).float()
        dones = self._to_tensor(batch.done).float()

        loss = self._loss_fn(states_t, actions_t, rewards, states_tp1, dones)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

    def update_target(self):
        soft_update(self.target_net, self.agent_net, 1.0)

    def get_q_values(self, states):
        states = self._to_tensor(states).permute(0, 3, 1, 2).float()
        net_outputs = self.agent_net(states)
        if self.distribution == "quantile":
            q_values = net_outputs.mean(dim=-1)
        elif self.distribution == "categorical":
            probs = torch.softmax(net_outputs, dim=-1)
            q_values = torch.sum(probs * self.z, dim=-1)
        else:
            q_values = net_outputs.squeeze(1)
        return q_values.detach().cpu().numpy()

    def get_greedy_action(self, state):
        states = self._to_tensor(state).unsqueeze(0).permute(0, 3, 1, 2)
        q_values = self.get_q_values(states)[0]
        action = np.argmax(q_values)
        return action

    def _base_loss(self, states_t, actions_t, rewards_t, states_tp1, done_t):

        q_values_t = self.agent_net(states_t).squeeze(-1).gather(-1, actions_t)
        q_values_tp1 = \
            self.target_net(states_tp1).squeeze(-1).max(-1, keepdim=True)[0]
        q_target_t = \
            rewards_t + (1 - done_t) * self.gamma * q_values_tp1.detach()
        value_loss = self.criterion(q_values_t, q_target_t).mean()

        return value_loss

    def _quantile_loss(
        self, states_t, actions_t, rewards_t, states_tp1, done_t
    ):

        indices_t = actions_t.repeat(1, self.num_atoms).unsqueeze(1)
        atoms_t = self.agent_net(states_t).gather(1, indices_t).squeeze(1)

        all_atoms_tp1 = self.target_net(states_tp1).detach()
        q_values_tp1 = all_atoms_tp1.mean(dim=-1)
        actions_tp1 = torch.argmax(q_values_tp1, dim=-1, keepdim=True)
        indices_tp1 = actions_tp1.repeat(1, self.num_atoms).unsqueeze(1)
        atoms_tp1 = all_atoms_tp1.gather(1, indices_tp1).squeeze(1)
        atoms_target_t = rewards_t + (1 - done_t) * self.gamma * atoms_tp1

        value_loss = quantile_loss(
            atoms_t, atoms_target_t, self.tau, self.num_atoms, self.criterion
        )

        return value_loss

    def _categorical_loss(
        self, states_t, actions_t, rewards_t, states_tp1, done_t
    ):

        indices_t = actions_t.repeat(1, self.num_atoms).unsqueeze(1)
        logits_t = self.agent_net(states_t).gather(1, indices_t).squeeze(1)

        all_logits_tp1 = self.target_net(states_tp1).detach()
        q_values_tp1 = torch.sum(
            torch.softmax(all_logits_tp1, dim=-1) * self.z, dim=-1
        )
        actions_tp1 = torch.argmax(q_values_tp1, dim=-1, keepdim=True)
        indices_tp1 = actions_tp1.repeat(1, self.num_atoms).unsqueeze(1)
        logits_tp1 = all_logits_tp1.gather(1, indices_tp1).squeeze(1)
        atoms_target_t = rewards_t + (1 - done_t) * self.gamma * self.z

        value_loss = categorical_loss(
            logits_t, logits_tp1, atoms_target_t, self.z, self.delta_z,
            self.v_min, self.v_max
        )

        return value_loss
