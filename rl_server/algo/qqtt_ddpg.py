import tensorflow as tf
from .ddpg import BaseDDPG


class QQTTDDPG(BaseDDPG):
    def __init__(self,
                 state_shapes,
                 action_size,
                 critic,
                 critic_optimizer,
                 n_step=1,
                 discount_factor=0.99,
                 target_critic_update_rate=1.0):

        self._state_shapes = state_shapes
        self._action_size = action_size
        self._critic = critic
        self._target_critic = critic.copy(scope='target_critic')
        self._critic_optimizer = critic_optimizer
        self._n_step = n_step
        self._gamma = discount_factor
        self._update_rates = [target_critic_update_rate]
        self._target_critic_update_rate = tf.constant(target_critic_update_rate)
        
        self._create_placeholders()
        self._create_variables()

    def _get_action_for_state(self):
        actions = self._critic([self._state_for_act])
        return actions

    def _get_q_values(self, states, actions):
        q_values = self._critic([states, actions])
        return q_values

    def _get_critic_update(self):

        # left hand side of the Bellman equation
        agent_q = self._critic([self._state, self._given_action])[:, None]

        # right hand side of the Bellman equation
        next_action = self._target_critic([self._next_state])
        next_q = self._target_critic([self._next_state, next_action])[:, None]
        discount = self._gamma ** self._n_step
        target_q = self._rewards[:, None] + discount * (1 - self._terminator[:, None]) * next_q
        
        print (agent_q, next_q, target_q)

        # critic gradient and update rule
        critic_loss = tf.losses.huber_loss(agent_q, tf.stop_gradient(target_q))
        critic_gradients = self._critic_optimizer.compute_gradients(
            critic_loss, var_list=self._critic.variables())
        critic_update = self._critic_optimizer.apply_gradients(critic_gradients)

        return [critic_loss, tf.reduce_mean(agent_q**2)], critic_update

    def _get_target_critic_update(self):
        target_critic_update = BaseDDPG._update_target_network(
            self._critic, self._target_critic, self._target_critic_update_rate)
        return target_critic_update

    def _get_targets_init(self):
        target_critic_update = BaseDDPG._update_target_network(self._critic, self._target_critic, 1.0)
        return target_critic_update

    def _create_variables(self):

        with tf.name_scope("taking_action"):
            self._actor_action = self._get_action_for_state()

        with tf.name_scope("critic_update"):
            self._critic_loss, self._critic_update = self._get_critic_update()

        with tf.name_scope("target_networks_update"):
            self._targets_init = self._get_targets_init()
            self._target_critic_update = self._get_target_critic_update()

    def act_batch(self, sess, states):
        feed_dict = {}
        for i in range(len(states)):
            feed_dict[self._state_for_act[i]] = states[i]
        actions = sess.run(self._actor_action, feed_dict=feed_dict)
        return actions.tolist()

    def train(self, sess, batch):

        feed_dict = {self._rewards: batch.r,
                     self._given_action: batch.a,
                     self._terminator: batch.done}
        for i in range(len(batch.s)):
            feed_dict[self._state[i]] = batch.s[i]
            feed_dict[self._next_state[i]] = batch.s_[i]
        loss, _ = sess.run([self._critic_loss,
                            self._critic_update],
                           feed_dict=feed_dict)
        return loss

    def target_actor_update(self, sess):
        pass
        
    def target_critic_update(self, sess):
        sess.run(self._target_critic_update)

    def target_network_init(self, sess):
        sess.run(self._targets_init)
        
    def _get_info(self):
        info = {}
        info['algo'] = 'ddpg'
        info['critic'] = self._critic.get_info()
        info['discount_factor'] = self._gamma
        info['target_critic_update_rate'] = self._update_rates[0]
        return info
