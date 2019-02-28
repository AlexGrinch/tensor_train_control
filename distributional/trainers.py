import numpy as np
import random
import tensorflow as tf
import tensorflow.contrib.layers as layers
from tensorflow.contrib.layers import convolution2d as conv
from tensorflow.contrib.layers import fully_connected as fc
from tensorflow.contrib.layers import xavier_initializer as xavier

####################################################################################################
########################################### Core modules ###########################################
####################################################################################################

def conv_module(input_layer, convs, activation_fn=tf.nn.relu):
    """ convolutional module
    """
    out = input_layer
    for num_outputs, kernel_size, stride in convs:
        out = conv(
            out,
            num_outputs=num_outputs,
            kernel_size=kernel_size,
            stride=stride,
            padding='VALID',
            activation_fn=activation_fn)
    return out
    
def fc_module(input_layer, fully_connected, activation_fn=tf.nn.relu):
    """ fully connected module
    """
    out = input_layer
    for num_outputs in fully_connected:
        out = fc(
            out,
            num_outputs=num_outputs,
            activation_fn=activation_fn,
            weights_initializer=xavier())
    return out

def full_module(
        input_layer, convs, fully_connected,
        num_outputs, activation_fn=tf.nn.relu):
    """ convolutional + fully connected + output
    """
    out = input_layer
    out = conv_module(out, convs, activation_fn)
    out = layers.flatten(out)
    out = fc_module(out, fully_connected, activation_fn)
    out = fc_module(out, [num_outputs], None)
    return out


class DQNTrainer:
    
    def __init__(
            self, num_actions, state_shape=[8, 8, 5],
            convs=[[32, 4, 2], [64, 2, 1]],
            fully_connected=[128],
            activation_fn=tf.nn.relu,
            optimizer=tf.train.AdamOptimizer(2.5e-4, epsilon=0.01/32),
            optimizer_target=tf.train.AdamOptimizer(2.5e-4, epsilon=0.01/32),
            gradient_clip=10.0, gamma=0.99, scope="dqn", reuse=False):
        self.gamma = gamma

        self.agent_net = DeepQNetwork(
            num_actions, state_shape, convs, fully_connected,
            activation_fn, optimizer, gradient_clip,
            scope=scope+"_agent", reuse=reuse)
        
        self.target_net = DeepQNetwork(
            num_actions, state_shape, convs, fully_connected,
            activation_fn, optimizer_target, gradient_clip,
            scope=scope+"_target", reuse=reuse)
        
        self.agent_vars = tf.global_variables(scope=scope+"_agent")
        self.target_vars = tf.global_variables(scope=scope+"_target")
        
    def train(self, sess, batch):
        agent_actions = self.agent_net.get_q_argmax(sess, batch.s_)
        q_double = self.target_net.get_q_values_sa(sess, batch.s_, agent_actions)
        targets = batch.r + (self.gamma * q_double * (1 - batch.done))
        self.agent_net.update(sess, batch.s, batch.a, targets)
        
    def get_greedy_action(self, sess, states):
        return self.agent_net.get_q_argmax(sess, states)
    
    def update_target(self, sess):
        update_ops = []
        for v_agnt, v_trgt in zip(
                self.agent_vars, self.target_vars):
            update_ops.append(v_trgt.assign(v_agnt))
        sess.run(update_ops)

    def get_q_values(self, sess, states):
        return self.agent_net.get_q_values_s(sess, states)


class QuantileTrainer:
    
    def __init__(
            self, num_actions, state_shape=[8, 8, 5],
            convs=[[32, 4, 2], [64, 2, 1]],
            fully_connected=[128],
            activation_fn=tf.nn.relu, num_atoms=50, kappa=1.0,
            optimizer=tf.train.AdamOptimizer(2.5e-4, epsilon=0.01/32),
            optimizer_target=tf.train.AdamOptimizer(2.5e-4, epsilon=0.01/32),
            gamma=0.99, scope="quant_dqn", reuse=False):
        self.gamma = gamma

        self.agent_net = QuantileRegressionDeepQNetwork(
            num_actions, state_shape, convs, fully_connected,
            num_atoms, kappa, activation_fn, optimizer,
            scope=scope+"_agent", reuse=reuse)
        
        self.target_net = QuantileRegressionDeepQNetwork(
            num_actions, state_shape, convs, fully_connected,
            num_atoms, kappa, activation_fn, optimizer_target,
            scope=scope+"_target", reuse=reuse)
        
        self.agent_vars = tf.global_variables(scope=scope+"_agent")
        self.target_vars = tf.global_variables(scope=scope+"_target")

    def train(self, sess, batch):
        agent_actions = self.agent_net.get_q_argmax(sess, batch.s_)
        next_atoms = self.target_net.get_atoms_sa(sess, batch.s_, agent_actions)
        target_atoms = batch.r[:, None] + self.gamma * next_atoms * (1 - batch.done[:, None])
        self.agent_net.update(sess, batch.s, batch.a, target_atoms)   
        
    def get_greedy_action(self, sess, states):
        return self.agent_net.get_q_argmax(sess, states)
    
    def update_target(self, sess):
        update_ops = []
        for v_agnt, v_trgt in zip(
                self.agent_vars, self.target_vars):
            update_ops.append(v_trgt.assign(v_agnt))
        sess.run(update_ops)

    def get_q_values(self, sess, states):
        return self.agent_net.get_q_values_s(sess, states)
    
    
class CategoricalTrainer:
    
    def __init__(
            self, num_actions, state_shape=[8, 8, 5],
            convs=[[32, 4, 2], [64, 2, 1]],
            fully_connected=[128],
            activation_fn=tf.nn.relu, num_atoms=50, v=(-5, 5),
            optimizer=tf.train.AdamOptimizer(2.5e-4, epsilon=0.01/32),
            optimizer_target=tf.train.AdamOptimizer(2.5e-4, epsilon=0.01/32),
            gamma=0.99, scope="cat_dqn", reuse=False):
        self.gamma = gamma

        self.agent_net = CategoricalDeepQNetwork(
            num_actions, state_shape, convs, fully_connected,
            num_atoms, v, activation_fn, optimizer,
            scope=scope+"_agent", reuse=reuse)
        
        self.target_net = CategoricalDeepQNetwork(
            num_actions, state_shape, convs, fully_connected,
            num_atoms, v, activation_fn, optimizer_target,
            scope=scope+"_target", reuse=reuse)
        
        self.agent_vars = tf.global_variables(scope=scope+"_agent")
        self.target_vars = tf.global_variables(scope=scope+"_target")

    def train(self, sess, batch):
        agent_actions = self.agent_net.get_q_argmax(sess, batch.s_)
        target_probs = self.target_net.cat_proj(
            sess, batch.s_, agent_actions, batch.r, batch.done, gamma=self.gamma)
        self.agent_net.update(sess, batch.s, batch.a, target_probs)  

    def get_greedy_action(self, sess, states):
        return self.agent_net.get_q_argmax(sess, states)

    def update_target(self, sess):
        update_ops = []
        for v_agnt, v_trgt in zip(
                self.agent_vars, self.target_vars):
            update_ops.append(v_trgt.assign(v_agnt))
        sess.run(update_ops)

    def get_q_values(self, sess, states):
        return self.agent_net.get_q_values_s(sess, states)

####################################################################################################
########################################## Deep Q-Network ##########################################
####################################################################################################

class DeepQNetwork:

    def __init__(self, num_actions, state_shape=[8, 8, 5],
                 convs=[[32, 4, 2], [64, 2, 1]], 
                 fully_connected=[128],
                 activation_fn=tf.nn.relu,
                 optimizer=tf.train.AdamOptimizer(2.5e-4, epsilon=0.01/32),
                 gradient_clip=10.0,
                 scope="dqn", reuse=False):
        
        with tf.variable_scope(scope, reuse=reuse):

            ########################### Neural network architecture ###########################

            input_shape = [None] + state_shape
            self.input_states = tf.placeholder(dtype=tf.float32, shape=input_shape)

            self.q_values = full_module(self.input_states, convs, fully_connected,
                                        num_actions, activation_fn)

            ############################## Optimization procedure #############################

            # convert input actions to indices for q-values selection
            self.input_actions = tf.placeholder(dtype=tf.int32, shape=[None])
            indices_range = tf.range(tf.shape(self.input_actions)[0])
            action_indices = tf.stack([indices_range, self.input_actions], axis=1)
            
            # select q-values for input actions
            self.q_values_selected = tf.gather_nd(self.q_values, action_indices)
            
            # select best actions (according to q-values)
            self.q_argmax = tf.argmax(self.q_values, axis=1)

            # define loss function and update rule
            self.q_targets = tf.placeholder(dtype=tf.float32, shape=[None])
            self.loss = tf.losses.huber_loss(self.q_targets, self.q_values_selected, 
                                             delta=gradient_clip)
            self.update_model = optimizer.minimize(self.loss)

    def get_q_values_s(self, sess, states):
        feed_dict = {self.input_states:states}
        q_values = sess.run(self.q_values, feed_dict)
        return q_values
    
    def get_q_values_sa(self, sess, states, actions):
        feed_dict = {self.input_states:states, self.input_actions:actions}
        q_values_selected = sess.run(self.q_values_selected, feed_dict)
        return q_values_selected
    
    def get_q_argmax(self, sess, states):
        feed_dict = {self.input_states:states}
        q_argmax = sess.run(self.q_argmax, feed_dict)
        return q_argmax

    def update(self, sess, states, actions, q_targets):

        feed_dict = {self.input_states:states,
                     self.input_actions:actions,
                     self.q_targets:q_targets}
        sess.run(self.update_model, feed_dict)
        
####################################################################################################
################################ Qantile Regression Deep Q-Network #################################
####################################################################################################

class QuantileRegressionDeepQNetwork:
    
    def __init__(self, num_actions, state_shape=[8, 8, 5],
                 convs=[[32, 4, 2], [64, 2, 1]],
                 fully_connected=[128], 
                 num_atoms=50, kappa=1.0,
                 activation_fn=tf.nn.relu,
                 optimizer=tf.train.AdamOptimizer(2.5e-4, epsilon=0.01/32),
                 scope="qr_dqn", reuse=False):
        
        with tf.variable_scope(scope, reuse=reuse):

            ########################### Neural network architecture ###########################

            input_shape = [None] + state_shape
            self.input_states = tf.placeholder(dtype=tf.float32, shape=input_shape)
        
            # distribution parameters
            tau_min = 1 / (2 * num_atoms) 
            tau_max = 1 - tau_min
            tau_vector = tf.lin_space(start=tau_min, stop=tau_max, num=num_atoms)
            
            # reshape tau to matrix for fast loss calculation
            tau_matrix = tf.tile(tau_vector, [num_atoms])
            self.tau_matrix = tf.reshape(tau_matrix, shape=[num_atoms, num_atoms])
            
            # main module
            out = full_module(self.input_states, convs, fully_connected,
                              num_outputs=num_actions*num_atoms, activation_fn=activation_fn)
            self.atoms = tf.reshape(out, shape=[-1, num_actions, num_atoms])
            self.q_values = tf.reduce_mean(self.atoms, axis=2)

            ############################## Optimization procedure #############################

            # convert input actions to indices for atoms and q-values selection
            self.input_actions = tf.placeholder(dtype=tf.int32, shape=[None])
            indices_range = tf.range(tf.shape(self.input_actions)[0])
            action_indices = tf.stack([indices_range, self.input_actions], axis=1)

            # select q-values for input actions
            self.q_values_selected = tf.gather_nd(self.q_values, action_indices)
            self.atoms_selected = tf.gather_nd(self.atoms, action_indices)
            
            # select best actions (according to q-values)
            self.q_argmax = tf.argmax(self.q_values, axis=1)
            
            # reshape chosen atoms to matrix for fast loss calculation
            atoms_matrix = tf.tile(self.atoms_selected, [1, num_atoms])
            self.atoms_matrix = tf.reshape(atoms_matrix, shape=[-1, num_atoms, num_atoms])
            
            # reshape target atoms to matrix for fast loss calculation
            self.atoms_targets = tf.placeholder(dtype=tf.float32, shape=[None, num_atoms])
            targets_matrix = tf.tile(self.atoms_targets, [1, num_atoms])
            targets_matrix = tf.reshape(targets_matrix, shape=[-1, num_atoms, num_atoms])
            self.targets_matrix = tf.transpose(targets_matrix, perm=[0, 2, 1])
            
            # define loss function and update rule
            atoms_diff = self.targets_matrix - self.atoms_matrix
            delta_atoms_diff = tf.where(atoms_diff<0, tf.ones_like(atoms_diff), tf.zeros_like(atoms_diff))
            huber_weights = tf.abs(self.tau_matrix - delta_atoms_diff) / num_atoms
            self.loss = self.huber_loss(
                self.targets_matrix, 
                self.atoms_matrix,
                huber_weights,
                kappa)
            self.loss2 = tf.losses.huber_loss(self.targets_matrix, self.atoms_matrix, weights=huber_weights,
                                             delta=kappa, reduction=tf.losses.Reduction.SUM)
            self.update_model = optimizer.minimize(self.loss)

    def get_q_values_s(self, sess, states):
        feed_dict = {self.input_states:states}
        q_values = sess.run(self.q_values, feed_dict)
        return q_values
    
    def get_q_values_sa(self, sess, states, actions):
        feed_dict = {self.input_states:states, self.input_actions:actions}
        q_values_selected = sess.run(self.q_values_selected, feed_dict)
        return q_values_selected
    
    def get_q_argmax(self, sess, states):
        feed_dict = {self.input_states:states}
        q_argmax = sess.run(self.q_argmax, feed_dict)
        return q_argmax
    
    def get_atoms_s(self, sess, states):
        feed_dict = {self.input_states:states}
        atoms = sess.run(self.atoms, feed_dict)
        return probs
    
    def get_atoms_sa(self, sess, states, actions):
        feed_dict = {self.input_states:states, self.input_actions:actions}
        atoms_selected = sess.run(self.atoms_selected, feed_dict)
        return atoms_selected

    def update(self, sess, states, actions, atoms_targets):

        feed_dict = {self.input_states:states,
                     self.input_actions:actions,
                     self.atoms_targets:atoms_targets}
        loss1, loss2,  _ = sess.run([self.loss, self.loss2, self.update_model], feed_dict)
        #print (loss1, loss2)
        
    def huber_loss(self, source, target, weights, kappa=1.0):
        err = tf.subtract(source, target)
        loss = tf.where(
            tf.abs(err) < kappa,
            0.5 * tf.square(err),
            kappa * (tf.abs(err) - 0.5 * kappa))
        return tf.reduce_sum(tf.multiply(loss, weights))

####################################################################################################
#################################### Categorical Deep Q-Network ####################################
####################################################################################################

class CategoricalDeepQNetwork:
    
    def __init__(self, num_actions, state_shape=[8, 8, 5],
                 convs=[[32, 4, 2], [64, 2, 1]],
                 fully_connected=[128], 
                 num_atoms=21, v=(-10, 10),
                 activation_fn=tf.nn.relu,
                 optimizer=tf.train.AdamOptimizer(2.5e-4, epsilon=0.01/32),
                 scope="cat_dqn", reuse=False):
        
        with tf.variable_scope(scope, reuse=reuse):

            ########################### Neural network architecture ###########################

            input_shape = [None] + state_shape
            self.input_states = tf.placeholder(dtype=tf.float32, shape=input_shape)
        
            # distribution parameters
            self.num_atoms = num_atoms
            self.v_min, self.v_max = v
            self.delta_z = (self.v_max - self.v_min) / (self.num_atoms - 1)
            self.z = np.linspace(start=self.v_min, stop=self.v_max, num=num_atoms)

            # main module
            out = full_module(self.input_states, convs, fully_connected,
                              num_outputs=num_actions*num_atoms, activation_fn=activation_fn)
                      
            self.logits = tf.reshape(out, shape=[-1, num_actions, num_atoms])
            self.probs = tf.nn.softmax(self.logits, axis=2)
            self.q_values = tf.reduce_sum(tf.multiply(self.probs, self.z), axis=2)

            ############################## Optimization procedure #############################

            # convert input actions to indices for probs and q-values selection
            self.input_actions = tf.placeholder(dtype=tf.int32, shape=[None])
            indices_range = tf.range(tf.shape(self.input_actions)[0])
            action_indices = tf.stack([indices_range, self.input_actions], axis=1)

            # select q-values and probs for input actions
            self.q_values_selected = tf.gather_nd(self.q_values, action_indices)
            self.probs_selected = tf.gather_nd(self.probs, action_indices)
            
            # select best actions (according to q-values)
            self.q_argmax = tf.argmax(self.q_values, axis=1)

            # define loss function and update rule
            self.probs_targets = tf.placeholder(dtype=tf.float32, shape=[None, self.num_atoms])
            self.loss = -tf.reduce_sum(self.probs_targets * tf.log(self.probs_selected+1e-6))
            self.update_model = optimizer.minimize(self.loss)
    
    def get_q_values_s(self, sess, states):
        feed_dict = {self.input_states:states}
        q_values = sess.run(self.q_values, feed_dict)
        return q_values
    
    def get_q_values_sa(self, sess, states, actions):
        feed_dict = {self.input_states:states, self.input_actions:actions}
        q_values_selected = sess.run(self.q_values_selected, feed_dict)
        return q_values_selected
    
    def get_q_argmax(self, sess, states):
        feed_dict = {self.input_states:states}
        q_argmax = sess.run(self.q_argmax, feed_dict)
        return q_argmax
    
    def get_probs_s(self, sess, states):
        feed_dict = {self.input_states:states}
        probs = sess.run(self.probs, feed_dict)
        return probs
    
    def get_probs_sa(self, sess, states, actions):
        feed_dict = {self.input_states:states, self.input_actions:actions}
        probs_selected = sess.run(self.probs_selected, feed_dict)
        return probs_selected
    
    def update(self, sess, states, actions, probs_targets):

        feed_dict = {self.input_states:states,
                     self.input_actions:actions,
                     self.probs_targets:probs_targets}
        sess.run(self.update_model, feed_dict)
  
    def cat_proj(self, sess, states, actions, rewards, done, gamma=0.99):
        """
        Categorical algorithm from https://arxiv.org/abs/1707.06887
        """
    
        atoms_targets = rewards[:,None] + gamma * self.z * (1 - done[:,None])
        tz = np.clip(atoms_targets, self.v_min, self.v_max)
        tz_z = tz[:, None, :] - self.z[None, :, None]
        tz_z = np.clip((1.0 - (np.abs(tz_z) / self.delta_z)), 0, 1)
        
        probs = self.get_probs_sa(sess, states, actions)
        probs_targets = np.einsum('bij,bj->bi', tz_z, probs)

        return probs_targets 