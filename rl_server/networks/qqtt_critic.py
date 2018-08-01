import numpy as np
import tensorflow as tf
import t3f
from tensorflow.python import keras
from tensorflow.python.keras.layers import Layer, Lambda, Input, Reshape


def tf_dynamic_stack(x, y):
    shape = x.get_shape().as_list()
    tmp = tf.reshape(x, (-1, ))
    tile = tf.tile(tmp, tf.shape(y)[0:1])
    return tf.reshape(tile, [tf.shape(y)[0]] + shape)


def gather_nd_partial(tt, indices, full=True):
    tt_elements = tf.ones(tf.shape(indices)[0], dtype=tt.dtype)
    tt_elements = tf.reshape(tt_elements, (-1, 1, 1))
    ind_shape = indices.get_shape().as_list()

    for core_idx in range(ind_shape[1]):
        curr_core = tt.tt_cores[core_idx]
        curr_core = tf.transpose(curr_core, (1, 0, 2))
        core_slices = tf.gather(curr_core, indices[:, core_idx])
        tt_elements = tf.matmul(tt_elements, core_slices)

    remaining_cores = tt.tt_cores[ind_shape[1]:]
    new_core = tf.einsum('ijk,klm->ijlm', tt_elements, remaining_cores[0])
    cores = [new_core]

    for i in range(1, len(remaining_cores)):
        cores.append(tf_dynamic_stack(remaining_cores[i]), indices)
    slice_batch = t3f.TensorTrainBatch(cores)
    if full:
        return t3f.full(slice_batch)
    else:
        return slice_batch


class QTLayer(Layer):

    def __init__(self, state_shape, tt_shape, partition_size=64, tt_rank=8, **kwargs):
        self.state_shape = state_shape
        self.tt_shape = tt_shape
        self.part_size = partition_size
        self.tt_rank = tt_rank
        super(QTLayer, self).__init__(**kwargs)

    def build(self, input_shape):

        q_init = t3f.random_tensor(shape=self.tt_shape, tt_rank=self.tt_rank, stddev=1e-3)
        q_init = t3f.cast(q_init, dtype=tf.float32)
        self.Q = t3f.get_variable('Q', initializer=q_init)
        self._trainable_weights = list(self.Q.tt_cores)
        self.built = True

    def call(self, x, mode='q_sa'):
        print(x)
        if mode == 'q_sa':
            states, actions = x
            reshaped_s = tf.reshape(states, (-1, np.prod(self.state_shape)))
            reshaped_a = tf.reshape(actions, (-1, 1))
            input_s_and_a = tf.concat([reshaped_s, reshaped_a], axis=1)
            q_values_sa = t3f.gather_nd(self.Q, input_s_and_a)
            return q_values_sa

        elif mode == 'q_s':
            states = x
            reshaped_s = tf.reshape(states, (-1, np.prod(self.state_shape)))
            q_values_s = gather_nd_partial(self.Q, reshaped_s)
            return q_values_s

    def compute_output_shape(self, input_shape):
        return input_shape


class QQTTCriticNetwork:

    def __init__(self, state_shape, action_size,
                 state_bound, action_bound,
                 tt_rank, partition_size=64, scope=None):
        """
        state_bound = (state_low, state_high)
        action_bound = (action_low, action_high)
        """
        self.state_shape = state_shape
        self.action_size = action_size
        self.state_bound = state_bound
        self.action_bound = action_bound
        self.tt_rank = tt_rank
        self.part_size = partition_size
        self.scope = scope or 'QQttCriticNetwork'
        self.input_shape = (partition_size, ) * (state_shape[0] + 1)

        self.state_low = tf.constant(self.state_bound[0])
        self.state_high = tf.constant(self.state_bound[1])
        self.state_step = (self.state_high - self.state_low) / (self.part_size - 1)

        self.action_low = tf.constant(self.action_bound[0])
        self.action_high = tf.constant(self.action_bound[1])
        self.action_step = (self.action_high - self.action_low) / (self.part_size - 1)

        self.model_critic, self.model_actor = self.build_models()

    def build_models(self):

        input_state = Input(shape=self.state_shape, name='state_input')
        input_action = Input(shape=(self.action_size, ), name='action_input')
        model_inputs = [input_state, input_action]

        input_size = self.get_input_size(self.state_shape)
        input_reshaped = Reshape((input_size, ))(input_state)

        with tf.variable_scope(self.scope):

            states = Lambda(lambda x: self.discretize_states(x))(input_reshaped)
            actions = Lambda(lambda x: self.discretize_actions(x))(input_action)

            tt_shape = (self.part_size, ) * (self.state_shape[1] + 1)

            qt_layer = QTLayer(self.state_shape, tt_shape, self.part_size, self.tt_rank)

            q_values_sa = qt_layer([states, actions], mode='q_sa')
            q_values_s = qt_layer(states, mode='q_s')

            best_actions = Lambda(lambda x: tf.argmax(x, axis=1))(q_values_s)
            best_actions_cont = Lambda(lambda x: self.discretize_actions_back(x))(best_actions)

            model_critic = keras.models.Model(inputs=[input_state, input_action], outputs=q_values_sa)
            model_actor = keras.models.Model(inputs=[input_state], outputs=best_actions_cont)
            model_critic.summary()
            model_actor.summary()

        return model_critic, model_actor

    def get_input_size(self, shape):
        if len(shape) == 1:
            return shape[0]
        elif len(shape) == 2:
            return shape[0] * shape[1]

    def discretize_states(self, states):
        disc_states = tf.round((states - self.state_low) / self.state_step)
        disc_states = tf.cast(disc_states, dtype=tf.int32)
        return disc_states

    def discretize_actions(self, actions):
        disc_actions = tf.round((actions - self.action_low) / self.action_step)
        disc_actions = tf.cast(disc_actions, dtype=tf.int32)
        return disc_actions

    def discretize_actions_back(self, disc_actions):
        disc_actions = tf.cast(disc_actions, dtype=tf.float32)
        actions = self.action_low + disc_actions * self.action_step
        return actions

    def __call__(self, inputs):

        if len(inputs) == 2:
            state_input = inputs[0][0]
            action_input = inputs[1]
            q_values_sa = self.model_critic([state_input, action_input])
            return q_values_sa
        if len(inputs) == 1:
            state_input = inputs[0][0]
            best_actions = self.model_actor([state_input])
            return best_actions

    def variables(self):
        return self.model_critic.trainable_weights

    def copy(self, scope=None):
        """copy network architecture"""
        scope = scope or self.scope + "_copy"
        with tf.variable_scope(scope):
            return QQTTCriticNetwork(state_shape=self.state_shape,
                                     action_size=self.action_size,
                                     state_bound=self.state_bound,
                                     action_bound=self.action_bound,
                                     tt_rank=self.tt_rank,
                                     partition_size=self.part_size,
                                     scope=scope)

    def get_info(self):
        info = {}
        info['architecture'] = 'qqtt'
        info['tt_rank'] = self.tt_rank
        info['partition_size'] = self.part_size
        return info
