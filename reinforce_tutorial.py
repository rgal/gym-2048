# Encoding network
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from absl import logging
import gin
import tensorflow as tf

from tf_agents.networks import network
from tf_agents.networks import utils
from tf_agents.utils import nest_utils

from tensorflow.python.util import nest  # pylint:disable=g-direct-tensorflow-import  # TF internal

CONV_TYPE_2D = '2d'
CONV_TYPE_1D = '1d'


def _copy_layer(layer):
  """Create a copy of a Keras layer with identical parameters.

  The new layer will not share weights with the old one.

  Args:
    layer: An instance of `tf.keras.layers.Layer`.

  Returns:
    A new keras layer.

  Raises:
    TypeError: If `layer` is not a keras layer.
    ValueError: If `layer` cannot be correctly cloned.
  """
  if not isinstance(layer, tf.keras.layers.Layer):
    raise TypeError('layer is not a keras layer: %s' % str(layer))

  # pylint:disable=unidiomatic-typecheck
  if type(layer) == tf.compat.v1.keras.layers.DenseFeatures:
    raise ValueError('DenseFeatures V1 is not supported. '
                     'Use tf.compat.v2.keras.layers.DenseFeatures instead.')
  if layer.built:
    logging.warn(
        'Beware: Copying a layer that has already been built: \'%s\'.  '
        'This can lead to subtle bugs because the original layer\'s weights '
        'will not be used in the copy.', layer.name)
  # Get a fresh copy so we don't modify an incoming layer in place.  Weights
  # will not be shared.
  return type(layer).from_config(layer.get_config())


@gin.configurable
class EncodingNetwork(network.Network):
  """Feed Forward network with CNN and FNN layers."""

  def __init__(self,
               input_tensor_spec,
               preprocessing_layers=None,
               preprocessing_combiner=None,
               conv_layer_params=None,
               fc_layer_params=None,
               dropout_layer_params=None,
               activation_fn=tf.keras.activations.relu,
               kernel_initializer=None,
               batch_squash=True,
               dtype=tf.float32,
               name='EncodingNetwork',
               conv_type=CONV_TYPE_2D):
    """Creates an instance of `EncodingNetwork`.

    Network supports calls with shape outer_rank + input_tensor_spec.shape. Note
    outer_rank must be at least 1.

    For example an input tensor spec with shape `(2, 3)` will require
    inputs with at least a batch size, the input shape is `(?, 2, 3)`.

    Input preprocessing is possible via `preprocessing_layers` and
    `preprocessing_combiner` Layers.  If the `preprocessing_layers` nest is
    shallower than `input_tensor_spec`, then the layers will get the subnests.
    For example, if:

    ```python
    input_tensor_spec = ([TensorSpec(3)] * 2, [TensorSpec(3)] * 5)
    preprocessing_layers = (Layer1(), Layer2())
    ```

    then preprocessing will call:

    ```python
    preprocessed = [preprocessing_layers[0](observations[0]),
                    preprocessing_layers[1](obsrevations[1])]
    ```

    However if

    ```python
    preprocessing_layers = ([Layer1() for _ in range(2)],
                            [Layer2() for _ in range(5)])
    ```

    then preprocessing will call:
    ```python
    preprocessed = [
      layer(obs) for layer, obs in zip(flatten(preprocessing_layers),
                                       flatten(observations))
    ]
    ```

    **NOTE** `preprocessing_layers` and `preprocessing_combiner` are not allowed
    to have already been built.  This ensures calls to `network.copy()` in the
    future always have an unbuilt, fresh set of parameters.  Furtheremore,
    a shallow copy of the layers is always created by the Network, so the
    layer objects passed to the network are never modified.  For more details
    of the semantics of `copy`, see the docstring of
    `tf_agents.networks.Network.copy`.

    Args:
      input_tensor_spec: A nest of `tensor_spec.TensorSpec` representing the
        input observations.
      preprocessing_layers: (Optional.) A nest of `tf.keras.layers.Layer`
        representing preprocessing for the different observations. All of these
        layers must not be already built.
      preprocessing_combiner: (Optional.) A keras layer that takes a flat list
        of tensors and combines them.  Good options include
        `tf.keras.layers.Add` and `tf.keras.layers.Concatenate(axis=-1)`. This
        layer must not be already built.
      conv_layer_params: Optional list of convolution layers parameters, where
        each item is either a length-three tuple indicating
        `(filters, kernel_size, stride)` or a length-four tuple indicating
        `(filters, kernel_size, stride, dilation_rate)`.
      fc_layer_params: Optional list of fully_connected parameters, where each
        item is the number of units in the layer.
      dropout_layer_params: Optional list of dropout layer parameters, each item
        is the fraction of input units to drop or a dictionary of parameters
        according to the keras.Dropout documentation. The additional parameter
        `permanent', if set to True, allows to apply dropout at inference for
        approximated Bayesian inference. The dropout layers are interleaved with
        the fully connected layers; there is a dropout layer after each fully
        connected layer, except if the entry in the list is None. This list must
        have the same length of fc_layer_params, or be None.
      activation_fn: Activation function, e.g. tf.keras.activations.relu,.
      kernel_initializer: Initializer to use for the kernels of the conv and
        dense layers. If none is provided a default variance_scaling_initializer
      batch_squash: If True the outer_ranks of the observation are squashed into
        the batch dimension. This allow encoding networks to be used with
        observations with shape [BxTx...].
      dtype: The dtype to use by the convolution and fully connected layers.
      name: A string representing name of the network.
      conv_type: string, '1d' or '2d'. Convolution layers will be 1d or 2D
        respectively

    Raises:
      ValueError: If any of `preprocessing_layers` is already built.
      ValueError: If `preprocessing_combiner` is already built.
      ValueError: If the number of dropout layer parameters does not match the
        number of fully connected layer parameters.
      ValueError: If conv_layer_params tuples do not have 3 or 4 elements each.
    """
    if preprocessing_layers is None:
      flat_preprocessing_layers = None
    else:
      flat_preprocessing_layers = [
          _copy_layer(layer) for layer in tf.nest.flatten(preprocessing_layers)
      ]
      # Assert shallow structure is the same. This verifies preprocessing
      # layers can be applied on expected input nests.
      input_nest = input_tensor_spec
      # Given the flatten on preprocessing_layers above we need to make sure
      # input_tensor_spec is a sequence for the shallow_structure check below
      # to work.
      if not nest.is_sequence(input_tensor_spec):
        input_nest = [input_tensor_spec]
      nest.assert_shallow_structure(
          preprocessing_layers, input_nest, check_types=False)

    if (len(tf.nest.flatten(input_tensor_spec)) > 1 and
        preprocessing_combiner is None):
      raise ValueError(
          'preprocessing_combiner layer is required when more than 1 '
          'input_tensor_spec is provided.')

    if preprocessing_combiner is not None:
      preprocessing_combiner = _copy_layer(preprocessing_combiner)

    if not kernel_initializer:
      kernel_initializer = tf.compat.v1.variance_scaling_initializer(
          scale=2.0, mode='fan_in', distribution='truncated_normal')

    layers = []

    if conv_layer_params:
      if conv_type == '2d':
        conv_layer_type = tf.keras.layers.Conv2D
      elif conv_type == '1d':
        conv_layer_type = tf.keras.layers.Conv1D
      else:
        raise ValueError('unsupported conv type of %s. Use 1d or 2d' % (
            conv_type))

      for config in conv_layer_params:
        if len(config) == 4:
          (filters, kernel_size, strides, dilation_rate) = config
        elif len(config) == 3:
          (filters, kernel_size, strides) = config
          dilation_rate = (1, 1) if conv_type == '2d' else (1,)
        else:
          raise ValueError(
              'only 3 or 4 elements permitted in conv_layer_params tuples')
        layers.append(
            conv_layer_type(
                filters=filters,
                kernel_size=kernel_size,
                strides=strides,
                dilation_rate=dilation_rate,
                padding='same',
                #activation=activation_fn,
                kernel_initializer=kernel_initializer,
                dtype=dtype,
                name='%s/conv%s' % (name, conv_type)))
        layers.append(tf.keras.layers.BatchNormalization())
        layers.append(tf.keras.layers.Activation('relu'))

    layers.append(tf.keras.layers.Flatten())

    if fc_layer_params:
      if dropout_layer_params is None:
        dropout_layer_params = [None] * len(fc_layer_params)
      else:
        if len(dropout_layer_params) != len(fc_layer_params):
          raise ValueError('Dropout and fully connected layer parameter lists'
                           'have different lengths (%d vs. %d.)' %
                           (len(dropout_layer_params), len(fc_layer_params)))
      for num_units, dropout_params in zip(fc_layer_params,
                                           dropout_layer_params):
        layers.append(
            tf.keras.layers.Dense(
                num_units,
                activation=activation_fn,
                kernel_initializer=kernel_initializer,
                dtype=dtype,
                name='%s/dense' % name))
        if not isinstance(dropout_params, dict):
          dropout_params = {'rate': dropout_params} if dropout_params else None

        if dropout_params is not None:
          layers.append(utils.maybe_permanent_dropout(**dropout_params))

    super(EncodingNetwork, self).__init__(
        input_tensor_spec=input_tensor_spec, state_spec=(), name=name)

    # Pull out the nest structure of the preprocessing layers. This avoids
    # saving the original kwarg layers as a class attribute which Keras would
    # then track.
    self._preprocessing_nest = tf.nest.map_structure(lambda l: None,
                                                     preprocessing_layers)
    self._flat_preprocessing_layers = flat_preprocessing_layers
    self._preprocessing_combiner = preprocessing_combiner
    self._postprocessing_layers = layers
    self._batch_squash = batch_squash

  def call(self, observation, step_type=None, network_state=()):
    del step_type  # unused.

    if self._batch_squash:
      outer_rank = nest_utils.get_outer_rank(
          observation, self.input_tensor_spec)
      batch_squash = utils.BatchSquash(outer_rank)
      observation = tf.nest.map_structure(batch_squash.flatten, observation)

    if self._flat_preprocessing_layers is None:
      processed = observation
    else:
      processed = []
      for obs, layer in zip(
          nest.flatten_up_to(
              self._preprocessing_nest, observation, check_types=False),
          self._flat_preprocessing_layers):
        processed.append(layer(obs))
      if len(processed) == 1 and self._preprocessing_combiner is None:
        # If only one observation is passed and the preprocessing_combiner
        # is unspecified, use the preprocessed version of this observation.
        processed = processed[0]

    states = processed

    if self._preprocessing_combiner is not None:
      states = self._preprocessing_combiner(states)

    for layer in self._postprocessing_layers:
      states = layer(states)

    if self._batch_squash:
      states = tf.nest.map_structure(batch_squash.unflatten, states)

    return states, network_state

#Actor distribution network
#from __future__ import absolute_import
#from __future__ import division
#from __future__ import print_function

import gin
import numpy as np
import tensorflow as tf

from tf_agents.networks import categorical_projection_network
#from tf_agents.networks import encoding_network
from tf_agents.networks import network
from tf_agents.networks import normal_projection_network
from tf_agents.specs import tensor_spec
from tf_agents.utils import nest_utils


def _categorical_projection_net(action_spec, logits_init_output_factor=0.1):
  return categorical_projection_network.CategoricalProjectionNetwork(
      action_spec, logits_init_output_factor=logits_init_output_factor)


def _normal_projection_net(action_spec,
                           init_action_stddev=0.35,
                           init_means_output_factor=0.1):
  std_bias_initializer_value = np.log(np.exp(init_action_stddev) - 1)

  return normal_projection_network.NormalProjectionNetwork(
      action_spec,
      init_means_output_factor=init_means_output_factor,
      std_bias_initializer_value=std_bias_initializer_value,
      scale_distribution=False)


@gin.configurable
class ActorDistributionNetwork(network.DistributionNetwork):
  """Creates an actor producing either Normal or Categorical distribution.

  Note: By default, this network uses `NormalProjectionNetwork` for continuous
  projection which by default uses `tanh_squash_to_spec` to normalize its
  output. Due to the nature of the `tanh` function, values near the spec bounds
  cannot be returned.
  """

  def __init__(self,
               input_tensor_spec,
               output_tensor_spec,
               preprocessing_layers=None,
               preprocessing_combiner=None,
               conv_layer_params=None,
               fc_layer_params=(200, 100),
               dropout_layer_params=None,
               activation_fn=tf.keras.activations.relu,
               kernel_initializer=None,
               batch_squash=True,
               dtype=tf.float32,
               discrete_projection_net=_categorical_projection_net,
               continuous_projection_net=_normal_projection_net,
               name='ActorDistributionNetwork'):
    """Creates an instance of `ActorDistributionNetwork`.

    Args:
      input_tensor_spec: A nest of `tensor_spec.TensorSpec` representing the
        input.
      output_tensor_spec: A nest of `tensor_spec.BoundedTensorSpec` representing
        the output.
      preprocessing_layers: (Optional.) A nest of `tf.keras.layers.Layer`
        representing preprocessing for the different observations.
        All of these layers must not be already built. For more details see
        the documentation of `networks.EncodingNetwork`.
      preprocessing_combiner: (Optional.) A keras layer that takes a flat list
        of tensors and combines them. Good options include
        `tf.keras.layers.Add` and `tf.keras.layers.Concatenate(axis=-1)`.
        This layer must not be already built. For more details see
        the documentation of `networks.EncodingNetwork`.
      conv_layer_params: Optional list of convolution layers parameters, where
        each item is a length-three tuple indicating (filters, kernel_size,
        stride).
      fc_layer_params: Optional list of fully_connected parameters, where each
        item is the number of units in the layer.
      dropout_layer_params: Optional list of dropout layer parameters, each item
        is the fraction of input units to drop or a dictionary of parameters
        according to the keras.Dropout documentation. The additional parameter
        `permanent', if set to True, allows to apply dropout at inference for
        approximated Bayesian inference. The dropout layers are interleaved with
        the fully connected layers; there is a dropout layer after each fully
        connected layer, except if the entry in the list is None. This list must
        have the same length of fc_layer_params, or be None.
      activation_fn: Activation function, e.g. tf.nn.relu, slim.leaky_relu, ...
      kernel_initializer: Initializer to use for the kernels of the conv and
        dense layers. If none is provided a default glorot_uniform
      batch_squash: If True the outer_ranks of the observation are squashed into
        the batch dimension. This allow encoding networks to be used with
        observations with shape [BxTx...].
      dtype: The dtype to use by the convolution and fully connected layers.
      discrete_projection_net: Callable that generates a discrete projection
        network to be called with some hidden state and the outer_rank of the
        state.
      continuous_projection_net: Callable that generates a continuous projection
        network to be called with some hidden state and the outer_rank of the
        state.
      name: A string representing name of the network.

    Raises:
      ValueError: If `input_tensor_spec` contains more than one observation.
    """

    if not kernel_initializer:
      kernel_initializer = tf.compat.v1.keras.initializers.glorot_uniform()

    encoder = EncodingNetwork(
        input_tensor_spec,
        preprocessing_layers=preprocessing_layers,
        preprocessing_combiner=preprocessing_combiner,
        conv_layer_params=conv_layer_params,
        fc_layer_params=fc_layer_params,
        dropout_layer_params=dropout_layer_params,
        activation_fn=activation_fn,
        kernel_initializer=kernel_initializer,
        batch_squash=batch_squash,
        dtype=dtype)

    def map_proj(spec):
      if tensor_spec.is_discrete(spec):
        return discrete_projection_net(spec)
      else:
        return continuous_projection_net(spec)

    projection_networks = tf.nest.map_structure(map_proj, output_tensor_spec)
    output_spec = tf.nest.map_structure(lambda proj_net: proj_net.output_spec,
                                        projection_networks)

    super(ActorDistributionNetwork, self).__init__(
        input_tensor_spec=input_tensor_spec,
        state_spec=(),
        output_spec=output_spec,
        name=name)

    self._encoder = encoder
    self._projection_networks = projection_networks
    self._output_tensor_spec = output_tensor_spec

  @property
  def output_tensor_spec(self):
    return self._output_tensor_spec

  def call(self, observations, step_type, network_state):
    state, network_state = self._encoder(
        observations, step_type=step_type, network_state=network_state)
    outer_rank = nest_utils.get_outer_rank(observations, self.input_tensor_spec)
    output_actions = tf.nest.map_structure(
        lambda proj_net: proj_net(state, outer_rank), self._projection_networks)
    return output_actions, network_state

# Reinforce agent code
#from __future__ import absolute_import
#from __future__ import division
#from __future__ import print_function

#import matplotlib
#import matplotlib.pyplot as plt

import tensorflow as tf

from tf_agents.agents.reinforce import reinforce_agent
from tf_agents.drivers import dynamic_step_driver
from tf_agents.environments import suite_gym
from tf_agents.environments import tf_py_environment
from tf_agents.eval import metric_utils
from tf_agents.metrics import tf_metrics
#from tf_agents.networks import actor_distribution_network
from tf_agents.replay_buffers import tf_uniform_replay_buffer
from tf_agents.trajectories import trajectory
from tf_agents.utils import common

import gym_2048

tf.compat.v1.enable_v2_behavior()

env_name = '2048-v0'  # @param
num_iterations = 10000  # @param
collect_episodes_per_iteration = 2  # @param
replay_buffer_capacity = 2000  # @param

filters = 64
# filters, size, stride
conv_layer_params = [
  (filters, 3, 1),
  (filters, 3, 1),
  (filters, 3, 1),
  (filters, 3, 1),
  (2, 1, 1),
]
fc_layer_params = []#(1024, 1024)

learning_rate = 1e-4  # @param
log_interval = 25  # @param
num_eval_episodes = 10  # @param
eval_interval = 50  # @param

env = suite_gym.load(env_name)

env.reset()

# Informative
print('Observation Spec:')
print(env.time_step_spec().observation)
print('Action Spec:')
print(env.action_spec())

time_step = env.reset()
print('Time step:')
print(time_step)

action = 1

next_time_step = env.step(action)
print('Next time step:')
print(next_time_step)

# Back to actually doing some learning
train_py_env = suite_gym.load(env_name)
eval_py_env = suite_gym.load(env_name)

train_env = tf_py_environment.TFPyEnvironment(train_py_env)
eval_env = tf_py_environment.TFPyEnvironment(eval_py_env)

actor_net = ActorDistributionNetwork(
    train_env.observation_spec(),
    train_env.action_spec(),
    conv_layer_params=conv_layer_params,
    fc_layer_params=fc_layer_params)

optimizer = tf.compat.v1.train.AdamOptimizer(learning_rate=learning_rate)

train_step_counter = tf.compat.v2.Variable(0)

tf_agent = reinforce_agent.ReinforceAgent(
    train_env.time_step_spec(),
    train_env.action_spec(),
    actor_network=actor_net,
    optimizer=optimizer,
    gamma=0.9,
    normalize_returns=True,
    train_step_counter=train_step_counter)
tf_agent.initialize()

eval_policy = tf_agent.policy
collect_policy = tf_agent.collect_policy

# Please also see the metrics module for standard implementations of different
# metrics.
def compute_avg_return(environment, policy, num_episodes=10):
  total_return = 0.0
  max_return = 0.0
  for _ in range(num_episodes):

    time_step = environment.reset()
    episode_return = 0.0

    while not time_step.is_last():
      action_step = policy.action(time_step)
      time_step = environment.step(action_step.action)
      episode_return += time_step.reward
    total_return += episode_return

    max_return = max(episode_return, max_return)

  avg_return = total_return / num_episodes
  return avg_return.numpy()[0], max_return.numpy()[0]

replay_buffer = tf_uniform_replay_buffer.TFUniformReplayBuffer(
    data_spec=tf_agent.collect_data_spec,
    batch_size=train_env.batch_size,
    max_length=replay_buffer_capacity)

# This loop is so common in RL, that we provide standard implementations of
# these. For more details see the drivers module.
def collect_episode(environment, policy, num_episodes):
  episode_counter = 0
  environment.reset()

  while episode_counter < num_episodes:
    time_step = environment.current_time_step()
    action_step = policy.action(time_step)
    next_time_step = environment.step(action_step.action)
    traj = trajectory.from_transition(time_step, action_step, next_time_step)

    # Add trajectory to the replay buffer
    replay_buffer.add_batch(traj)

    if traj.is_boundary():
      episode_counter += 1

# (Optional) Optimize by wrapping some of the code in a graph using TF function.
tf_agent.train = common.function(tf_agent.train)

# Reset the train step
tf_agent.train_step_counter.assign(0)

# Evaluate the agent's policy once before training.
(avg_return, max_return) = compute_avg_return(eval_env, tf_agent.policy, num_eval_episodes)
returns = [avg_return]

for _ in range(num_iterations):

  # Collect a few episodes using collect_policy and save to the replay buffer.
  collect_episode(
      train_env, tf_agent.collect_policy, collect_episodes_per_iteration)

  # Use data from the buffer and update the agent's network.
  experience = replay_buffer.gather_all()
  train_loss = tf_agent.train(experience)
  replay_buffer.clear()

  step = tf_agent.train_step_counter.numpy()

  if step % log_interval == 0:
    print('step = {0}: loss = {1}'.format(step, train_loss.loss))

  if step % eval_interval == 0:
    (avg_return, max_return) = compute_avg_return(eval_env, tf_agent.policy, num_eval_episodes)
    print('step = {0}: Average Return = {1}, Maximum Return = {2}'.format(step, avg_return, max_return))
    returns.append(avg_return)

steps = range(0, num_iterations + 1, eval_interval)
#plt.plot(steps, returns)
#plt.ylabel('Average Return')
#plt.xlabel('Step')
#plt.ylim(top=250)
#plt.savefig('out.png')
