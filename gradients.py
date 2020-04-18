import os
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'

import tensorflow as tf
import gym
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense, Input
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.activations import relu, linear, softmax
import tensorflow.keras.backend as kb  # not used?
import numpy as np
import multiprocessing as mp
import time
import os

tf.keras.backend.set_floatx('float64')
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

# ENVIRONMENT = 'LunarLander-v2'
ENVIRONMENT = 'CartPole-v1'

class Agent:

  def __init__(
      self,
      num_actions,
      num_states,
      learning_rate = 5e-4,
      epsilon       = 0.5,
      epsilon_min   = 0.01,
      epsilon_decay = 0.996):

    kb.clear_session()  # maybe not needed

    self.num_actions   = num_actions
    self.num_states    = num_states
    self.learning_rate = learning_rate
    self.epsilon       = epsilon
    self.epsilon_min   = epsilon_min
    self.epsilon_decay = epsilon_decay
    self.optimizer     = Adam()

    self.policy = Sequential()
    self.policy.add(Dense(200,
                          input_dim  = num_states,
                          activation = relu)
                   )
    self.policy.add(Dense(100, activation = relu))
    self.policy.add(Dense(num_actions, activation = softmax))

    self.value = Sequential()
    self.value.add(Dense(200,
                         input_dim  = num_states,
                         activation = relu)
                  )
    self.value.add(Dense(100, activation = relu))
    self.value.add(Dense(1, activation = linear))

    ins    = Input(shape = (num_states,))
    l1     = Dense(300,          activation = relu   )(ins)
    l2     = Dense(200,          activation = relu   )(l1)
    value  = Dense(1,            activation = linear )(l2)
    policy = Dense(num_actions,  activation = softmax)(l2)
    
    self.combined_model = Model(inputs = ins, outputs = [value, policy])

  def act(self, state):
    
    if np.random.rand() <= self.epsilon:
      action = np.random.randint(self.num_actions)
    else:
      # acts = self.combined_model.predict(state)
      # action = np.argmax(acts[1][0])
      acts = self.policy.predict(state)
      action = np.argmax(acts[0])

    if self.epsilon > self.epsilon_min:
      self.epsilon = self.epsilon * self.epsilon_decay

    return action

  def advantage(self, cum_reward, state):
    return cum_reward - self.value(tf.convert_to_tensor(state))

  def policy_gradients(self, state, action, cum_reward):
    with tf.GradientTape() as tape:
      tape.watch(self.policy.trainable_weights)

      advantage = self.advantage(cum_reward, state)
      acts = self.policy(tf.convert_to_tensor(state))
      policy = acts[0, action]
      loss = - tf.math.log(policy) * tf.stop_gradient(advantage)

      return tape.gradient(loss, self.policy.trainable_weights)

  def value_gradients(self, state, cum_reward):
    with tf.GradientTape() as tape:
      tape.watch(self.value.trainable_weights)

      advantage = self.advantage(cum_reward, state)
      loss = tf.math.pow(advantage, 2)

      return tape.gradient(loss, self.value.trainable_weights)

  def combined_gradients(self, state, action, cum_reward):
    with tf.GradientTape() as tape:
      tape.watch(self.combined_model.trainable_weights)

      advantage = self.advantage(cum_reward, state)
      acts = self.policy(tf.convert_to_tensor(state))
      policy = acts[0, action]
      p_loss = - tf.math.log(policy) * tf.stop_gradient(advantage)
      v_loss =   tf.math.pow(advantage, 2)

      return 0.5 * p_loss + 0.5 * v_loss


  # Should only be used on the global agent.
  # .assign_sub() subtracts the gradient (dws) from the weights (ws)
  def apply_policy_gradients(self, gradients):
    self.optimizer.apply_gradients(zip(gradients, self.policy.trainable_weights))

  def apply_value_gradients(self, gradients):
    self.optimizer.apply_gradients(zip(gradients, self.value.trainable_weights))

  def apply_combined_gradients(self, gradients):
    self.optimizer.apply_gradients(
        zip(gradients, self.combined_model.trainable_weights)
    )


def worker_main(id, gradient_queue, exit_queue, sync_connection):

  env = gym.make(ENVIRONMENT)
  env._max_episode_steps = 2000
  print("Agent %d made environment" % id)
  # env.seed(0)
  num_states = env.observation_space.shape[0]

  T = 0  # TODO global, shared between all processes
  T_max = 1e5
  t_max = 20
  max_episode_length = 2000
  gamma = 0.99

  agent = Agent(
      env.action_space.n,
      num_states,
      epsilon = 1.0
  )
  print("Agent %d made agent" % id)

  terminated = True
  t = 1
  while T <= T_max:

    # Request weights from global agent. 1 is a dummy
    sync_connection.send(1)

    # Reset local gradients
    policy_gradients = \
        [tf.zeros_like(tw) for tw in agent.policy.trainable_weights]
    value_gradients  = \
        [tf.zeros_like(tw) for tw in agent.value.trainable_weights]
    # combined_gradients = \
        # [tf.zeros_like(tw) for tw in agent.combined_model.trainable_weights]

    # Synchronise local and global parameters
    weights_pair = sync_connection.recv()
    agent.value.set_weights(weights_pair[0])
    agent.policy.set_weights(weights_pair[1])
    # weights = sync_connection.recv()
    # agent.combined_model.set_weights(weights)

    state_buffer  = []
    action_buffer = []
    reward_buffer = []

    t_start = t
    if terminated:
      # env.seed(0)  # DEBUG TEMP
      state = env.reset()
      state = np.reshape(state, (1, num_states))  # TODO set correct input shape to agent networks
      score = 0
      terminated = False

    while (not terminated) and (t - t_start < t_max):

      if id == 0:
        env.render(mode = 'close')
      action = agent.act(state)
      next_state, reward, terminated, _ = env.step(action)
      # ATTEMPT AT FIX for cartpole only(?)
      if terminated:
        reward = -1

      if __debug__ and id == 0:
        print(agent.policy.predict(state))

      state_buffer.append(state)
      action_buffer.append(action)
      reward_buffer.append(reward)
      score = score + reward
      
      state = np.reshape(next_state, (1, num_states))
      t = t + 1
      T = T + 1

    if terminated:
      print("Agent %d got score %f" % (id, score))
      if __debug__:
        print("Agent %d epsilon is %f" % (id, agent.epsilon))
    cum_reward = 0 if terminated else agent.value(tf.convert_to_tensor(state))
    for i in reversed(range(t - t_start)):
      cum_reward = reward_buffer[i] + gamma * cum_reward

      # Accumulate gradients
      policy_gradients = add_gradients(
          policy_gradients,
          agent.policy_gradients(
              state_buffer[i],
              action_buffer[i],
              cum_reward
          )
      )
      value_gradients = add_gradients(
          value_gradients,
          agent.value_gradients(
              state_buffer[i],
              cum_reward
          )
      )
      # combined_gradients = add_gradients(
          # combined_gradients,
          # agent.combined_gradients(
              # state_buffer[i],
              # action_buffer[i],
              # cum_reward
          # )
      # )

    gradient_queue.put((value_gradients, policy_gradients))
    # gradient_queue.put(combined_gradients)

  exit_queue.put(id)
  print("Agent %d quit" % id)


def global_main(num_agents, gradient_queue, exit_queue, sync_connections):

  env = gym.make(ENVIRONMENT)
  print("Globl agent made environment")
  # env.seed(0)


  T = 0  # TODO global, shared between all processes
  T_max = 1
  t_max = 1
  gamma = 0.99

  global_agent = Agent(
      env.action_space.n,
      env.observation_space.shape[0]
  )
  print("Global agent made agent")

  print("\n=======================================")
  print("   environment action space.n = ", env.action_space.n)
  print("   environment observation space.shape = ", env.observation_space.shape)
  print("=======================================\n")

  # Array keeping track of which local agents have finished all work
  has_exited = [False for _ in range(num_agents)]
  while not all(has_exited):

    for i in range(num_agents):
      if sync_connections[i].poll():
        _ = sync_connections[i].recv()

        value_weights  = global_agent.value.get_weights()
        policy_weights = global_agent.policy.get_weights()
        sync_connections[i].send((value_weights, policy_weights))
        # weights = global_agent.combined_model.get_weights()
        # sync_connections[i].send(weights)

    # Queue.empty() is unreliable according to docs, may cause bugs
    while not gradient_queue.empty():
      grads = gradient_queue.get()
      global_agent.apply_value_gradients(grads[0])
      global_agent.apply_policy_gradients(grads[1])
      # global_agent.apply_combined_gradients(grads)

    while not exit_queue.empty():
      exited_id = exit_queue.get()
      has_exited[exited_id] = True
      print("Global agent found agent %d has exited" % exited_id)

    time.sleep(0.2)

  print("Global agent is done")

def main():

  num_agents = 4
  
  gradient_queue =  mp.Queue()
  exit_queue     =  mp.Queue()
  sync_pipes     = [mp.Pipe() for _ in range(num_agents)]
  connections_0  = [c[0] for c in sync_pipes]
  connections_1  = [c[1] for c in sync_pipes]

  global_process = mp.Process(
      target = global_main,
      args   = (num_agents, gradient_queue, exit_queue, connections_0)
  )
  global_process.start()
  print("Started global process")

  processes = []
  for id in range(num_agents):
    proc = mp.Process(
        target = worker_main,
        args   = (id, gradient_queue, exit_queue, connections_1[id])
    )
    processes.append(proc)
    proc.start()
    print("Started process %d" % id)

  for id in range(num_agents):
    processes[id].join()
    print("Joined process %d" % id)

  global_process.join()
  print("Joined global process")



# ==== utils ====

# Elementwise addition of list of tensors
def add_gradients(xs, ys):
  return list( map(lambda pair: pair[0] + pair[1], zip(xs, ys)) )

if __name__ == '__main__':

  print("Starting")
  main()
  print("Done")

