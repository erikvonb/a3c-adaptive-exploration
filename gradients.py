import os
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'

import tensorflow as tf
import gym
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense, Input
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.activations import relu, linear
import tensorflow.keras.backend as kb  # not used?
import numpy as np
import multiprocessing as mp
import time


class Agent:

  def __init__(self, action_space, state_space, learning_rate = 0.5):
    kb.clear_session()  # maybe not needed

    self.action_space = action_space
    self.state_space  = state_space
    self.learning_rate = learning_rate

    self.policy = Sequential()
    self.policy.add(Dense(150,
                          input_dim  = state_space,
                          activation = relu)
                   )
    self.policy.add(Dense(120, activation = relu))
    self.policy.add(Dense(action_space, activation = linear))

    self.value = Sequential()
    self.value.add(Dense(150,
                         input_dim  = state_space,
                         activation = relu)
                  )
    self.value.add(Dense(120, activation = relu))
    self.value.add(Dense(action_space, activation = linear))

    # self.optimizer = Adam(lr = 0.1)

  def act(self, state):
    # TODO epsilon-greedy
    acts = self.policy.predict(state)
    return np.argmax(acts[0])

  def advantage(self, cum_reward, state):
    return cum_reward - self.value(tf.convert_to_tensor(state))

  def policy_gradients(self, state, action, cum_reward):
    with tf.GradientTape() as tape:
      tape.watch(self.policy.trainable_weights)

      advantage = self.advantage(cum_reward, state)
      acts = self.policy(tf.convert_to_tensor(state))
      policy = acts[0, action]
      loss = tf.math.log(policy) * tf.stop_gradient(advantage)

      return tape.gradient(loss, self.policy.trainable_weights)

  def value_gradients(self, state, cum_reward):
    with tf.GradientTape() as tape:
      tape.watch(self.value.trainable_weights)

      advantage = self.advantage(cum_reward, state)
      loss = tf.math.pow(advantage, 2)

      return tape.gradient(loss, self.value.trainable_weights)

  # Should only be used on the global agent.
  # .assign_sub() subtracts the gradient (dws) from the weights (ws)
  def apply_policy_gradients(self, gradients):
    map(
        lambda ws_dws: ws_dws[0].assign_sub(self.learning_rate * ws_dws[1]),
        zip(self.policy.trainable_weights, gradients)
    )

  def apply_value_gradients(self, gradients):
    map(
        lambda ws_dws: ws_dws[0].assign_sub(self.learning_rate * ws_dws[1]),
        zip(self.value.trainable_weights, gradients)
    )


def worker_main(id, gradient_queue, exit_queue):

  env = gym.make('LunarLander-v2')
  print("Agent %d made environment" % id)
  # env.seed(0)

  T = 0  # TODO global, shared between all processes
  T_max = 1
  t_max = 1
  gamma = 0.9

  agent = Agent(
      env.action_space.n,
      env.observation_space.shape[0]
  )
  print("Agent %d made agent" % id)

  t = 1
  while T <= T_max:
    # Reset local gradients
    policy_gradients = \
        [tf.zeros_like(tw) for tw in agent.policy.trainable_weights]
    value_gradients  = \
        [tf.zeros_like(tw) for tw in agent.value.trainable_weights]

    # TODO synchronise local and global parameters

    t_start = t
    state = env.reset()
    state = np.reshape(state, (1, 8))  # TODO set correct input shape to agent networks

    state_buffer  = []
    action_buffer = []
    reward_buffer = []
    done = False
    while (not done) and (t - t_start < t_max):
      # print("\nT = %d, environment step %d" % (T, t - t_start))

      # env.render()
      action = agent.act(state)
      next_state, reward, done, _ = env.step(action)

      state_buffer.append(state)
      action_buffer.append(action)
      reward_buffer.append(reward)
      
      state = np.reshape(next_state, (1, 8))
      t = t + 1
      T = T + 1

    cum_reward = 0 if done else agent.value(tf.convert_to_tensor(state))
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

    gradient_queue.put((value_gradients, policy_gradients))
    print("Process %d put gradients in queue" % id)

  exit_queue.put(id)
  print("Process %d quit" % id)


def global_main(num_agents, gradient_queue, exit_queue):

  env = gym.make('LunarLander-v2')
  print("Globl agent made environment")
  # env.seed(0)

  T = 0  # TODO global, shared between all processes
  T_max = 1
  t_max = 1
  gamma = 0.9

  global_agent = Agent(
      env.action_space.n,
      env.observation_space.shape[0]
  )
  print("Global agent made agent")

  # Array keeping track of which local agents have finished all work
  has_exited = [False for _ in range(num_agents)]
  while not all(has_exited):

    # Queue.empty() is unreliable according to docs, may cause bugs
    while not gradient_queue.empty():
      grads = gradient_queue.get()
      global_agent.apply_value_gradients(grads[0])
      global_agent.apply_policy_gradients(grads[1])
      print("Global agent applied gradients")

    while not exit_queue.empty():
      exited_id = exit_queue.get()
      has_exited[exited_id] = True
      print("Global agent found agent %d has exited" % exited_id)

    time.sleep(0.2)

  print("Global agent is done")

def main():

  num_agents = 2
  
  gradient_queue = mp.Queue()
  exit_queue     = mp.Queue()

  global_process = mp.Process(
      target = global_main,
      args   = (num_agents, gradient_queue, exit_queue)
  )
  global_process.start()
  print("Started global process")

  processes = []
  for id in range(num_agents):
    proc = mp.Process(
        target = worker_main,
        args   = (id, gradient_queue, exit_queue)
    )
    processes.append(proc)
    proc.start()
    print("Started process %d" % id)

  # grads = gradient_queue.get(block = True)
  # print("Main process took gradients from queue, received:")
  # print(grads)
  
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

