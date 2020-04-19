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
      learning_rate = 0.001,
      epsilon       = 0.5,
      epsilon_min   = 0.05,
      epsilon_decay = 0.9):

    kb.clear_session()  # maybe not needed

    self.num_actions   = num_actions
    self.num_states    = num_states
    self.epsilon       = epsilon
    self.epsilon_min   = epsilon_min
    self.epsilon_decay = epsilon_decay
    self.optimizer     = Adam(learning_rate = learning_rate)


    ins    = Input(shape = (num_states,))
    l1     = Dense(100,          activation = relu   )(ins)
    l2     = Dense(100,          activation = relu   )(ins)
    value  = Dense(1,            activation = linear )(l1)
    # NOTE policy output is unnormalised probabilities; use softmax explicitly
    policy = Dense(num_actions,  activation = linear)(l2)
    
    self.combined_model = Model(inputs = ins, outputs = [value, policy])

  # not used?
  def act(self, state):
    
    # if np.random.rand() <= self.epsilon:
    if False:
      action = np.random.randint(self.num_actions)
    else:
      policy = self.combined_model(tf.convert_to_tensor(state))[1]
      policy = tf.nn.softmax(policy)
      # action = np.argmax(acts[1][0])
      action = np.random.choice(self.num_actions, p = policy.numpy()[0])

    if self.epsilon > self.epsilon_min:
      self.epsilon = self.epsilon * self.epsilon_decay

    return action

  def advantage(self, cum_reward, state):
    preds = self.combined_model(tf.convert_to_tensor(state))
    return cum_reward - preds[0]

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

      predictions = self.combined_model(tf.convert_to_tensor(state, dtype = tf.float64))
      value       = predictions[0]
      policy      = predictions[1]
      probs       = tf.nn.softmax(policy)

      advantage = tf.convert_to_tensor(cum_reward, dtype = tf.float64) - value

      v_loss = tf.math.pow(advantage, 2)

      actions_index = tf.one_hot(
          tf.convert_to_tensor(action),
          self.num_actions
      )
      p_loss = tf.nn.softmax_cross_entropy_with_logits(
          labels = actions_index,
          logits = policy
      )
      p_loss = p_loss * tf.stop_gradient(advantage)

      loss = 0.5 * p_loss + 0.5 * v_loss
    grad = tape.gradient(loss, self.combined_model.trainable_weights)
    return grad

  # Should only be used on the global agent.
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
  T_max = 200  # max number of episodes
  t_max = 20
  max_episode_length = 2000
  gamma = 0.99

  agent = Agent(
      env.action_space.n,
      num_states,
      epsilon = 0.0
  )
  print("Agent %d made agent" % id)

  # Reset local gradients
  combined_gradients = \
      [tf.zeros_like(tw) for tw in agent.combined_model.trainable_weights]

  while T <= T_max:
    T = T + 1

    state = env.reset()
    state = np.reshape(state, (1, num_states))
    if __debug__ and id == 0:
      print("AGENT 0 STARTED NEW EPISODE, T=%d" % (T,))

    state_buffer  = []
    action_buffer = []
    reward_buffer = []
    score = 0

    t = 0  # local time step counter
    terminated = False

    while not terminated:
      if __debug__ and id == 0:
        print("AGENT 0 TAKING STEP IN ENVIRONMENT, t=%d" % (t,))

      predictions = agent.combined_model(tf.convert_to_tensor(state))
      value       = predictions[0]
      logits      = predictions[1]
      probs       = tf.nn.softmax(logits)

      if id == 0:
        env.render(mode = 'close')

      action = np.random.choice(agent.num_actions, p = probs.numpy()[0])
      # if __debug__ and id == 0:
        # print("Probabilities", probs.numpy(), "action", action)

      next_state, reward, terminated, _ = env.step(action)
      if terminated:
        reward = -1

      state_buffer.append(state)
      action_buffer.append(action)
      reward_buffer.append(reward)
      score = score + reward

      t = t + 1

      if t == t_max or terminated:

        # Compute and send gradients
        cum_reward = \
            0 if terminated \
              else agent.combined_model(tf.convert_to_tensor(state))[0]

        for i in reversed(range(t)):
          cum_reward = reward_buffer[i] + gamma * cum_reward

          combined_gradients = add_gradients(
              combined_gradients,
              agent.combined_gradients(
                  state_buffer[i],
                  action_buffer[i],
                  cum_reward
              )
          )

        # Send local gradients to global agent
        gradient_queue.put(combined_gradients)
        if __debug__ and id == 0:
          print("Agent 0 put gradients in queue")

        # Reset local gradients
        combined_gradients = \
            [tf.zeros_like(tw) for tw in agent.combined_model.trainable_weights]

        # Request weights from global agent. 1 is a dummy
        sync_connection.send(1)

        # Synchronise local and global parameters
        weights = sync_connection.recv()
        # old_weights = agent.combined_model.get_weights()
        agent.combined_model.set_weights(weights)

        state_buffer  = []
        action_buffer = []
        reward_buffer = []

        if  terminated:
          print("Agent %d got score %f" % (id, score))
        
        # reset update timer
        t = 0

      state = next_state
      state = np.reshape(state, (1, num_states))

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

        weights = global_agent.combined_model.get_weights()
        sync_connections[i].send(weights)

    # Queue.empty() is unreliable according to docs, may cause bugs
    while not gradient_queue.empty():
      grads = gradient_queue.get()
      global_agent.apply_combined_gradients(grads)
      if __debug__:
        print("Global agent updated weights")

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

