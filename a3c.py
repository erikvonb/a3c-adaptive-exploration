import os
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'

import tensorflow as tf
import gym
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense, Input
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.activations import relu, linear, softmax
# import tensorflow.keras.backend as kb  # not used?
import numpy as np
import multiprocessing as mp
import time
import os
import argparse
from tensorflow.keras.backend import manual_variable_initialization
manual_variable_initialization(True)


tf.keras.backend.set_floatx('float64')
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

# ENVIRONMENT = 'LunarLander-v2'
ENVIRONMENT = 'CartPole-v1'

parser = argparse.ArgumentParser()
parser.add_argument('--test', dest = 'test_model', type = str)
parser.add_argument('--freq', dest = 'freq', default = 20, type = int)
parser.add_argument('--stochastic', dest = 'stoc', action = 'store_true')
args = parser.parse_args()

# Setup for saving models during training
save_dir = os.path.join('.', ENVIRONMENT)
try:
  os.mkdir(save_dir)
  print("Made directory %s" % save_dir)
except FileExistsError:
  print("Directory %s already exists, skipping creation" % save_dir)


class Agent:

  def __init__(
      self,
      num_actions,
      num_states,
      learning_rate = 0.001):

    # kb.clear_session()  # maybe not needed
    manual_variable_initialization(True)

    self.num_actions   = num_actions
    self.num_states    = num_states
    self.optimizer     = Adam(learning_rate = learning_rate)


    ins    = Input(shape = (num_states,))
    l1     = Dense(100,          activation = relu   )(ins)
    l2     = Dense(100,          activation = relu   )(ins)
    value  = Dense(1,            activation = linear )(l1)
    # NOTE policy output is unnormalised probabilities; use softmax explicitly
    policy = Dense(num_actions,  activation = linear)(l2)
    
    self.combined_model = Model(inputs = ins, outputs = [value, policy])

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

      # non-sparse takes index vector; sparse takes a single index
      # p_loss = tf.nn.softmax_cross_entropy_with_logits(
      p_loss = tf.nn.sparse_softmax_cross_entropy_with_logits(
          # labels = actions_index,
          labels = tf.convert_to_tensor([action]),
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

  epsilon       = 1.0
  epsilon_min   = 0.01
  epsilon_decay = 0.95
  gamma = 0.99

  T = 0  # TODO global, shared between all processes
  T_max = 50  # max number of episodes
  t_max = args.freq
  max_episode_length = 2000  # NOTE not used

  best_episode_score = 0

  env = gym.make(ENVIRONMENT)
  env._max_episode_steps = 2000
  print("Agent %d made environment" % id)
  # env.seed(0)
  num_actions = env.action_space.n
  num_states = env.observation_space.shape[0]

  agent = Agent(
      num_actions,
      num_states
  )
  print("Agent %d made agent" % id)

  # Reset local gradients
  combined_gradients = \
      [tf.zeros_like(tw) for tw in agent.combined_model.trainable_weights]

  while T < T_max:
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

      if args.stoc:
        action = np.random.choice(num_actions, p = probs.numpy()[0])
      else:
        if np.random.rand() <= epsilon:
          action = np.random.randint(num_actions)
        else:
          action = np.argmax(probs.numpy())

        if epsilon > epsilon_min:
          epsilon = epsilon * epsilon_decay

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

        combined_gradients = mult_gradients(combined_gradients, 1 / t)
        # Send local gradients to global agent
        gradient_queue.put(combined_gradients)
        if __debug__ and id == 0:
          print("Agent 0 put gradients in queue")

        # Reset local gradients
        combined_gradients = \
            [tf.zeros_like(tw) for tw in agent.combined_model.trainable_weights]

        # Request weights from global agent. 1 is a dummy
        sync_connection.send(1)

        if  terminated:
          print("Agent %d, episode %d/%d, got score %f" % (id, T, T_max, score))

          # Save model if better than previous
          # (may be false due to frequent updating)
          if score > best_episode_score:
            print("Agent %d saving local model with score %.0f" % (id, score))
            # agent.combined_model.save_weights(
                # os.path.join(save_dir, 'model_score_%.0f' % score)
            # )
            agent.combined_model.save_weights('./tmpa3c')
            best_episode_score = score

        # Synchronise local and global parameters
        weights = sync_connection.recv()
        # old_weights = agent.combined_model.get_weights()
        agent.combined_model.set_weights(weights)

        state_buffer  = []
        action_buffer = []
        reward_buffer = []
        
        # reset update timer
        t = 0

      state = next_state
      state = np.reshape(state, (1, num_states))

  if id == 0:
    time.sleep(10)
    print("\n\nAgent 0 testing its latest local agent")
    for _ in range(4):
      score = 0
      state = env.reset()
      state = np.reshape(state, (1, num_states))
      terminated = False
      while not terminated:
        predictions = agent.combined_model(tf.convert_to_tensor(state))
        logits      = predictions[1]
        probs       = tf.nn.softmax(logits)
        env.render(mode = 'close')
        time.sleep(1/50)
        if args.stoc:
          action = np.random.choice(num_actions, p = probs.numpy()[0])
        else:
          action = np.argmax(probs.numpy())
        next_state, reward, terminated, _ = env.step(action)
        score = score + reward
        state = next_state
        state = np.reshape(state, (1, num_states))
      print("Final test done with score %f" % score)
    
    agent.combined_model.save_weights('./tmpa3c-0')
    time.sleep(2)
    print("\n\nAgent 0 testing again")
    agent.combined_model.load_weights('./tmpa3c-0')
    for _ in range(4):
      score = 0
      state = env.reset()
      state = np.reshape(state, (1, num_states))
      terminated = False
      while not terminated:
        predictions = agent.combined_model(tf.convert_to_tensor(state))
        logits      = predictions[1]
        probs       = tf.nn.softmax(logits)
        env.render(mode = 'close')
        time.sleep(1/50)
        if args.stoc:
          action = np.random.choice(num_actions, p = probs.numpy()[0])
        else:
          action = np.argmax(probs.numpy())
        next_state, reward, terminated, _ = env.step(action)
        score = score + reward
        state = next_state
        state = np.reshape(state, (1, num_states))
      print("Final test done with score %f" % score)

  exit_queue.put(id)
  print("Agent %d quit" % id)


def global_main(num_agents, gradient_queue, exit_queue, sync_connections):

  env = gym.make(ENVIRONMENT)
  print("Globl agent made environment")
  # env.seed(0)

  T = 0  # TODO global, shared between all processes
  gamma = 0.99
  save_freq = 20
  save_counter = 0

  num_actions = env.action_space.n
  num_states  = env.observation_space.shape[0]
  global_agent = Agent(
      num_actions,
      num_states
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
      save_counter = save_counter + 1

      # if save_counter == save_freq:
        # save_counter = 0
        # global_agent.combined_model.save_weights('./tmpa3c')
        # print("Global agent saved weights")

      if __debug__:
        print("Global agent updated weights")

    while not exit_queue.empty():
      exited_id = exit_queue.get()
      has_exited[exited_id] = True
      print("Global agent found agent %d has exited" % exited_id)

    time.sleep(0.2)

  print("Global agent is done")

  print("\nSTARTING GLOBAL TEST")
  # test(None, global_agent)
  global_agent.combined_model.load_weights('./tmpa3c-0')
  for _ in range(4):
    score = 0
    state = env.reset()
    state = np.reshape(state, (1, num_states))
    terminated = False
    while not terminated:
      predictions = global_agent.combined_model(tf.convert_to_tensor(state))
      logits      = predictions[1]
      probs       = tf.nn.softmax(logits)
      env.render(mode = 'close')
      time.sleep(1/50)
      if args.stoc:
        action = np.random.choice(num_actions, p = probs.numpy()[0])
      else:
        action = np.argmax(probs.numpy())
      next_state, reward, terminated, _ = env.step(action)
      score = score + reward
      state = next_state
      state = np.reshape(state, (1, num_states))
    print("Final test done with score %f" % score)

def train():

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


def test2(file, agent = None, episodes = 100):
  print("Running single agent with weights from %s" % file)

  env = gym.make(ENVIRONMENT)
  print("Made environment")
  env.seed(0)
  np.random.seed(0)

  num_states = env.observation_space.shape[0]
  num_actions = env.action_space.n

  if agent is None:
    agent = Agent(
        num_actions, 
        num_states
    )
    print("Made agent")

  # agent.combined_model.load_weights(os.path.join(save_dir, file))
  agent.combined_model.load_weights('./tmpa3c')
  print("Agent loaded weights from %s", file)

  T = 0
  while T < episodes:
    T = T + 1

    state = env.reset()
    state = np.reshape(state, (1, num_states))

    score = 0
    terminated = False

    while not terminated:

      predictions = agent.combined_model(tf.convert_to_tensor(state))
      logits      = predictions[1]
      probs       = tf.nn.softmax(logits)

      env.render(mode = 'close')

      # if args.stoc:
      if False:
        action = np.random.choice(num_actions, p = probs.numpy()[0])
      else:
        action = np.argmax(probs.numpy())

      next_state, reward, terminated, _ = env.step(action)
      score = score + reward

      time.sleep(1/50)

    print("Finished episode %d/%d with score %f" % (T, episodes, score))


def test():
  env = gym.make(ENVIRONMENT)
  print("Globl agent made environment")
  # env.seed(0)

  num_actions = env.action_space.n
  num_states  = env.observation_space.shape[0]
  global_agent = Agent(
      num_actions,
      num_states
  )
  global_agent.combined_model.load_weights('./tmpa3c-0')
  for _ in range(4):
    score = 0
    state = env.reset()
    state = np.reshape(state, (1, num_states))
    terminated = False
    while not terminated:
      predictions = global_agent.combined_model(tf.convert_to_tensor(state))
      logits      = predictions[1]
      probs       = tf.nn.softmax(logits)
      env.render(mode = 'close')
      time.sleep(1/50)
      if args.stoc:
        action = np.random.choice(num_actions, p = probs.numpy()[0])
      else:
        action = np.argmax(probs.numpy())
      next_state, reward, terminated, _ = env.step(action)
      score = score + reward
      state = next_state
      state = np.reshape(state, (1, num_states))
    print("Final test done with score %f" % score)


# ==== utils ====

# Elementwise addition of list of tensors
def add_gradients(xs, ys):
  return list( map(lambda pair: pair[0] + pair[1], zip(xs, ys)) )

def mult_gradients(xs, fac):
  return list( map(lambda x: x * fac, xs) )

if __name__ == '__main__':

  if args.test_model is not None:
    # test(args.test_model)
    test()

  else:
    print("Starting training")
    train()

  print("Done")


