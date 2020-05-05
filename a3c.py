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
import matplotlib.pyplot as plt
import datetime as dt


tf.keras.backend.set_floatx('float64')
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

# ENVIRONMENT = 'LunarLander-v2'
ENVIRONMENT = 'CartPole-v1'

parser = argparse.ArgumentParser()
parser.add_argument('--test', dest = 'test_model', type = str)
parser.add_argument('--freq', dest = 'freq', default = 20, type = int)
parser.add_argument('--avg', dest = 'num_trainings', default = 1, type = int)
parser.add_argument('--stochastic', dest = 'stoc', action = 'store_true')
parser.add_argument('--Tmax', dest = 'global_T_max', default = 200, type = int)
args = parser.parse_args()

# Setup for saving models during training
save_dir = os.path.join('.', ENVIRONMENT)
try:
  os.mkdir(os.path.join(save_dir, 'training_episode_scores'))
except FileExistsError:
  print("Directory exists")
try:
  os.mkdir(os.path.join(save_dir, 'figures'))
  print("Made directory %s" % save_dir)
except FileExistsError:
  print("Directory %s already exists, skipping creation" % os.path.join(save_dir, 'figures'))
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
      learning_rate = 0.0025):

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
      p_loss = tf.nn.sparse_softmax_cross_entropy_with_logits(
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


def worker_main(id, gradient_queue, scores_queue, exit_queue, sync_connection, global_T):

  epsilon_min   = 0.01
  epsilon_decay = 0.99
  eps           = 0.5

  gamma = 0.99

  explore_time = 5
  exploring_counter = 10
  # explore_check_freq = 35
  explore_check_freq = 20
  explore_check_counter = 0
  explore_eps = 1.0
  
  moving_avg_buffer = np.zeros(5)
  moving_avg_ptr    = 0
  moving_avg_score  = 0
  prev_moving_avg_score = 0
  delta = 0

  t_max = args.freq

  best_episode_score = 0

  env = gym.make(ENVIRONMENT)
  env._max_episode_steps = 500
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

  while global_T.value < args.global_T_max:
    if exploring_counter > 0:
      # Set epsilon for exploration period
      # current_eps = max(
          # min(explore_eps, 10 / (abs(delta) + 1)),
          # 0.1,
          # eps)
      current_eps = max(explore_eps, eps + 0.1)
      exploring_counter -= 1
      print("\t\t---- AGENT %d EXPLORING WITH EPS=%f----" % (id, current_eps))
    else:
      # Set espilon to the normal decay-epsilon
      current_eps = eps
      if eps > epsilon_min:
        eps = eps * epsilon_decay

      explore_check_counter += 1

    if explore_check_counter == explore_check_freq:
      explore_check_counter = 0
      moving_avg_score = np.mean(moving_avg_buffer)
      delta = moving_avg_score - prev_moving_avg_score - 20
      if id == 0:
        print("\t\t---- AGENT 0 CHECKING IF IT SHOULD EXPLORE, delta=%f ----" % delta)
      if delta <= 0:
        print("\t\t---- AGENT %d STARTS EXPLORING ----" % id)
        exploring_counter = explore_time

      prev_moving_avg_score = moving_avg_score

    with global_T.get_lock():
      global_T.value += 1
      current_episode = global_T.value

    state = env.reset()
    state = np.reshape(state, (1, num_states))

    state_buffer  = []
    action_buffer = []
    reward_buffer = []
    score = 0

    t = 0  # local time step counter
    terminated = False

    while not terminated:

      predictions = agent.combined_model(tf.convert_to_tensor(state))
      value       = predictions[0]
      logits      = predictions[1]
      probs       = tf.nn.softmax(logits)

      # if id == 0:
        # env.render(mode = 'close')

      if args.stoc:
        action = np.random.choice(num_actions, p = probs.numpy()[0])
      else:
        if np.random.rand() <= current_eps:
          action = np.random.randint(num_actions)
        else:
          action = np.argmax(probs.numpy())
        # if eps > epsilon_min:
          # eps = eps * epsilon_decay

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

        # Reset local gradients
        combined_gradients = \
            [tf.zeros_like(tw) for tw in agent.combined_model.trainable_weights]

        # Request weights from global agent. 1 is a dummy
        sync_connection.send(1)

        if  terminated:
          print("Agent %d, episode %d/%d, got score %f" % (id, current_episode, args.global_T_max, score))
          scores_queue.put(score)
          moving_avg_buffer[moving_avg_ptr] = score
          moving_avg_ptr = (moving_avg_ptr + 1) % len(moving_avg_buffer)

          # Save model if better than previous
          # (may be false due to frequent updating)
          if score > best_episode_score:
            print("Agent %d saving local model with score %.0f" % (id, score))
            agent.combined_model.save_weights(
                os.path.join(save_dir, 'tmpa3c')
            )
            # agent.combined_model.save_weights(
                # os.path.join(save_dir, 'model_score_%.0f' % score)
            # )
            best_episode_score = score

        # Synchronise local and global parameters
        weights = sync_connection.recv()
        agent.combined_model.set_weights(weights)

        state_buffer  = []
        action_buffer = []
        reward_buffer = []
        
        # Reset update timer
        t = 0

      state = next_state
      state = np.reshape(state, (1, num_states))

  exit_queue.put(id)
  print("Agent %d quit" % id)


def global_main(num_agents, gradient_queue, scores_queue, exit_queue, sync_connections, score_history_conn, global_T):

  env = gym.make(ENVIRONMENT)
  print("Globl agent made environment")
  # env.seed(0)

  # T = 0  # TODO global, shared between all processes
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

    while not exit_queue.empty():
      exited_id = exit_queue.get()
      has_exited[exited_id] = True
      print("Global agent found agent %d has exited" % exited_id)

    time.sleep(0.2)

  scores = []
  while not scores_queue.empty():
    local_score = scores_queue.get()
    scores.append(local_score)

  score_history_conn.send(scores)
  print("Global agent is done")


def train():

  num_agents = 4
  
  score_history_pipe = mp.Pipe()
  score_history_conn = score_history_pipe[0]

  gradient_queue =  mp.Queue()
  scores_queue   =  mp.Queue()
  exit_queue     =  mp.Queue()
  sync_pipes     = [mp.Pipe() for _ in range(num_agents)]
  connections_0  = [c[0] for c in sync_pipes]
  connections_1  = [c[1] for c in sync_pipes]

  global_T = mp.Value('i', 0)

  global_process = mp.Process(
      target = global_main,
      args   = (num_agents, gradient_queue, scores_queue, exit_queue, connections_0, score_history_pipe[1], global_T)
  )
  global_process.start()
  print("Started global process")

  processes = []
  for id in range(num_agents):
    proc = mp.Process(
        target = worker_main,
        args   = (id, gradient_queue, scores_queue, exit_queue, connections_1[id], global_T)
    )
    processes.append(proc)
    proc.start()
    print("Started process %d" % id)

  for id in range(num_agents):
    processes[id].join()
    print("Joined process %d" % id)

  training_score_history = score_history_conn.recv()
  global_process.join()
  print("Joined global process")

  return training_score_history

def test():
  env = gym.make(ENVIRONMENT)
  print("Globl agent made environment")

  num_actions = env.action_space.n
  num_states  = env.observation_space.shape[0]
  global_agent = Agent(
      num_actions,
      num_states
  )
  global_agent.combined_model.load_weights(os.path.join(save_dir, './tmpa3c'))
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
    for _ in range(args.num_trainings):
      print("Starting training")
      score_history = train()

      moving_avg_scores = []
      moving_avg_len    = 50
      for i in range(len(score_history)):
        l = min(i, moving_avg_len)
        moving_avg_scores.append( np.mean(score_history[i - l: i + 1]) )

      fname = dt.datetime.today().strftime("%Y-%m-%d-%X") \
          + ":training_scores"
      np.savetxt(os.path.join(save_dir, 'training_episode_scores', fname), score_history)

      plt.figure(figsize = (8, 4))
      plt.plot(score_history)
      plt.xlabel("Episode")
      plt.ylabel("Episode score")
      plt.savefig(os.path.join(save_dir, 'figures', 'scores.pdf'))

      plt.figure(figsize = (8, 4))
      plt.plot(moving_avg_scores)
      plt.xlabel("Episode")
      plt.ylabel("Moving average episode score (length %d)" % moving_avg_len)
      plt.savefig(os.path.join(save_dir, 'figures', 'moving_avg_scores.pdf'))
      print("Saved figures in", os.path.join(save_dir, 'figures'))

  print("Done")


