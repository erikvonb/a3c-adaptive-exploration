import numpy as np
import matplotlib.pyplot as plt
import os
import datetime as dt


SCORES_DIR_1 = os.path.join('.', 'plot_scores', 'set1')
SCORES_DIR_2 = os.path.join('.', 'plot_scores', 'set2')
SAVE_DIR   = os.path.join('.', 'plots')

def plot_moving_average(length):
  n_files = len(os.listdir(SCORES_DIR_1))
  xs_1 = None
  xs_2 = None
  
  plt.figure(figsize = (8, 4))

  for file in os.listdir(SCORES_DIR_1):
    scores = np.loadtxt(os.path.join(SCORES_DIR_1, file))
    moving_avgs = moving_averages(scores, length)

    if xs_1 is None:
      xs_1 = moving_avgs
    else:
      xs_1 = xs_1 + moving_avgs
  xs_1 = xs_1 / n_files

  for file in os.listdir(SCORES_DIR_2):
    scores = np.loadtxt(os.path.join(SCORES_DIR_2, file))
    moving_avgs = moving_averages(scores, length)

    if xs_2 is None:
      xs_2 = moving_avgs
    else:
      xs_2 = xs_2 + moving_avgs
  xs_2 = xs_2 / n_files

  plt.plot(xs_1, label = "Standard decay")
  plt.plot(xs_2, label = "Adaptive", linestyle = 'dashed')
  plt.legend()
  plt.grid()
  plt.xlabel("Episode")
  plt.ylabel("Moving average episode score (length %d)" % length)

  fname = dt.datetime.today().strftime("%Y-%m-%d-%X") \
      + ":moving_averages_%d" % length \
      + ".pdf"
  fpath = os.path.join(SAVE_DIR, fname)

  plt.savefig(fpath)
  print("Saved figure as", fpath)
    
    
def moving_averages(xs, length):
  ys = np.zeros(len(xs))
  for i in range(len(xs)):
    l = min(i, length)
    ys[i] = np.mean(xs[i - l: i + 1])

  return ys


if __name__ == '__main__':
  plot_moving_average(50)
