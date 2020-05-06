import numpy as np
import matplotlib.pyplot as plt
import os
import datetime as dt


SCORES_DIR = os.path.join('.', 'plot_scores')
SAVE_DIR   = os.path.join('.', 'plots')

def plot_moving_average(length):
  files = os.listdir(SCORES_DIR)
  xs = None
  
  plt.figure(figsize = (8, 4))

  for file in os.listdir(SCORES_DIR):
    scores = np.loadtxt(os.path.join(SCORES_DIR, file))
    moving_avgs = moving_averages(scores, length)

    plt.plot(moving_avgs, alpha = 0.2, linestyle = 'dashed')

    if xs is None:
      xs = moving_avgs
    else:
      xs = xs + moving_avgs

  xs = xs / len(files)

  plt.plot(xs)
  plt.grid()
  plt.xlabel("Episode")
  plt.ylabel("Moving average episode score (length %d)" % length)
  # plt.ylim(0, 150)

  fname = dt.datetime.today().strftime("%Y-%m-%d-%X") \
      + ":moving_averages_%d" % length \
      + ".pdf"
  fpath = os.path.join(SAVE_DIR, fname)

  plt.savefig(fpath)
  print("Saved figure as", fpath)
  # fname = dt.datetime.today().strftime("%Y-%m-%d-%X") \
      # + ":moving_averages_%d_zoom" % length \
      # + ".pdf"
  # fpath = os.path.join(SAVE_DIR, fname)
  # plt.ylim(0, 200)
  # plt.savefig(fpath)
  
    
    
def moving_averages(xs, length):
  ys = np.zeros(len(xs))
  for i in range(len(xs)):
    l = min(i, length)
    ys[i] = np.mean(xs[i - l: i + 1])

  return ys


if __name__ == '__main__':
  plot_moving_average(50)
