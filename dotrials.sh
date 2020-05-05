source ./env/bin/activate

for i in {1..5}; do
  python3 a3c.py --Tmax 600
done

mv CartPole-v1/training_episode_scores/* plot_scores/
python3 plot.py
mv plot_scores/* results_standard-eps-decay-adaptive-3
mv plots/* results_standard-eps-decay-adaptive-3


