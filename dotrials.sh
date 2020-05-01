source ./env/bin/activate

for i in {1..10}; do
  python3 a3c.py --Tmax 400
done

mv CartPole-v1/training_episode_scores/* plot_scores/
python3 plot.py
mv plot_scores/* archived/epsdecay1_lr0-0025
mv plots/* archived/epsdecay1_lr0-0025

