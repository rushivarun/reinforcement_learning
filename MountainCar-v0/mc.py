import gym
import numpy as np
import h5py
import random
import keras
import tensorflow as tf

env = gym.make('MountainCar-v0')
env.reset()

model_episodes = 1
model_scores = []
model_actions = []
game_memory = []

for model_episode in range(model_episodes):
  score = 0
  game_steps = 200
#   game_memory = []
  prev_state = []
  for step in range(game_steps):
    #env.render()
    if len(prev_state) == 0:
      action = random.randrange(0, 2)
    else:
      with h5py.File('/home/rushi/Documents/MountainCar.hdf5') as f:
        action = np.argmax(f.predict(prev_state.reshape(-1, len(prev_state)))[0])
    
    new_state, reward, done, info = env.step(action)
    prev_state = new_state
    score += reward
    model_actions.append(action)
    model_scores.append(reward)
    game_memory.append([prev_state, action])
    if done:
      print('-----------------------done--------------------')
      break
    #env.close()
      


print(score)
print(game_memory)

env.reset()

