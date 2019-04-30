import gym
import gym_roos
import numpy as np

import matplotlib.pyplot as plt
plt.ion()
import pdb

env = gym.make('SaccadeMultDigits-v0')
env.render()

n_action_labels = 11
n_action_not_label = 8   # s, c, u, d, xfix, yfix, w, h
n_actions = n_action_labels + n_action_not_label

while True:
    a = np.random.rand(n_actions) * 2 - 1
    
    ta = input('Action: (1) Saccade, (2) Classify, (3) Uncertain, (4) Done: ')
    if ta=='':
        a[0] = 1.0  # saccade
    else:
        a[int(ta)-1] = 1.0

    loc = input('Location (x, y): ')
    if loc!='':
        xy = loc.split(',')
        x = float(xy[0]) / 128 - 1
        y = float(xy[1]) / 128 - 1
        a[4] = np.arctanh(x)
        a[5] = np.arctanh(y)

    # a[4] = np.arctanh(env.char_locations[0,0])
    # a[5] = np.arctanh(env.char_locations[0,1])
    # a[6] = np.arctanh((env.char_locations[0,2]*1.5)/128 - 1)
    # a[7] = np.arctanh((env.char_locations[0,3]*1.5)/128 - 1)


    label = input('Class: ')
    if label != '':
        label = int(label)
        a[n_action_not_label+label] = 1

    state, reward, done, _ = env.step(a)
    env.render()

    pdb.set_trace()
