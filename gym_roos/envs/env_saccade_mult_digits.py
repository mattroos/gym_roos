# env_saccade_digit.py
#
# To register and use with gym, first install this package from
# the top directory:
#   pip install -e .
#
# Then load it from gym when needed, in python:
#
#   import gym
#   import gym_roos
#   env = gym.make('SaccadeDigit-v0')
#
# Or, to import it without using gym:
#   from gym_roos.envs.env_saccade_digit import EnvSaccadeDigit
#   env = EnvSaccadeDigit()


from gym_roos.envs.models import RnnChars2
from gym_roos.envs import net_utils

import numpy as np

from gym import Env
from gym.spaces import Discrete, MultiDiscrete, MultiBinary, Box

import PIL
import cv2
import torch
import torch.nn.functional as F

import os
import time
import random
import pdb
import matplotlib.pyplot as plt
plt.ion()


#########################
## TODO
# - Might need/want action vector to have dimensions for x/y character center
#   location that are distinct from those for x/y fixation location.


R_CLASSIFY = 100.
R_LOCALIZE = 100.
R_FOVEAL = 0.   # TODO: Does this help convergence? Accuracy?
R_SACCADE = -0. # TODO: Does this help avoid excessive saccades (push for faster decisions)
# R_MISCLASSIFY = -R_CLASSIFY   # Misclassification penalty hurts convergence?
R_MISCLASSIFY = 0.

SCALES = [1, 3, 5]
IM_PIX = 256
FOV_PIX = 32
N_CNN_CHANNELS_OUT = [16, 16, 16, 16]
# CHARS = '0123456789 '
CHARS = '0123456789X'
EP_LENGTH = 10

# NOTE: If more fonts are desired, see here:
#   https://www.codesofinterest.com/2017/07/more-fonts-on-opencv.html
CVFONTS = (cv2.FONT_HERSHEY_SIMPLEX,
           cv2.FONT_HERSHEY_PLAIN,
           cv2.FONT_HERSHEY_DUPLEX,
           cv2.FONT_HERSHEY_COMPLEX,
           cv2.FONT_HERSHEY_TRIPLEX,
           cv2.FONT_HERSHEY_COMPLEX_SMALL,
           cv2.FONT_HERSHEY_SCRIPT_SIMPLEX,
           cv2.FONT_HERSHEY_SCRIPT_COMPLEX,
           cv2.FONT_ITALIC)
CVLINES = (cv2.FILLED, cv2.LINE_8, cv2.LINE_4, cv2.LINE_AA)

eps = np.finfo(np.float).eps

class EnvSaccadeMultDigits(Env):
    def __init__(self, seed=None, cuda=False,
            # model_file='gym_roos/envs/model_params/model_env_saccade_digit.h5'):
            # model_file='/home/mroos/Code/gym_roos/gym_roos/envs/model_params/model_env_saccade_digit.h5'):
            model_file='/Users/mattroos/Code/gym_roos/gym_roos/envs/model_params/model_env_saccade_digit.h5'):
        if seed is None:
            seed = time.time()
        random.seed(seed)

        self.cuda = cuda
        self.device = torch.device("cuda:0" if torch.cuda.is_available() and cuda else "cpu")
        self.classes = CHARS
        self.scales = SCALES
        self.n_classes = len(CHARS)
        self.n_scales = len(SCALES)
        self.current_step = 0
        self.ep_length = EP_LENGTH
        self.im_pix = IM_PIX
        self.fov_pix = FOV_PIX
        self.fov_pix_half = FOV_PIX//2
        self.images = [None]
        self.mask = None
        self.char = None
        self.reward_sum = 0

        # Actions are: [saccade, classify, uncertain, done, x_fix, y_fix, x_char, y_char, w, h, classes (11)]
        # TODO: Should "uncertain" be a different class, rather than a different action?
        self.action_space = Box(low=-1.0, high=1.0, shape=(self.n_classes+10,), dtype=np.float32)
        
        # WARNING: Can observation space be [0,Inf]?
        self.observation_space = Box(low=0.0, high=np.Inf, shape=(256,), dtype=np.float32)    # ReLU output from RNN

        ## Construct network.
        #  The network is similar to that by Minh, et al.
        #       http://papers.nips.cc/paper/5542-recurrent-models-of-visual-attention
        #  It converts the input image (glimpse) and fixation coordinates into
        #  a feature vector that is stored in an RNN. The output of the RNN
        #  serves as the observation vector for this environment.
        self.net = RnnChars2(n_pix_1d=self.fov_pix,
                             n_scales=self.n_scales,
                             n_cnn_channels_out=N_CNN_CHANNELS_OUT,
                             n_classes=self.n_classes)

        # Initialize network parameters.
        # cwd = os.getcwd()
        # print('\nLoading stored model parameters from %s' % os.path.join(cwd, model_file))
        print('\nLoading stored model parameters from %s' % model_file)
        step_start, learning_rate = net_utils.load_net(model_file, self.net, cuda=self.cuda)

        self.net.to(self.device)
        self.net.eval()

        self.action_last = np.zeros(self.n_classes+10)
        self.true_action_last = -1
        self.reset()


    def _add_char(self):
        b_success = False
        while not b_success:
            try:
                color_bg = (random.randint(0,255), random.randint(0,255), random.randint(0,255))
                color_fg = (random.randint(0,255), random.randint(0,255), random.randint(0,255))
                if np.sum(self.color_bg)==np.sum(color_fg) or np.sum(color_bg)==np.sum(color_fg):
                    # No color contrast. Method below for finding crop will fail.
                    continue

                char_label = np.random.randint(self.n_classes)
                char = CHARS[char_label]
                font = np.random.choice(CVFONTS)
                line = np.random.choice(CVLINES)
                # scale = random.uniform(3,10)
                # thickness = np.random.randint(1,20)
                scale = random.uniform(3,5)
                thickness = np.random.randint(1,10)

                im = np.array(PIL.Image.new('RGB', (self.im_pix, self.im_pix), self.color_bg))
                cv2.putText(im, char, (self.im_pix//8, self.im_pix*7//8), font, scale, color_fg, thickness, line)

                # Get black/white version, for output loss function
                imbw = np.array(PIL.Image.new('RGB', (self.im_pix, self.im_pix), (0,0,0)))
                cv2.putText(imbw, char, (self.im_pix//8, self.im_pix*7//8), font, scale, (1,1,1), thickness, line)
                imbw = imbw[:,:,0]

                if char != ' ':
                    # Can't really know where character is in image. So find its edges and move it
                    # to a random location that doesn't clip the character.
                    im_x = np.sum(im, axis=(0,2))
                    im_y = np.sum(im, axis=(1,2))
                    dx = np.diff(im_x)
                    dy = np.diff(im_y)

                    ix = np.where(dx>0)
                    left = ix[0][0]+1
                    right = ix[0][-1]

                    ix = np.where(dy>0)
                    top = ix[0][0]+1
                    bottom = ix[0][-1]

                    crop = im[top:bottom+1, left:right+1, :]
                    (h, w, _) = crop.shape

                    ix_left = random.randint(0, self.im_pix-w)
                    ix_top = random.randint(0, self.im_pix-h)

                    center_x = 2 * (ix_left + w/2.)/self.im_pix - 1
                    center_y = 2 * (ix_top + h/2.)/self.im_pix - 1
                    char_location = (center_x, center_y, w, h)

                    im = np.array(PIL.Image.new('RGB', (self.im_pix, self.im_pix), self.color_bg))
                    im[ix_top:ix_top+h, ix_left:ix_left+w, :] = crop

                    # Get black/white version, for output loss function
                    cropbw = imbw[top:bottom+1, left:right+1]
                    imbw = np.array(PIL.Image.new('L', (self.im_pix, self.im_pix), (0)))
                    imbw[ix_top:ix_top+h, ix_left:ix_left+w] = cropbw
                
                else:
                    self.char_center = (0.0, 0.0)   # meaningless for blank character

                imbw = np.greater(imbw, 0.0)
                if np.any(np.logical_and(self.imbw, imbw)):
                    # character overlaps with one or more existing characters
                    continue

                # copy character into main image
                self.im_hires[imbw] = im[imbw]
                self.imbw = np.logical_or(self.imbw, imbw)

                # update reward landscape
                rew = np.outer(np.hanning(h), np.hanning(w))
                rew = rew / np.sqrt(np.sum(rew**2))
                self.reward_landscapes[char_label][ix_top:ix_top+h, ix_left:ix_left+w] += rew

                b_success = True

            except:
                import sys, traceback
                traceback.print_exc(file=sys.stdout)
                continue

        return char_label, char_location


    def _create_image(self, num_chars=1):
        ## Initialize background and (empty) mask
        self.color_bg = (random.randint(0,255), random.randint(0,255), random.randint(0,255))
        self.im_hires = np.array(PIL.Image.new('RGB', (self.im_pix, self.im_pix), self.color_bg))
        self.imbw = np.full((self.im_pix, self.im_pix), False)
        self.reward_landscapes = np.zeros((self.n_classes, self.im_pix, self.im_pix))

        ## Add characters
        self.char_labels = []
        self.char_locations = []
        for i_char in range(num_chars):
            char, location = self._add_char()
            self.char_labels.append(char)
            self.char_locations.append(location)
        self.char_labels = np.array(self.char_labels)
        self.char_locations = np.array(self.char_locations)

        ## Build images at other scales/resolutions, using torch
        im = self.im_hires.astype(np.float32)/128 - 1
        images = net_utils.np_to_variable(im, is_cuda=torch.cuda.is_available() and self.cuda).permute(2, 0, 1).contiguous()
        # Pad to accommodate fixations at edges...
        images = F.pad(images, (self.fov_pix_half,self.fov_pix_half, self.fov_pix_half,self.fov_pix_half, 0,0),
                        'constant', value=0)

        pooled = [] # List of pooled images, only those at the desired scales
        if 1 in self.scales:
            pooled.append(images)

        # Iteratively pool by factors of 2...
        yh = (self.im_pix+self.fov_pix)//2
        xh = (self.im_pix+self.fov_pix)//2
        for i_pool in np.arange(2, self.scales[-1]+1):
            x = F.pad(images, (xh,xh, yh,yh, 0,0), 'constant', value=0)
            x = x.view([3, self.im_pix+self.fov_pix, 2, self.im_pix+self.fov_pix, 2])
            x = x.mean(4).mean(2)
            images = x
            if i_pool in self.scales:
                pooled.append(images)

        self.images = pooled
        self.unobserved = np.copy(self.imbw)


    def _get_glimpse(self, fix_loc=None):
        # fix_loc (x, y) values should be in [-1, 1] range
        # fix_loc should be one-element list containing tensor of size(2,)

        if fix_loc is None:
            # Randomly select fixation location
            fix_x = 2 * random.randint(0,self.im_pix)/self.im_pix - 1
            fix_y = 2 * random.randint(0,self.im_pix)/self.im_pix - 1
            fix_loc = np.array([fix_x, fix_y])
        fix_loc = torch.FloatTensor(fix_loc)
        fix_loc = fix_loc.to(self.device)
        fix_loc = [fix_loc]

        # Get crops for each fixation/glimpse
        glimpse = []
        for i_scale in range(0,self.n_scales):
            k = int(2**(self.scales[i_scale]-1))   # pooling kernel size
            r = (np.random.rand(2)-0.5)/(k*10)  # small noise, so rounding isn't biased
            ix = int(np.round(self.im_pix//2*fix_loc[0][0].item()/k + r[0])) + (self.im_pix+self.fov_pix)//2
            iy = int(np.round(self.im_pix//2*fix_loc[0][1].item()/k + r[1])) + (self.im_pix+self.fov_pix)//2
            z = self.images[i_scale][:, iy-self.fov_pix_half:iy+self.fov_pix_half, ix-self.fov_pix_half:ix+self.fov_pix_half]
            # Glimpse is list of lists, because RNN is initially trained on sequence of random
            # fixation locations, and with batches of images. Number of elements in the inner lists is just one.
            # So, glimpse[scale][fixation] should be torch tensor of size (1, 3, self.fov_pix, self.fox_pix),
            glimpse.append([z.view([1, 3, self.fov_pix, self.fov_pix])])
            
            if k==1:
                # Update the unobserved mask
                ix = ix - self.fov_pix_half
                iy = iy - self.fov_pix_half
                ix_left = max(ix-self.fov_pix_half, 0)
                ix_right = min(ix+self.fov_pix_half, self.im_pix)
                iy_top = max(iy-self.fov_pix_half, 0)
                iy_bottom = min(iy+self.fov_pix_half, self.im_pix)
                z = self.unobserved[iy_top:iy_bottom, ix_left:ix_right]
                num_pix_observed = np.sum(z)
                self.unobserved[iy_top:iy_bottom, ix_left:ix_right] = False

                # Draw an outline of the foveal region on the image, for rendering
                cv2.rectangle(self.im_hires, (ix_left,iy_top), (ix_right,iy_bottom), (0,0,0), 2)

        return glimpse, fix_loc, num_pix_observed


    def _get_classify_reward(self, char_prediction, location):
        ## Make prediction landscape
        ix_x = (location[0] + 1)/2 * self.im_pix//2
        ix_y = (location[1] + 1)/2 * self.im_pix//2
        # TODO?:  Scaling below might not be the best approach thanks to tanh() mapping to [-1,1].
        #         The w and h predictions could be very sensitive for low w and low h.
        w_half = (location[2] + 1)/2 * self.im_pix / 2
        h_half = (location[3] + 1)/2 * self.im_pix / 2
        ix_left = np.clip(round(ix_x - w_half), 0, self.im_pix-2).astype(np.int)
        ix_right = np.clip(round(ix_x + w_half), ix_left+3, self.im_pix-1).astype(np.int)
        ix_top = np.clip(round(ix_y - h_half), 0, self.im_pix-2).astype(np.int)
        ix_bottom = np.clip(round(ix_y + h_half), ix_top+3, self.im_pix-1).astype(np.int)
        w = ix_right - ix_left
        h = ix_bottom - ix_top

        p = np.outer(np.hanning(h), np.hanning(w))
        p = p / np.sqrt(np.sum(p**2))
        pred = np.zeros((self.im_pix, self.im_pix))
        pred[ix_top:ix_bottom, ix_left:ix_right] = p

        ## Compute reward based on reward landscape
        rl = self.reward_landscapes[char_prediction]
        reward = np.minimum(rl*pred, rl**2)
        self.reward_landscapes[char_prediction] = np.sqrt(rl**2 - reward)
        reward = np.sum(reward)

        return reward


    def reset(self):
        self.reward_sum = 0
        self.current_step = 0
        self._create_image(2)
        glimpse, self.fix_loc, num_pix_observed = self._get_glimpse()

        # Exclude pix in initial foveal region from total number of pix "observable" by agent...
        self.max_pix_observable = np.sum(self.imbw) - num_pix_observed

        # Output of independently trained RNN is the environment state.
        out_rnn = self.net.forward_rnn(glimpse, self.fix_loc)
        self.last_state = out_rnn.detach().cpu().numpy().flatten()
        return self.last_state


    def step(self, action):
        # Below is a hacky fix to deal with policies that produce actions that fall outside of the action space.
        # See these conversations for possible alternative solutions:
        #   https://github.com/ikostrikov/pytorch-a2c-ppo-acktr-gail/issues/20
        #   https://github.com/ikostrikov/pytorch-a2c-ppo-acktr-gail/issues/109
        #   https://github.com/openai/baselines/issues/121
        action = (np.tanh(action) + 1) / 2 * (self.action_space.high - self.action_space.low) + self.action_space.low

        self.current_step += 1
        done = self.current_step >= self.ep_length
        reward = 0
        true_action = np.argmax(action[0:4]) # [saccade, classify, uncertain, done]

        if true_action == 0:
            # Declaration: saccade
            
            glimpse, self.fix_loc, num_pix_observed = self._get_glimpse(action[4:6])    # action[4,5] = x,y
            out_rnn = self.net.forward_rnn(glimpse, self.fix_loc).flatten()

            state = out_rnn.detach().cpu().numpy()
            self.last_state = state
            reward = (num_pix_observed) / (self.max_pix_observable + eps) * R_FOVEAL
            reward += R_SACCADE   # saccade penalty (MIGHT NOT WANT THIS. Just rely on time discounting to promote minimal saccades.)

        elif true_action == 1 or true_action == 2:
            # Declaration: classify or uncertain
            
            char_prediction = np.argmax(action[-self.n_classes:])

            # Get reward based on remaining reward landscape for the predicted class
            r = self._get_classify_reward(char_prediction, action[6:10])

            if true_action == 1:
                # Classify
                reward += R_CLASSIFY * r
            else:
                # Uncertain. In this case the reward landscape is decremented as if the
                # correct classification was given, but the actual reward is only a fraction
                # of what a true classification reward would be.
                reward += R_CLASSIFY * r / 2.

            state = self.last_state

        elif true_action == 3:
            # Declaration: done
            done = True
            state = np.zeros(self.observation_space.shape)

        else:
            # Should never reach this
            print('Error.')
            sys.exit()

        self.action_last = action
        self.true_action_last = true_action
        self.reward_sum += reward
        # print('a=%d, x=%0.2f, y=%0.2f, r=%0.1f, tr= %0.1f, d=%s' % (true_action, self.fix_loc[0][0], self.fix_loc[0][1],
        #                                                             reward, self.reward_sum, done,))
        # self.render()

        return state, reward, done, {}


    def render(self, fig_name=None, mode='human'):
        fig = plt.figure(1)
        fig.clear()
        (ax1, ax2) = fig.subplots(1, 2)

        ax1.imshow(self.im_hires, aspect='equal')
        ax1.set_xticklabels([])
        ax1.set_yticklabels([])
        # ax1.set_title('s %d, la=%d, r=%0.1f, loc=%0.2f,%0.2f' %(self.current_step, self.true_action_last,
        #     self.reward_sum, self.fix_loc[0][0], self.fix_loc[0][1]))
        ax1.set_title('s %d, la=%d, r=%0.1f, fix=%0.1f,%0.1f, c=%0.1f,%0.1f' %(self.current_step, self.true_action_last,
            self.reward_sum, self.fix_loc[0][0], self.fix_loc[0][1], self.action_last[5], self.action_last[6]))

        ax2.imshow(self.unobserved, aspect='equal')
        ax2.set_xticklabels([])
        ax2.set_yticklabels([])

        plt.pause(0.05)
        pdb.set_trace()
