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

import time
import random
import matplotlib.pyplot as plt
plt.ion()


R_CLASSIFY = 0.
R_LOCALIZE = 0.
R_FOVEAL = 100.
# R_SACCADE = -0.05
R_SACCADE = -0.0
R_MISCLASSIFY = -R_CLASSIFY

SCALES = [1, 3, 5]
IM_PIX = 256
FOV_PIX = 32
N_CNN_CHANNELS_OUT = [16, 16, 16, 16]
CHARS = '0123456789 '
EP_LENGTH = 20

class EnvSaccadeDigit(Env):
    def __init__(self, seed=None, cuda=False):
        if seed is None:
            seed = time.time()
        random.seed(seed)

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

        # Actions are: [saccade, classify, uncertain, x, y, classes (11)]
        self.action_space = Box(low=-1.0, high=1.0, shape=(self.n_classes+5,), dtype=np.float32)
        
        # WARNING: Can observation space be [0,Inf]?
        self.observation_space = Box(low=0.0, high=np.Inf, shape=(256,), dtype=np.float32)    # ReLU output from RNN

        ## Construct network and optimizer, and initialize parameters...
        self.net = RnnChars2(n_pix_1d=self.fov_pix,
                             n_scales=self.n_scales,
                             n_cnn_channels_out=N_CNN_CHANNELS_OUT,
                             n_classes=self.n_classes)

        self.cuda = cuda
        self.device = torch.device("cuda:0" if torch.cuda.is_available() and cuda else "cpu")
        self.net.to(self.device)
        self.net.eval()

        self.reset()


    def _create_image(self):
        # NOTE: If more fonts are desired, see here:
        #   https://www.codesofinterest.com/2017/07/more-fonts-on-opencv.html
        cvfonts = (cv2.FONT_HERSHEY_SIMPLEX,
                   cv2.FONT_HERSHEY_PLAIN,
                   cv2.FONT_HERSHEY_DUPLEX,
                   cv2.FONT_HERSHEY_COMPLEX,
                   cv2.FONT_HERSHEY_TRIPLEX,
                   cv2.FONT_HERSHEY_COMPLEX_SMALL,
                   cv2.FONT_HERSHEY_SCRIPT_SIMPLEX,
                   cv2.FONT_HERSHEY_SCRIPT_COMPLEX,
                   cv2.FONT_ITALIC)

        cvlines = (cv2.FILLED, cv2.LINE_8, cv2.LINE_4, cv2.LINE_AA)

        b_success = False
        while not b_success:
            try:
                color_bg = (random.randint(0,255), random.randint(0,255), random.randint(0,255))
                color_fg = (random.randint(0,255), random.randint(0,255), random.randint(0,255))
                if np.sum(color_bg)==np.sum(color_fg):
                    # No color contrast. Method below for finding crop will fail.
                    continue

                char_label = np.random.randint(self.n_classes)
                char = CHARS[char_label]
                font = np.random.choice(cvfonts)
                line = np.random.choice(cvlines)
                # scale = random.uniform(3,10)
                # thickness = np.random.randint(1,20)
                scale = random.uniform(3,5)
                thickness = np.random.randint(1,10)

                im = np.array(PIL.Image.new('RGB', (self.im_pix, self.im_pix), color_bg))
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
                    center_y = 2 * (ix_top + w/2.)/self.im_pix - 1
                    self.char_center = (center_x, center_y)

                    im = np.array(PIL.Image.new('RGB', (self.im_pix, self.im_pix), color_bg))
                    im[ix_top:ix_top+h, ix_left:ix_left+w, :] = crop

                    # Get black/white version, for output loss function
                    cropbw = imbw[top:bottom+1, left:right+1]
                    imbw = np.array(PIL.Image.new('L', (self.im_pix, self.im_pix), (0)))
                    imbw[ix_top:ix_top+h, ix_left:ix_left+w] = cropbw
                
                else:
                    self.char_center = (0.0, 0.0)   # meaningless for blank character

                b_success = True

            except:
                import sys, traceback
                traceback.print_exc(file=sys.stdout)
                continue


            ## Build images at other scales/resolutions, using torch
            self.im_hires = im
            im = im.astype(np.float32)/128 - 1
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
            self.char = char
            self.char_label = char_label
            self.imbw = np.greater(imbw, 0.0)
            self.unobserved = self.imbw


    def _get_glimpse(self, fix_loc=None):
        # fix_loc (x, y) values should be in [-1, 1] range
        # fix_loc should be one-element list containing tensor of size(2,)
        if fix_loc is None:
            # Randomly select fixation location
            fix_x = 2 * random.randint(0,self.im_pix)/self.im_pix - 1
            fix_y = 2 * random.randint(0,self.im_pix)/self.im_pix - 1
            fix_loc = np.array([fix_x, fix_y])
        fix_loc = torch.FloatTensor(fix_loc)
        fix_loc.to(self.device)
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


    def reset(self):
        self.current_step = 0
        self._create_image()
        glimpse, self.fix_loc, num_pix_observed = self._get_glimpse()

        # Exclude pix in initial foveal region from total number of pix "observable" by agent...
        self.max_pix_observable = np.sum(self.imbw) - num_pix_observed

        # Output of independently trained RNN is the environment state.
        out_rnn = self.net.forward_rnn(glimpse, self.fix_loc)
        return out_rnn.detach().cpu().numpy().flatten()


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
        true_action = np.argmax(action[0:3]) # [saccade, classify, uncertain]

        if true_action == 0:
            # Declaration: saccade
            
            glimpse, self.fix_loc, num_pix_observed = self._get_glimpse(action[3:5])    # action[3,4] = x,y
            out_rnn = self.net.forward_rnn(glimpse, self.fix_loc).flatten()

            state = out_rnn.detach().cpu().numpy()
            reward = num_pix_observed / (self.max_pix_observable + 0.0001) * R_FOVEAL
            reward += R_SACCADE   # saccade penalty (MIGHT NOT WANT THIS. Just rely on time discounting to promote minimal saccades.)

        elif true_action == 1:
            # Declaration: classify
            
            char_prediction = np.argmax(action[-self.n_classes:])
            if char_prediction==self.char_label:
                # Correct
                reward += R_CLASSIFY
                reward += np.sum(self.unobserved) / (self.max_pix_observable + 0.0001) * R_FOVEAL
            else:
                # Incorrect
                reward += R_MISCLASSIFY

            # Give partial reward if predicted (x,y) location is within 1 unit of actual center location
            if self.char != ' ':
                dist = np.sqrt((action[3]-self.char_center[0])**2 + (action[4]-self.char_center[1])**2)
                dist = np.clip(dist, 0, 1)
                r = (np.cos(np.pi*dist) + 1) / 2
                reward += R_LOCALIZE * r
            state = np.zeros(self.observation_space.shape)
            done = True

        elif true_action == 2:
            # Declaration: uncertain

            # Give partial reward if predicted (x,y) location is within 1 unit of actual center location
            if self.char != ' ':
                dist = np.sqrt((action[3]-self.char_center[0])**2 + (action[4]-self.char_center[1])**2)
                dist = np.clip(dist, 0, 1)
                r = (np.cos(np.pi*dist) + 1) / 2
                reward += R_LOCALIZE * r
            state = np.zeros(self.observation_space.shape)
            done = True

        else:
            # Should never reach this
            print('Error.')
            sys.exit()

        return state, reward, done, {}


    def render(self, fig_name=None, mode='human'):
        # print('EnvSaccadeDigit.render() not yet implemented.')
        plt.subplot(1,2,1)
        plt.imshow(self.im_hires, aspect='equal')
        plt.xticks([])
        plt.yticks([])
        plt.subplot(1,2,2)
        plt.imshow(self.unobserved, aspect='equal')
        plt.xticks([])
        plt.yticks([])

        plt.pause(0.05)
