# models.py

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

import pdb


class CnnChars(nn.Module):
    def __init__(self, im_size_hi=256, im_size_lo=16, n_classes=10):
        super(CnnChars, self).__init__()

        self.im_size_hi = im_size_hi
        self.im_size_lo = im_size_lo

        # hi-res convolutional layers
        self.h_conv1 = nn.Conv2d(3, 16, 3, padding=1)
        self.h_bn1 = nn.BatchNorm2d(16)

        self.h_conv2 = nn.Conv2d(16, 32, 3, padding=1)
        self.h_bn2 = nn.BatchNorm2d(32)

        self.h_conv3 = nn.Conv2d(32, 32, 3, padding=1)
        self.h_bn3 = nn.BatchNorm2d(32)

        self.h_conv4 = nn.Conv2d(32, 32, 3, padding=1)
        self.h_bn4 = nn.BatchNorm2d(32)
        self.h_dropout4 = nn.Dropout(p=0.5)

        n_maxpool_hi = 4
        sz_cnn_out_hi = (self.im_size_hi//(2**n_maxpool_hi))**2 * 32


        # low-res convolutional layers
        self.l_conv1 = nn.Conv2d(3, 16, 3, padding=1)
        self.l_bn1 = nn.BatchNorm2d(16)
        self.l_dropout1 = nn.Dropout(p=0.5)

        n_maxpool_lo = 1
        sz_cnn_out_lo = (self.im_size_lo//(2**n_maxpool_lo))**2 * 16


        # flat layers, applied to (b)oth hi and lo data
        self.b_fc1 = nn.Linear(sz_cnn_out_hi + sz_cnn_out_lo, 1024)

        self.b_bn1 = nn.BatchNorm1d(1024)
        self.b_dropout1 = nn.Dropout(p=0.5)

        self.b_fc2 = nn.Linear(1024, n_classes)

    def forward(self, x_hi, x_lo):
        # hi-res convolutional layers
        x_hi = self.h_conv1(x_hi)
        x_hi = self.h_bn1(x_hi)
        x_hi = F.relu(x_hi)
        x_hi = F.max_pool2d(x_hi, 2)

        x_hi = self.h_conv2(x_hi)
        x_hi = self.h_bn2(x_hi)
        x_hi = F.relu(x_hi)
        x_hi = F.max_pool2d(x_hi, 2)

        x_hi = self.h_conv3(x_hi)
        x_hi = self.h_bn3(x_hi)
        x_hi = F.relu(x_hi)
        x_hi = F.max_pool2d(x_hi, 2)

        x_hi = self.h_conv4(x_hi)
        x_hi = self.h_bn4(x_hi)
        x_hi = F.relu(x_hi)
        x_hi = F.max_pool2d(x_hi, 2)
        x_hi = self.h_dropout4(x_hi)


        # lo-res convolutional layers
        x_lo = self.l_conv1(x_lo)
        x_lo = self.l_bn1(x_lo)
        x_lo = F.relu(x_lo)
        x_lo = F.max_pool2d(x_lo, 2)
        x_lo = self.l_dropout1(x_lo)


        # concatenated hi and lo data
        x_both = torch.cat((x_hi.view(x_hi.shape[0], -1), x_lo.view(x_lo.shape[0], -1)), dim=1)
        x_both = self.b_fc1(x_both)
        x_both = self.b_bn1(x_both)
        x_both = self.b_dropout1(x_both)

        x_both = self.b_fc2(x_both)

        return x_both


class CnnChars2(nn.Module):
    def __init__(self, im_size_hi=256, im_size_lo=16, n_classes=10):
        super(CnnChars2, self).__init__()

        self.n_channels_cnn_hi = 32
        self.n_channels_cnn_lo = 8

        self.im_size_hi = im_size_hi
        self.im_size_lo = im_size_lo

        # hi-res convolutional layers
        self.h_conv1 = nn.Conv2d(3, 16, 3, padding=1)
        self.h_bn1 = nn.BatchNorm2d(16)

        self.h_conv2 = nn.Conv2d(16, 32, 3, padding=1)
        self.h_bn2 = nn.BatchNorm2d(32)

        self.h_conv3 = nn.Conv2d(32, 32, 3, padding=1)
        self.h_bn3 = nn.BatchNorm2d(32)

        self.h_conv4 = nn.Conv2d(32, self.n_channels_cnn_hi, 3, padding=1)
        self.h_bn4 = nn.BatchNorm2d(self.n_channels_cnn_hi)
        self.h_dropout4 = nn.Dropout(p=0.5)

        n_maxpool_hi = 4
        xy_cnn_out_hi = self.im_size_hi//(2**n_maxpool_hi)
        sz_cnn_out_hi = xy_cnn_out_hi**2 * self.n_channels_cnn_hi


        # low-res convolutional layers
        self.l_conv1 = nn.Conv2d(3, self.n_channels_cnn_lo, 3, padding=1)
        self.l_bn1 = nn.BatchNorm2d(self.n_channels_cnn_lo)
        self.l_dropout1 = nn.Dropout(p=0.5)

        n_maxpool_lo = 0
        xy_cnn_out_lo = self.im_size_lo//(2**n_maxpool_lo)
        sz_cnn_out_lo = xy_cnn_out_lo**2 * self.n_channels_cnn_lo


        # # low and hi-res summation layer
        # self.sum_weight_lo = torch.nn.Parameter(torch.zeros(self.n_channels_cnn_lo))
        # self.sum_weight_hi = torch.nn.Parameter(torch.zeros(self.n_channels_cnn_lo))
        # self.sum_bias = torch.nn.Parameter(torch.zeros(self.n_channels_cnn_lo))


        # flatten summed ("b"oth) hi and lo data
        self.b_fc1 = nn.Linear(sz_cnn_out_lo + sz_cnn_out_hi, 1024)
        self.b_bn1 = nn.BatchNorm1d(1024)
        self.b_dropout1 = nn.Dropout(p=0.5)

        self.b_fc2 = nn.Linear(1024, n_classes)


    def forward(self, x_hi, x_lo):

        # hi-res convolutional layers
        x_hi = self.h_conv1(x_hi)
        x_hi = self.h_bn1(x_hi)
        x_hi = F.relu(x_hi)
        x_hi = F.max_pool2d(x_hi, 2)

        x_hi = self.h_conv2(x_hi)
        x_hi = self.h_bn2(x_hi)
        x_hi = F.relu(x_hi)
        x_hi = F.max_pool2d(x_hi, 2)

        x_hi = self.h_conv3(x_hi)
        x_hi = self.h_bn3(x_hi)
        x_hi = F.relu(x_hi)
        x_hi = F.max_pool2d(x_hi, 2)

        x_hi = self.h_conv4(x_hi)
        x_hi = self.h_bn4(x_hi)
        x_hi = F.relu(x_hi)
        x_hi = F.max_pool2d(x_hi, 2)
        x_hi = self.h_dropout4(x_hi)


        # lo-res convolutional layers
        x_lo = self.l_conv1(x_lo)
        x_lo = self.l_bn1(x_lo)
        x_lo = F.relu(x_lo)
        x_lo = self.l_dropout1(x_lo)


        # # Add lo-res channels to subset of hi-res channels
        # x_both = self.sum_weight_lo.unsqueeze(1).unsqueeze(2) * x_lo \
        #         + self.sum_weight_hi.unsqueeze(1).unsqueeze(2) * x_hi[:,0:self.n_channels_cnn_lo,:,:] \
        #         + self.sum_bias.unsqueeze(1).unsqueeze(2)


        # Concatenate hi and lo data
        # x_hi = x_hi[:,self.n_channels_cnn_lo:,:,:]  # lower channels were added into x_both, so don't include
        # x_both = torch.cat((x_both.view(x_both.shape[0], -1), x_hi.view(x_hi.shape[0], -1)), dim=1)
        x_both = torch.cat((x_lo.view(x_lo.shape[0], -1), x_hi.view(x_hi.shape[0], -1)), dim=1)


        # fully connected layers
        x_both = self.b_fc1(x_both.view(x_both.shape[0], -1))
        x_both = self.b_bn1(x_both)
        x_both = self.b_dropout1(x_both)

        x_both = self.b_fc2(x_both)

        return x_both

    def forward_foveal(self, x_hi, x_lo, fix_loc, opts, xy_acc=16, acc_in=None):
        # Assumption is that hi-res CNN will take input patch down to
        # vector of Nx1x1 (N channels), and hi-res CNN will take input
        # patch down to NxMxM channels.
        #
        # fix_loc = (x,y) is the normalized fixation location
        # xy_acc = size of x and y dimension of accumulation matrix
        # acc_in = accumulation matrix to which CNN outputs are added (averaged)
        im_hi = x_hi
        im_lo = x_lo

        batch_size = x_hi.shape[0]
        if acc_in==None:
            acc_in = [None, None]
            acc_in[0] = torch.zeros(batch_size, self.n_channels_cnn_lo, xy_acc, xy_acc) # lo-res info
            acc_in[1] = torch.zeros(batch_size, self.n_channels_cnn_hi, xy_acc, xy_acc) # hi-res info
            if opts.cuda:
                acc_in[0] = acc_in[0].cuda()
                acc_in[1] = acc_in[1].cuda()
        else:
            assert xy_acc==acc_in[0].shape[3]  # ==acc_in[0].shape[4]


        ## hi-res convolutional layers
        x_hi = self.h_conv1(x_hi)
        x_hi = self.h_bn1(x_hi)
        x_hi = F.relu(x_hi)
        x_hi = F.max_pool2d(x_hi, 2)

        x_hi = self.h_conv2(x_hi)
        x_hi = self.h_bn2(x_hi)
        x_hi = F.relu(x_hi)
        x_hi = F.max_pool2d(x_hi, 2)

        x_hi = self.h_conv3(x_hi)
        x_hi = self.h_bn3(x_hi)
        x_hi = F.relu(x_hi)
        x_hi = F.max_pool2d(x_hi, 2)

        x_hi = self.h_conv4(x_hi)
        x_hi = self.h_bn4(x_hi)
        x_hi = F.relu(x_hi)
        x_hi = F.max_pool2d(x_hi, 2)
        x_hi = self.h_dropout4(x_hi)


        ## lo-res convolutional layers
        x_lo = self.l_conv1(x_lo)
        x_lo = self.l_bn1(x_lo)
        x_lo = F.relu(x_lo)
        x_lo = self.l_dropout1(x_lo)


        ## For now, *replace* the relevant section of the accumulation
        # tensor with the CNN outputs.
        ## TODO: Should probably average the two, or learn weightings during training.
        fix_loc = np.round(fix_loc * xy_acc).astype(np.int)
        acc_in[1][:, :, fix_loc[1]:fix_loc[1]+1, fix_loc[0]:fix_loc[0]+1] = x_hi

        # Determine where CNN lo-res output patches go in accumulation tensor
        lo_ext = (x_lo.shape[3]-1)//2
        off_left = min(fix_loc[0], lo_ext)
        off_right = min(xy_acc-fix_loc[0]-1, lo_ext) + 1
        off_up = min(fix_loc[1], lo_ext)
        off_down = min(xy_acc-fix_loc[1]-1, lo_ext) + 1

        acc_in[0][:, :, fix_loc[1]-off_up:fix_loc[1]+off_down, fix_loc[0]-off_left:fix_loc[0]+off_right] = \
            x_lo[:, :, lo_ext-off_up:lo_ext+off_down, lo_ext-off_left:lo_ext+off_right].contiguous()

        # concatenated hi and lo data
        x_both = torch.cat((acc_in[0].view(acc_in[0].shape[0], -1), acc_in[1].view(acc_in[1].shape[0], -1)), dim=1)
        x_both = self.b_fc1(x_both)
        x_both = self.b_bn1(x_both)
        x_both = self.b_dropout1(x_both)

        x_both = self.b_fc2(x_both)

        return x_both, acc_in


class RnnChars(nn.Module):
    def __init__(self, glimpse_sz_hi=16, glimpse_sz_lo=9, n_classes=10):
        super(RnnChars, self).__init__()

        self.n_channels_cnn_hi = 32
        self.n_channels_cnn_lo = 8
        self.n_rnn_layers = 1

        p_dropout = 0.0

        self.glimpse_sz_hi = glimpse_sz_hi
        self.glimpse_sz_lo = glimpse_sz_lo

        # hi-res convolutional layers
        self.h_conv1 = nn.Conv2d(3, 16, 3, padding=1)
        self.h_bn1 = nn.BatchNorm2d(16)

        self.h_conv2 = nn.Conv2d(16, 32, 3, padding=1)
        self.h_bn2 = nn.BatchNorm2d(32)

        self.h_conv3 = nn.Conv2d(32, 32, 3, padding=1)
        self.h_bn3 = nn.BatchNorm2d(32)
        self.h_dropout3 = nn.Dropout(p=p_dropout)

        # self.h_conv4 = nn.Conv2d(32, self.n_channels_cnn_hi, 3, padding=1)
        # self.h_bn4 = nn.BatchNorm2d(self.n_channels_cnn_hi)
        # self.h_dropout4 = nn.Dropout(p=p_dropout)

        n_maxpool_hi = 3
        xy_cnn_out_hi = self.glimpse_sz_hi//(2**n_maxpool_hi)
        sz_cnn_out_hi = xy_cnn_out_hi**2 * self.n_channels_cnn_hi


        # low-res convolutional layers
        # self.l_conv1 = nn.Conv2d(3, 8, 2, padding=0)
        # self.l_bn1 = nn.BatchNorm2d(8)

        # self.l_conv2 = nn.Conv2d(8, 8, 3, padding=0)
        # self.l_bn2 = nn.BatchNorm2d(self.n_channels_cnn_lo)
        # self.l_dropout2 = nn.Dropout(p=p_dropout)

        self.l_fc1 = nn.Linear(5*5*3, 32)
        self.l_dropout1 = nn.Dropout(p=p_dropout)

        # n_maxpool_lo = 1
        # xy_cnn_out_lo = self.glimpse_sz_lo//(2**n_maxpool_lo)
        # sz_cnn_out_lo = xy_cnn_out_lo**2 * self.n_channels_cnn_lo
        sz_cnn_out_lo = 32


        # fixation locations layer (not clear why this is needed)
        sz_fix_out = 32
        self.f_fc1 = nn.Linear(2, sz_fix_out)


        # Combining "glimpse" layer
        self.g_fc1 = nn.Linear(sz_cnn_out_lo + sz_cnn_out_hi + sz_fix_out, 256)


        # RNN layer
        self.rnn = nn.RNN(256, 256, num_layers=self.n_rnn_layers, nonlinearity='relu')
        self.rnn_dropout1 = nn.Dropout(p=p_dropout)


        # Final classification layer
        # Decision to be made: Train on each output of sequence, or only the final one?
        self.c_fc1 = nn.Linear(256, n_classes)


    def forward(self, seq_hi, seq_lo, seq_fix_loc, opts):

        n_fix = len(seq_hi)
        batch_size = seq_hi[0].shape[0]
        glimpse = []

        for i_fix in range(n_fix):

            # Hi-res convolutional layers
            x_hi = self.h_conv1(seq_hi[i_fix])
            x_hi = self.h_bn1(x_hi)
            x_hi = F.relu(x_hi)
            x_hi = F.max_pool2d(x_hi, 2)

            x_hi = self.h_conv2(x_hi)
            x_hi = self.h_bn2(x_hi)
            x_hi = F.relu(x_hi)
            x_hi = F.max_pool2d(x_hi, 2)

            x_hi = self.h_conv3(x_hi)
            x_hi = self.h_bn3(x_hi)
            x_hi = F.relu(x_hi)
            x_hi = F.max_pool2d(x_hi, 2)
            x_hi = self.h_dropout3(x_hi)


            # Low-res convolutional layers
            # x_lo = self.l_conv1(seq_lo[i_fix])
            # x_lo = self.l_bn1(x_lo)
            # x_lo = F.relu(x_lo)
            # x_lo = F.max_pool2d(x_lo, 2)

            # x_lo = self.l_conv2(x_lo)
            # x_lo = self.l_bn2(x_lo)
            # x_lo = F.relu(x_lo)
            # # x_lo = F.max_pool2d(x_lo, 2)
            # x_lo = self.l_dropout2(x_lo)

            x_lo = seq_lo[i_fix]
            x_lo = x_lo.view(x_lo.shape[0], -1)
            x_lo = self.l_fc1(x_lo)
            x_lo = F.relu(x_lo)
            x_lo = self.l_dropout1(x_lo)


            # Fixation locations
            f = self.f_fc1(seq_fix_loc[i_fix])
            f = F.relu(f)
            f = f.repeat(batch_size, 1)  # location is same for all samples in the batch


            # Flatten cnn outputs and concatenate all glimpse info,
            # then put through final glimpse layer
            x_hi = x_hi.view(x_hi.shape[0], -1)
            x_lo = x_lo.view(x_lo.shape[0], -1)
            g = torch.cat((x_hi, x_lo, f), dim=1)
            g = self.g_fc1(g)
            g = F.relu(g)
            glimpse.append(g)


        # Combine into sequence tensors for RNN, and process
        glimpse = torch.stack(glimpse, 0)
        rnn_out, rnn_hidden = self.rnn(glimpse)
        rnn_out = self.rnn_dropout1(rnn_out[-1,:,:])    # ignore all but final output of RNN

        pred_label = self.c_fc1(rnn_out)

        return pred_label


class RnnChars2(nn.Module):
    # This is loosely modeled on the glimpse network by Mnih et al., Recurrent Models of Visual Attention, 2014.
    def __init__(self, n_pix_1d=32, n_scales=3, n_cnn_channels_out=[16,16,16,16], n_rnn_layers=1, n_classes=11, n_pix_im=256):
        super(RnnChars2, self).__init__()

        self.n_pix_1d = n_pix_1d
        self.n_scales = n_scales
        self.n_cnn_channels_out = n_cnn_channels_out
        self.n_rnn_layers = n_rnn_layers
        self.n_classes = n_classes
        self.n_pix_cnn_out = (n_pix_1d // (2**len(n_cnn_channels_out))) ** 2
        self.n_pix_im_out = n_pix_im

        p_dropout = 0.0

        ## Build ModuleList of ModuleLists, each containing a layer
        #  block for a CNN at a particular scale. Outer ModuleList is
        #  for scales, inner ModuleLists are for layer blocks.
        n_chans = [3] + n_cnn_channels_out  # Assuming three color channels for all scales.
        k = 3   # kernel size
        self.cnns = nn.ModuleList()
        for i_scale in range(n_scales):
            self.cnns.append(nn.ModuleList())
            for i_layerblock in range(len(n_cnn_channels_out)):
                self.cnns[i_scale].append(nn.Sequential( \
                    nn.Conv2d(n_chans[i_layerblock], n_chans[i_layerblock+1], k, padding=1), \
                    nn.BatchNorm2d(n_chans[i_layerblock+1]), \
                    nn.ReLU(), \
                    nn.MaxPool2d(2), \
                    nn.Dropout2d(p=p_dropout)  # Channel dropout. Do this or not?
                    ))
            # self.cnns[i_scale].append(nn.Dropout(p=p_dropout))

        ## Fixation location layer
        #  Not clear why this is needed, but Mnih et al. used it.  Seems like location info
        #  could inatead be input to the RNN layer(s) directory, rather than first transforming
        #  and then combining with the CNN outputs.
        #  Input should be (x,y) location, with range [-1,1] where (0,0) is the image center.
        sz_loc_out = 32
        self.fc_loc = nn.Sequential( \
            nn.Linear(2, sz_loc_out), \
            nn.ReLU(), \
            )

        # Combining CNN and locations layer outputs with "glimpse" layer
        sz_glimpse_out = 256
        self.fc_glimpse = nn.Sequential( \
            nn.Linear(self.n_pix_cnn_out*n_cnn_channels_out[-1]*n_scales + sz_loc_out, sz_glimpse_out), \
            nn.ReLU(), \
            )

        # RNN layer (can't use RNN+dropout in Sequential, because RNN gives two outputs in a tuple)
        sz_rnn = 256
        # self.rnn = nn.Sequential( \
        #     nn.RNN(sz_glimpse_out, sz_rnn, num_layers=n_rnn_layers, nonlinearity='relu'), \
        #     nn.Dropout(p=p_dropout), \
        #     )
        self.rnn = nn.RNN(sz_glimpse_out, sz_rnn, num_layers=n_rnn_layers, nonlinearity='relu')
        #self.rnn = nn.GRU(sz_glimpse_out, sz_rnn, num_layers=n_rnn_layers)
        self.rnn_dropout = nn.Dropout(p=p_dropout)

        # Decoding layers
        self.decode = nn.Sequential( \
            nn.Linear(sz_rnn, n_pix_im), \
            nn.ReLU(), \
            nn.Linear(n_pix_im, n_pix_im*n_pix_im), \
            nn.Sigmoid(), \
            )

        # Classification layers
        # Decision to be made: Train on each output of sequence, or only the final one?
        self.classify = nn.Sequential( \
            nn.Linear(sz_rnn, sz_rnn), \
            nn.ReLU(), \
            nn.Linear(sz_rnn, n_classes), \
            )


    def forward_rnn(self, seq_glimpse, seq_fix_loc):

        n_glimpses = len(seq_glimpse[0])
        batch_size = seq_glimpse[0][0].shape[0]
        glimpse = []

        # For each scale, stack all the glimpses, process with CNN and glimpse layers,
        # then reorder for sequential input to the RNN.
        cnns_out = []
        for i_scale in range(self.n_scales):
            cnns_out.append([])
            cnns_out[i_scale] = torch.stack(seq_glimpse[i_scale], 0)
            cnns_out[i_scale] = cnns_out[i_scale].view([-1] + list(cnns_out[i_scale].shape[2:]))
            for i_nn in range(len(self.cnns[i_scale])):
                cnns_out[i_scale] = self.cnns[i_scale][i_nn](cnns_out[i_scale])
            cnns_out[i_scale] = cnns_out[i_scale].view(n_glimpses*batch_size, -1)   # flatten features

        # Fixation locations
        loc = torch.stack(seq_fix_loc, 0)
        loc_out = self.fc_loc(loc).view(n_glimpses, 1, -1)
        loc_out = loc_out.repeat(1, batch_size, 1)  # location is same for all samples in the batch
        loc_out = loc_out.view(n_glimpses*batch_size, -1)   # flatten features

        # Concatentate all features (CNN features and location features)
        # and process with glimpse layer.
        glimpse_out = torch.cat(cnns_out + [loc_out], 1)
        glimpse_out = self.fc_glimpse(glimpse_out)

        # Reshape into sequences for RNN input
        glimpse_out = glimpse_out.view(n_glimpses, batch_size, -1)

        rnn_out, rnn_hidden = self.rnn(glimpse_out)
        # rnn_out = self.rnn_dropout(rnn_out) # Probably shouldn't have this dropout. B/C just before classification layer.

        return rnn_out[-1,:,:] # use only the final output of the RNN


    def forward_decode(self, x):
        im_out = self.decode(x)
        return im_out


    def forward_classify(self, x):
        class_out = self.classify(x)
        return class_out
