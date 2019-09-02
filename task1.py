import torch
import math
import numpy as np
import torch.distributions as D
import torch.nn as nn
import torchvision.utils as vutils
from torch.nn.functional import grid_sample, affine_grid

"""
Unsupervised Learning of 3D Structure from Images
"""


class Model(nn.Module):
    def __init__(self, params):
        super().__init__()

        self.T = params['T']  # Number of time steps
        self.A = params['A']  # Width
        self.B = params['B']  # Height
        self.C = params['C']  # Depth
        self.z_size = params['z_size']
        self.read_N = params['read_N']
        self.write_N = params['write_N']
        self.enc_size = params['enc_size']
        self.dec_size = params['dec_size']
        self.device = params['device']
        self.channel = params['channel']

        # Stores the generated image for each time step.
        self.cs = [0] * self.T
        self.y = [0] * self.T
        self.x = [0]

        # To store appropriate values used for calculating the latent loss (KL-Divergence loss)
        self.logsigmas = [0] * self.T
        self.sigmas = [0] * self.T
        self.mus = [0] * self.T

        # Encoder and Decoder
        self.encoder = nn.LSTMCell(
            self.read_N * self.read_N * self.read_N * self.channel + self.dec_size + 1 * self.A * self.B, self.enc_size)
        self.decoder = nn.LSTMCell(self.z_size, self.dec_size)
        self.decoding = nn.LSTMCell(self.read_N * self.read_N * self.channel + self.z_size, self.dec_size)

        # To get the mean and standard deviation for the distribution of z.
        self.fc_mu = nn.Linear(self.enc_size, self.z_size)
        self.fc_sigma = nn.Linear(self.enc_size, self.z_size)

        # Write Function
        self.fc_w1 = nn.Linear(self.dec_size, 4)
        self.fc_w2 = nn.Linear(self.dec_size, self.write_N * self.write_N * self.write_N * self.channel)
        self.fc_w3 = nn.Linear(self.dec_size, 1)  # for projection function

        # Read function fo getting the attention parameters
        self.fc_read = nn.Linear(self.dec_size, 4)
        self.fc_read_image = nn.Linear(self.dec_size, 3)

        self.conv_2d = nn.Conv2d(1, 1, kernel_size=5)

        # Projection
        self.conv3d = nn.Conv3d(10, 16, (32, 1, 1))
        self.conv2d = nn.Conv2d(16, 1, (1, 1))

    #         self.conv_3d = nn.MaxPool3d((32, 1, 1), stride=(1, 1, 1), padding=0)

    def forward(self, x):
        self.batch_size = x.size(0)

        # requires_grad should be set True to allow backpropagation of the gradients for training.
        h_enc_prev = torch.zeros(self.batch_size, self.enc_size, requires_grad=True, device=self.device)
        h_dec_prev = torch.zeros(self.batch_size, self.dec_size, requires_grad=True, device=self.device)

        enc_state = torch.zeros(self.batch_size, self.enc_size, requires_grad=True, device=self.device)
        dec_state = torch.zeros(self.batch_size, self.dec_size, requires_grad=True, device=self.device)

        r_t = torch.zeros(self.batch_size, self.read_N * self.read_N * self.read_N, requires_grad=True,
                          device=self.device)
        w_t = torch.zeros(self.batch_size, self.A * self.B * self.C, requires_grad=True, device=self.device)

        context = torch.zeros(self.batch_size, 1 * self.A * self.B, device=self.device)

        for t in range(self.T):
            c_prev = torch.zeros(self.batch_size, self.A * self.B * self.C * self.channel, requires_grad=True,
                                 device=self.device) if t == 0 else self.cs[t - 1]
            # Read function (x, h_dec_prev, output dimension)
            r_t = self.read(x, h_dec_prev)  # Use h_dec_prev to get affine transformation matrix
            # Encoder LSTM
            h_enc, enc_state = self.encoder(torch.cat((r_t, h_dec_prev, context), dim=1), (h_enc_prev, enc_state))
            # Sample
            z, self.mus[t], self.logsigmas[t], self.sigmas[t] = self.sampleQ(h_enc)
            # Context encoder
            #             e = self.f_read(context, h_dec_prev)
            # Decoder LSTM
            h_dec, dec_state = self.decoder(z, (h_dec_prev, dec_state))
            # Write and adding canvas functions (h_dec, output dimension)
            w_t = self.write(h_dec)
            self.cs[t] = c_prev + w_t

            #             h_enc_prev = h_enc
            h_dec_prev = h_dec

    # For volumetric inputs
    def read(self, x, h_dec_prev):
        # params (s, x, y, z)
        params = self.fc_read(h_dec_prev)

        # initiate theta with zeros
        theta = torch.zeros(3, 4).repeat(x.shape[0], 1, 1).to(x.device)
        # set scaling
        theta[:, 0, 0] = theta[:, 1, 1] = theta[:, 2, 2] = params[:, 0]
        # set translation
        theta[:, :, -1] = params[:, 1:]

        grid = affine_grid(theta, (self.batch_size, self.channel, self.read_N, self.read_N, self.read_N))
        out = grid_sample(x.view(x.size(0), 1, 32, 32, 32), grid)
        out = out.view(out.size(0), -1)

        return out  # size of output (64, 125)

    # For volumetric inputs
    def write(self, h_dec):
        # params (s, x, y, z)
        params = self.fc_w1(h_dec)
        x = self.fc_w2(h_dec)

        # initiate theta with zeros
        theta = torch.zeros(3, 4).repeat(x.size(0), 1, 1).to(x.device)
        # set scaling
        theta[:, 0, 0] = theta[:, 1, 1] = theta[:, 2, 2] = 1 / (params[:, 0] + 1e-9)
        # set translation
        theta[:, :, -1] = - params[:, 1:] / (params[:, 0].view(-1, 1) + 1e-9)

        grid = affine_grid(theta, (self.batch_size, self.channel, self.A, self.B, self.C))

        out = grid_sample(x.view(x.size(0), 1, 5, 5, 5), grid)
        out = out.view(out.size(0), -1)

        return out  # size ot output (64 batch size 1x16x16x16)

    def sampleQ(self, h_enc):
        e = torch.randn(self.batch_size, self.z_size, device=self.device)

        mu = self.fc_mu(h_enc)
        log_sigma = self.fc_sigma(h_enc)
        sigma = torch.exp(log_sigma)

        z = mu + e * sigma

        return z, mu, log_sigma, sigma

        # For image inputs in generative model

    def f_read(self, context, h_dec_prev):
        context = self.conv_2d(context.view(64, 1, 32, 32))
        # params (s, x, y)
        params = self.fc_read_image(h_dec_prev)

        theta = torch.zeros(2, 3).repeat(context.shape[0], 1, 1).to(context.device)
        # set scaling
        theta[:, 0, 0] = theta[:, 1, 1] = params[:, 0]
        # set translation
        theta[:, :, -1] = params[:, 1:]

        grid = affine_grid(theta, (self.batch_size, self.channel, self.read_N, self.read_N))
        out = grid_sample(context.view(context.size(0), 1, 28, 28), grid)
        out = out.view(out.size(0), -1)

        return out  # size (64, 25)

    def VST(self, x, h_dec):
        # Parameter of rotation
        angle = self.fc_w3(h_dec)

        # Initiate theta with zeros
        theta = torch.zeros(3, 4).repeat(x.shape[0], 1, 1).to(x.device)

        # Rotate around Z
        theta[:, 0, 0] = torch.cos(angle[:, 0])
        theta[:, 1, 1] = torch.cos(angle[:, 0])
        theta[:, 0, 1] = -torch.sin(angle[:, 0])
        theta[:, 1, 0] = torch.sin(angle[:, 0])
        theta[:, 2, 2] = 1

        #         # Rotate around X
        #         theta[:, 0, 0] = 1
        #         theta[:, 1, 1] = math.cos(angle)
        #         theta[:, 1, 2] = -math.sin(angle)
        #         theta[:, 2, 1] = math.sin(angle)
        #         theta[:, 2, 2] = math.cos(angle)

        #         # Rotate around Y
        #         theta[:, 0, 0] = math.cos(angle)
        #         theta[:, 1, 1] = 1
        #         theta[:, 0, 2] = math.sin(angle)
        #         theta[:, 2, 0] = -math.sin(angle)
        #         theta[:, 2, 2] = math.cos(angle)

        # Rx*Ry*Rz
        # |1  0   0| | Cy  0 Sy| |Cz -Sz 0|   | CyCz        -CySz         Sy  |
        # |0 Cx -Sx|*|  0  1  0|*|Sz  Cz 0| = | SxSyCz+CxSz -SxSySz+CxCz -SxCy|
        # |0 Sx  Cx| |-Sy  0 Cy| | 0   0 1|   |-CxSyCz+SxSz  CxSySz+SxCz  CxCy|

        #         theta[:, 0, 0] = torch.cos(angle[:, 1])*torch.cos(angle[:, 2])
        #         theta[:, 0, 1] = -torch.cos(angle[:, 1])*torch.sin(angle[:, 2])
        #         theta[:, 0, 2] = torch.sin(angle[:, 1])
        #         theta[:, 1, 0] = torch.sin(angle[:, 0])*torch.sin(angle[:, 1])*torch.cos(angle[:, 2])+torch.cos(angle[:, 0])*torch.sin(angle[:, 2])
        #         theta[:, 1, 1] = -torch.sin(angle[:, 0])*torch.sin(angle[:, 1])*torch.sin(angle[:, 2])+torch.cos(angle[:,0])*torch.cos(angle[:, 2])
        #         theta[:, 1, 2] = -torch.sin(angle[:, 0])*torch.cos(angle[:, 1])
        #         theta[:, 2, 0] = -torch.cos(angle[:, 0])*torch.sin(angle[:, 1])*torch.cos(angle[:, 2])+torch.sin(angle[:,0])*torch.sin(angle[:, 2])
        #         theta[:, 2, 1] = torch.cos(angle[:, 0])*torch.sin(angle[:, 1])*torch.sin(angle[:, 2])+torch.sin(angle[:,0])*torch.cos(angle[:, 2])
        #         theta[:, 2, 2] = torch.cos(angle[:, 0])*torch.cos(angle[:, 1])

        grid = affine_grid(theta, (self.batch_size, self.channel, self.A, self.B, self.C))
        out = grid_sample(x.view(x.size(0), 1, 32, 32, 32), grid)
        out = out.view(out.size(0), -1)

        return out  # size of output (64, 32x32x32)

    def projection(self, cs, h_dec):
        for i in range(10):
            x = self.VST(cs[i], h_dec)  # (64 x 32x32x32)

        cs = (torch.Tensor(cs)).transpose(0, 1)
        cs = self.conv3d(cs)
        cs = cs.squeeze()
        x = self.conv2d(cs)
        return x.view(64, 1, 32, 32)

    def proj(self, cs, h_dec):
        x = self.VST(cs, h_dec)  # (64 x 32x32x32)
        x = self.conv_3d(x.view(64, 1, 32, 32, 32))
        return x.view(64, 1, 32, 32)

    def loss(self, x):
        self.forward(x)

        criterion = nn.BCELoss()
        x_recon = torch.sigmoid(self.cs[-1])
        # Reconstruction loss.
        # Only want to average across the mini-batch, hence, multiply by the volume dimensions.
        Lx = criterion(x_recon, x) * self.A * self.B * self.C
        # Latent loss.
        Lz = 0

        for t in range(self.T):
            mu_2 = self.mus[t] * self.mus[t]
            sigma_2 = self.sigmas[t] * self.sigmas[t]
            logsigma = self.logsigmas[t]

            kl_loss = 0.5 * torch.sum(mu_2 + sigma_2 - 2 * logsigma, 1) - 0.5 * self.T
            Lz += kl_loss

        Lz = torch.mean(Lz)
        net_loss = Lx + Lz

        return net_loss

    # generative model
    def generate(self, num_output):
        self.batch_size = num_output
        h_dec_prev = torch.zeros(num_output, self.dec_size, device=self.device)
        dec_state = torch.zeros(num_output, self.dec_size, device=self.device)
        context = torch.zeros(self.batch_size, 1 * self.A * self.B)
        w_t = torch.zeros(self.batch_size, self.A * self.B * self.C, requires_grad=True, device=self.device)

        for t in range(self.T):
            c_prev = torch.zeros(self.batch_size, self.A * self.B * self.C, device=self.device) if t == 0 else self.cs[
                t - 1]

            # Sampling function from Normal Distribution (1)
            z = torch.randn(self.batch_size, self.z_size, device=self.device)  # Correct

            # Read function (2)
            #             r = self.f_read(c, h_dec_prev)

            # State function (3)
            h_dec, dec_state = self.decoder(z, (h_dec_prev, dec_state))  # need to incorporate context

            # Write function (4)
            w_t = self.write(h_dec)
            self.cs[t] = c_prev + w_t
            #             self.y[t] = self.VST(self.cs[t], 45)

            # Projection function (5)
            #             self.x[t] = self.proj(self.cs[t], h_dec)

            # Function (6)
            # self.cs[t] = D.Bernoulli(logits=self.cs[t].view(self.batch_size, self.channel, self.A, self.B, self.C)).probs

            h_dec_prev = h_dec

        self.projection(self.cs, h_dec_prev)

        # Visualization
        voxels = []

        for n, img in enumerate(self.cs):
            # the volume dimension is A x B x C
            img = img.view(-1, self.channel, self.A, self.B, self.C)
            voxels.append(img)

        #         voxel = []

        #         for n, img in enumerate(self.y):
        #             # the image dimension is A x B x C
        #             img = img.view(-1, self.channel, self.A, self.B, self.C)
        #             voxel.append(img)

        # Image visualization
        images = []

        img = self.projection.view(-1, self.channel, self.A, self.B)
        images.append(vutils.make_grid(torch.sigmoid(img).detach().cpu(), nrow=int(np.sqrt(int(num_output))), padding=1,
                                       normalize=True, pad_value=1))

        #         images1 = []

        #         img1 = projection1.view(-1, self.channel, self.A, self.B)
        #         images1.append(vutils.make_grid(torch.sigmoid(img1).detach().cpu(), nrow=int(np.sqrt(int(num_output))), padding=1, normalize=True, pad_value=1))

        return voxels, images