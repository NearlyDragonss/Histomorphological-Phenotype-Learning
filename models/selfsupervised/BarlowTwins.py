
# Network related lib.
from models.normalization import *

# Losses and Optimizers.
from models.loss import *
from models.networks import encoder_contrastive

import torch.nn as nn

class RepresentationsPathology(nn.Module):
    def __init__(self,
                 z_dim,	                    			# Latent space dimensions.
                 beta_1,                      			# Beta 1 value for Adam Optimizer
                 learning_rate_e,             			# Learning rate Encoder.
                 lambda_=5e-3,							# Lambda weight for redundant representation penalty.
                 temperature=0.1,                       # Temperature for contrastive cosine similarity norm.
                 spectral=True,							# Spectral Normalization for weights.
                 layers=5,					 			# Number for layers for Encoder.
                 attention=28,                			# Attention Layer dimensions, default after height and width equal 28 to pixels.
                 power_iterations=1,          			# Iterations of the power iterative method: Calculation of Eigenvalues, Singular Values.
                 init = 'orthogonal',    			    # Weight Initialization: default Orthogonal.
                 regularizer_scale=1e-4,      			# Orthogonal regularization.
                 model_name='RepresentationsPathology'  # Model Name.
                 ):
        super(RepresentationsPathology, self).__init__()
        # todo: reorder these
        self.relu = nn.ReLU()
        self.model_name = model_name
        self.attention = attention
        self.spectral = spectral
        self.layers = layers
        self.z_dim = z_dim

        ### Architecture parameters.
        self.attention = attention
        self.layers    = layers
        self.spectral  = spectral
        self.z_dim     = z_dim
        self.init      = init # unsure use
        self.lambda_ = lambda_

        ### Hyper-parameters.
        self.power_iterations  = power_iterations # unsure use
        self.regularizer_scale = regularizer_scale # unsure use
        self.learning_rate_e   = nn.Parameter(torch.tensor(learning_rate_e), requires_grad=False)
        self.beta_1            = nn.Parameter(torch.tensor(beta_1), requires_grad=False)
        self.temperature       = temperature # unsure use

        # Model parameters
        self.weight = None
        self.bias = None
        self.gamma = None
        self.encoder = encoder_contrastive.EncoderResnetContrastive(model_name=self.model_name, z_dim=self.z_dim, h_dim=1024, layers=self.layers, spectral=self.spectral,
                                                                    activation=ReLU, init=self.init, normalization=batch_norm, attention=self.attention)


    def forward(self, images, is_train):
        # Encoder Training.
        conv_space, h_rep, z_rep = self.encoder.forward(model=self, images=images, is_train=is_train)
        return conv_space, h_rep, z_rep

    def set_weight(self, weight):
        self.weight = nn.Parameter(weight)

    def set_bias(self, bias):
        self.bias = nn.Parameter(bias)

    def set_gamma(self, gamma):
        self.gamma = nn.Parameter(gamma)