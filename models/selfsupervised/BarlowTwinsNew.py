import torch
import tensorflow as tf
import numpy as np
import wandb

from models.wandb_utils import save_model_config

# try:
# 	import wandb
# 	from models.wandb_utils import *
# 	wandb_flag = True
# except:
# 	wandb_flag = False
# 	print('Not using W&B')

wandb_flag = False
# Evaluation and Visualization lib.
from models.evaluation.latent_space import *
from models.evaluation.features import *

# Data/Folder Manipulation lib.
from data_manipulation.utils import *
from models.evaluation import *
from models.tools import *
from models.utils import *

# Network related lib.
from models.data_augmentation import *
from models.normalization import *
from models.regularizers import *
from models.activations import *
from models.ops import *

# Losses and Optimizers.
from models.optimizer import *
from models.loss import *
from models.networks import encoder_contrastive

class RepresentationsPathology(torch.nn.Module):
    def __init__(self,
                 data,                                  # Dataset type, training, validation, and test data.
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
        self.relu = torch.nn.ReLU()
        self.model_name = model_name
        self.attention = attention
        self.spectral = spectral
        self.layers = layers
        self.z_dim = z_dim


        ### Input data variables.
        self.image_height   = data.patch_h
        self.image_width    = data.patch_w
        self.image_channels = data.n_channels
        self.batch_size     = data.batch_size

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
        self.learning_rate_e   = learning_rate_e
        self.beta_1            = beta_1
        self.temperature       = temperature # unsure use

        self.num_samples = data.training.images.shape[0]
        all_indx = list(range(self.num_samples))
        random.shuffle(all_indx)
        self.selected_indx = np.array(sorted(all_indx[:10000]))

        self.model_name = model_name
        self.encoder = encoder_contrastive.EncoderResnetContrastive(self.model_name, self.channels, self.z_dim, self.h_dim, self.layers, self.spectral,
                                    self.activation, self.is_train, self.reuse, self.init, self.regularizer,
                                    self.normalization, self.attention, self.down, self.name)


    def forward(self, images):
        # Encoder Training.
        transf_real_images_1, transf_real_images_2 = images
        conv_space_t1, h_rep_t1, z_rep_t1 = self.encoder.forward(transf_real_images_1)
        conv_space_t2, h_rep_t2, z_rep_t2 = self.encoder.forward(transf_real_images_2)
        return [[conv_space_t1, h_rep_t1, z_rep_t1], [conv_space_t2, h_rep_t2, z_rep_t2]]
