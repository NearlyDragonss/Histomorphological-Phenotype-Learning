import torch
import tensorflow as tf
import numpy as np
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
from models.networks.encoder_contrastive import *
from models.data_augmentation import *
from models.normalization import *
from models.regularizers import *
from models.activations import *
from models.ops import *

# Losses and Optimizers.
from models.optimizer import *
from models.loss import *

class RepresentationsPathology():
    def __init__(self,
                 data,                       			# Dataset type, training, validation, and test data.
                 z_dim,	                    			# Latent space dimensions.
                 beta_1,                      			# Beta 1 value for Adam Optimizer.
                 learning_rate_e,             			# Learning rate Encoder.
                 lambda_=5e-3,							# Lambda weight for redundant representation penalty.
                 temperature=0.1,                        # Temperature for contrastive cosine similarity norm.
                 spectral=True,							# Spectral Normalization for weights.
                 layers=5,					 			# Number for layers for Encoder.
                 attention=28,                			# Attention Layer dimensions, default after height and width equal 28 to pixels.
                 power_iterations=1,          			# Iterations of the power iterative method: Calculation of Eigenvalues, Singular Values.
                 init = 'orthogonal',    			    # Weight Initialization: default Orthogonal.
                 regularizer_scale=1e-4,      			# Orthogonal regularization.
                 model_name='RepresentationsPathology'   # Model Name.
                 ):

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
        self.init      = init

        ### Hyper-parameters.
        self.power_iterations  = power_iterations
        self.regularizer_scale = regularizer_scale
        self.learning_rate_e   = learning_rate_e
        self.beta_1            = beta_1
        self.temperature       = temperature

        ### Data augmentation conditions.
        # Spatial transformation.
        self.crop          = True
        self.rotation      = True
        self.flip          = True
        # Color transformation.
        self.color_distort = True
        # Gaussian Blur and Noise.
        self.g_blur        = False
        self.g_noise       = False
        # Sobel Filter.
        self.sobel_filter  = False

        self.lambda_ = lambda_

        self.num_samples = data.training.images.shape[0]
        all_indx = list(range(self.num_samples))
        random.shuffle(all_indx)
        self.selected_indx = np.array(sorted(all_indx[:10000]))

        self.model_name = model_name

        self.wandb_flag = wandb_flag
        self.build_model()

    # Self-supervised inputs.
    def model_inputs(self, data):
        # todo: maybe specify device and look at requires_grad()

        # Image input for transformation.
        real_images_1 = torch.tensor(data, dtype=torch.float32)
        real_images_2 = torch.tensor(data, dtype=torch.float32)

        transf_real_images_1 = real_images_1
        transf_real_images_2 = real_images_2
        # Learning rates.
        # todo: return to learning rate

        return real_images_1, real_images_2, transf_real_images_1, transf_real_images_2

    # Data Augmentation Layer.
    def data_augmentation_layer(self, images, crop, rotation, flip, g_blur, g_noise, color_distort, sobel_filter):
        images_trans = images
        # todo: check these functions work correctly in data_augmentation.py
        # Spatial transformations.
        if crop:
            images_trans = map_func(random_crop_and_resize_p075, images_trans)
        if rotation:
            images_trans = map_func(random_rotate, images_trans)
        if flip:
            images_trans = random_flip(images_trans)

        # Gaussian blur and noise transformations.
        if g_blur:
            images_trans = map_func(random_blur, images_trans)
        if g_noise:
            images_trans = map_func(random_gaussian_noise, images_trans)

        # Color distorsions.
        if color_distort:
            images_trans = map_func(random_color_jitter_1p0, images_trans)
        else:
            images_trans = tf_wrapper_rb_stain(images_trans) # revisit

        # Sobel filter.
        if sobel_filter:
            images_trans = map_func(random_sobel_filter, images_trans)

        # Make sure the image batch is in the right format.
        images_trans = torch.reshape(images_trans, [-1, self.image_height, self.image_width, self.image_channels])
        images_trans = torch.clamp(images_trans, 0., 1.)

        return images_trans

    # Encoder Network.
    def encoder(self, images, is_train, reuse, init, name, label_input=None):
        # todo: translate encoder
        if '_0' in self.model_name:
            conv_space, h, z = encoder_resnet_contrastive(images=images, z_dim=self.z_dim, h_dim=1024, layers=self.layers, spectral=self.spectral, activation=ReLU, reuse=reuse, init=init,
                                                          is_train=is_train, normalization=batch_norm, regularizer=None, attention=self.attention, name=name)
        elif '_1' in self.model_name:
            conv_space, h, z = encoder_resnet_contrastive_1(images=images, z_dim=self.z_dim, h_dim=1024, layers=self.layers, spectral=self.spectral, activation=ReLU, reuse=reuse, init=init,
                                                        is_train=is_train, normalization=batch_norm, regularizer=None, attention=self.attention, name=name)
        elif '_2' in self.model_name:
            conv_space, h, z = encoder_resnet_contrastive_2(images=images, z_dim=self.z_dim, h_dim=1024, layers=self.layers, spectral=self.spectral, activation=ReLU, reuse=reuse, init=init,
                                                        is_train=is_train, normalization=batch_norm, regularizer=None, attention=self.attention, name=name)
        elif '_3' in self.model_name:
            conv_space, h, z = encoder_resnet_contrastive_3(images=images, z_dim=self.z_dim, h_dim=1024, layers=self.layers, spectral=self.spectral, activation=ReLU, reuse=reuse, init=init,
                                                        is_train=is_train, normalization=batch_norm, regularizer=None, attention=self.attention, name=name)
        elif '_4' in self.model_name:
            conv_space, h, z = encoder_resnet_contrastive_4(images=images, z_dim=self.z_dim, h_dim=1024, layers=self.layers, spectral=self.spectral, activation=ReLU, reuse=reuse, init=init,
                                                        is_train=is_train, normalization=batch_norm, regularizer=None, attention=self.attention, name=name)
        elif '_5' in self.model_name:
            conv_space, h, z = encoder_resnet_contrastive_5(images=images, z_dim=self.z_dim, h_dim=1024, layers=self.layers, spectral=self.spectral, activation=ReLU, reuse=reuse, init=init,
                                                        is_train=is_train, normalization=batch_norm, regularizer=None, attention=self.attention, name=name)
        elif '_6' in self.model_name:
            conv_space, h, z = encoder_resnet_contrastive_6(images=images, z_dim=self.z_dim, h_dim=1024, layers=self.layers, spectral=self.spectral, activation=ReLU, reuse=reuse, init=init,
                                                        is_train=is_train, normalization=batch_norm, regularizer=None, attention=self.attention, name=name)
        elif '_7' in self.model_name:
            conv_space, h, z = encoder_resnet_contrastive_7(images=images, z_dim=self.z_dim, h_dim=1024, layers=self.layers, spectral=self.spectral, activation=ReLU, reuse=reuse, init=init,
                                                        is_train=is_train, normalization=batch_norm, regularizer=None, attention=self.attention, name=name)
        return conv_space, h, z


    # Loss Function.
    def loss(self, rep_t1, rep_t2):
        loss = cross_correlation_loss(z_a=rep_t1, z_b=rep_t2, lambda_=self.lambda_) #revisit
        return loss

    # Optimizer.
    def optimization(self):
        pass

    # Build the Self-supervised.
    def build_model(self, data):
        avail_gpus = [torch.cuda.device(i) for i in range(torch.cuda.device_count())]

        ################### INPUTS & DATA AUGMENTATION #####################################################################################################################################
        # Inputs.
        self.real_images_1, self.real_images_2, self.transf_real_images_1, self.transf_real_images_2 = self.model_inputs(data)
        # Data augmentation layer.
        self.real_images_1_t1 = self.data_augmentation_layer(images=self.real_images_1, crop=self.crop, rotation=self.rotation, flip=self.flip, g_blur=self.g_blur, g_noise=self.g_noise,
                                                             color_distort=self.color_distort, sobel_filter=self.sobel_filter)
        self.real_images_1_t2 = self.data_augmentation_layer(images=self.real_images_1, crop=self.crop, rotation=self.rotation, flip=self.flip, g_blur=self.g_blur, g_noise=self.g_noise,
                                                             color_distort=self.color_distort, sobel_filter=self.sobel_filter)
        ################### TRAINING #####################################################################################################################################
        # Encoder Training.
        self.conv_space_t1, self.h_rep_t1, self.z_rep_t1 = self.encoder(images=self.transf_real_images_1, is_train=True, reuse=False, init=self.init, name='contrastive_encoder')
        self.conv_space_t2, self.h_rep_t2, self.z_rep_t2 = self.encoder(images=self.transf_real_images_2, is_train=True, reuse=True,  init=self.init, name='contrastive_encoder')

        ################### LOSS & OPTIMIZER ##############################################################################################################################
        # Losses.
        self.loss_contrastive = self.loss(rep_t1=self.z_rep_t1, rep_t2=self.z_rep_t2)
        # Optimizers.
        self.train_encoder  = self.optimization()

    pass

    def project_subsample(self, session, data, epoch, data_out_path, report, batch_size=50): # uses data
        pass


    # Training function.
    def train(self, epochs, data_out_path, data, restore, print_epochs=10, n_images=25, checkpoint_every=None, report=False): # uses data
        if self.wandb_flag:
            train_config = save_model_config(self, data)
            run_name = self.model_name + '-' + data.dataset
            wandb.init(project='SSL Pathology', entity='adalbertocquiros', name=run_name, config=train_config)

        run_epochs = 0
        checkpoints, csvs = setup_output(data_out_path=data_out_path, model_name=self.model_name, restore=restore)
        losses = ['Redundancy Reduction Loss Train', 'Redundancy Reduction Loss Validation']
        setup_csvs(csvs=csvs, model=self, losses=losses)
        report_parameters(self, epochs, restore, data_out_path)

        # todo: ensure it lets an operation be placed on the cpu if no gpu available, allow full use of memory
        print('Starting run.')

        ## todo: find a way to restore progress

        batch_images, batch_labels = data.training.next_batch(n_images)
        data.training.reset()
        feed_dict = {self.real_images_1:batch_images, self.real_images_2:batch_images}
        output_layers_transformed = [self.real_images_1_t1, self.real_images_1_t2]

        build_model(batch_images)



        pass
