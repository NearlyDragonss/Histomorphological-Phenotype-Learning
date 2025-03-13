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
from models.networks.encoder_contrastive import *
from models.data_augmentation import *
from models.normalization import *
from models.regularizers import *
from models.activations import *
from models.ops import *

# Losses and Optimizers.
from models.optimizer import *
from models.loss import *

class RepresentationsPathology(torch.nn.Module):
    def __init__(self,
                 data,                       			# Dataset type, training, validation, and test data.
                 z_dim,	                    			# Latent space dimensions.
                 beta_1,                      			# Beta 1 value for Adam Optimizer.
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
        # self.train()

    # Self-supervised inputs.
    def model_inputs(self, data):
        # Image input for transformation.
        real_images_1 = torch.tensor(data, dtype=torch.float32)
        real_images_2 = torch.tensor(data, dtype=torch.float32)

        return real_images_1, real_images_2,

    # Data Augmentation Layer.
    def data_augmentation_layer(self, images, crop, rotation, flip, g_blur, g_noise, color_distort, sobel_filter):
        images_trans = images
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
        channels = [32, 64, 128, 256, 512, 1024]
        if '_0' in self.model_name: #todo: check this calls .forward
            conv_space, h, z = EncoderResnetContrastive(en_type='_0', images=images, channels=channels, z_dim=self.z_dim, h_dim=1024, layers=self.layers, spectral=self.spectral, activation=ReLU, reuse=reuse, init=init,
                                                          is_train=is_train, normalization=batch_norm, regularizer=None, attention=self.attention, name=name)
        elif '_1' in self.model_name:
            conv_space, h, z = EncoderResnetContrastive(en_type='_1', images=images, channels=channels, z_dim=self.z_dim, h_dim=1024, layers=self.layers, spectral=self.spectral, activation=ReLU, reuse=reuse, init=init,
                                                        is_train=is_train, normalization=batch_norm, regularizer=None, attention=self.attention, name=name)
        elif '_2' in self.model_name:
            conv_space, h, z = EncoderResnetContrastive(en_type='_2', images=images, channels=channels, z_dim=self.z_dim, h_dim=1024, layers=self.layers, spectral=self.spectral, activation=ReLU, reuse=reuse, init=init,
                                                        is_train=is_train, normalization=batch_norm, regularizer=None, attention=self.attention, name=name)
        elif '_3' in self.model_name:
            conv_space, h, z = EncoderResnetContrastive(en_type='_3', images=images, channels=channels, z_dim=self.z_dim, h_dim=1024, layers=self.layers, spectral=self.spectral, activation=ReLU, reuse=reuse, init=init,
                                                        is_train=is_train, normalization=batch_norm, regularizer=None, attention=self.attention, name=name)
        elif '_4' in self.model_name:
            conv_space, h, z = EncoderResnetContrastive(en_type='_4', images=images, channels=channels, z_dim=self.z_dim, h_dim=1024, layers=self.layers, spectral=self.spectral, activation=ReLU, reuse=reuse, init=init,
                                                        is_train=is_train, normalization=batch_norm, regularizer=None, attention=self.attention, name=name)
        elif '_5' in self.model_name:
            conv_space, h, z = EncoderResnetContrastive(en_type='_5', images=images, channels=channels, z_dim=self.z_dim, h_dim=1024, layers=self.layers, spectral=self.spectral, activation=ReLU, reuse=reuse, init=init,
                                                        is_train=is_train, normalization=batch_norm, regularizer=None, attention=self.attention, name=name)
        elif '_6' in self.model_name:
            channels = [64, 128, 256, 512, 1024, 2048]
            conv_space, h, z = EncoderResnetContrastive(en_type='_6', images=images, channels=channels, z_dim=self.z_dim, h_dim=1024, layers=self.layers, spectral=self.spectral, activation=ReLU, reuse=reuse, init=init,
                                                        is_train=is_train, normalization=batch_norm, regularizer=None, attention=self.attention, name=name)
        elif '_7' in self.model_name:
            conv_space, h, z = EncoderResnetContrastive(en_type='_7', images=images, channels=channels, z_dim=self.z_dim, h_dim=1024, layers=self.layers, spectral=self.spectral, activation=ReLU, reuse=reuse, init=init,
                                                        is_train=is_train, normalization=batch_norm, regularizer=None, attention=self.attention, name=name)
        return conv_space, h, z


    # Loss Function.
    def loss(self, rep_t1, rep_t2):
        loss = cross_correlation_loss(z_a=rep_t1, z_b=rep_t2, lambda_=self.lambda_)
        return loss

    # Optimizer.
    # def optimization(self, data_transformed_1, data_transformed_2):
    #     # todo: call in train
    #     # todo: add weight decay to optimizer
    #
    #
    #     # Optimization function
    #     opt = torch.optim.Adam(params=model.parameters(), lr=self.learning_rate_e, betas=(self.beta_1, 0.999)) #todo: get model, define weight decay
    #
    #     # Learning rate decay.
    #     self.lr_decayed_fn = torch.optim.lr_scheduler.PolynomialLR(optimizer=opt, total_iters=3*self.num_samples, power= 0.5) # todo: look at diff in parameters
    #
    #     #  Optimizer
    #     opt.zero_grad()
    #     output_1 = encoder(data_transformed_1) # Forward pass
    #     output_2 = encoder(data_transformed_2)
    #     self.loss_contrastive.backward(output_1, output_2) # compute gradients
    #     opt.step() # update params
    #     self.lr_decayed_fn.step() # update learning rate
    #
    #     return train_encoder

    # Build the Self-supervised.
    def build_model(self, data):
        avail_gpus = [torch.cuda.device(i) for i in range(torch.cuda.device_count())]

        ################### INPUTS & DATA AUGMENTATION #####################################################################################################################################
        # Inputs.
        self.real_images_1, self.real_images_2 = self.model_inputs(data)
        # Data augmentation layer.
        self.real_images_1_t1 = self.data_augmentation_layer(images=self.real_images_1, crop=self.crop, rotation=self.rotation, flip=self.flip, g_blur=self.g_blur, g_noise=self.g_noise,
                                                             color_distort=self.color_distort, sobel_filter=self.sobel_filter)
        self.real_images_2_t2 = self.data_augmentation_layer(images=self.real_images_2, crop=self.crop, rotation=self.rotation, flip=self.flip, g_blur=self.g_blur, g_noise=self.g_noise,
                                                             color_distort=self.color_distort, sobel_filter=self.sobel_filter)
        return self.real_images_1_t, self.real_images_1_t


    def project_subsample(self, data, epoch, data_out_path, report, batch_size=50):
        # todo
        # Updated
        if not report:
            return

        # Handle directories and copies.
        results_path = os.path.join(data_out_path, 'results')
        epoch_path = os.path.join(results_path, 'epoch_%s' % epoch)
        check_epoch_path = os.path.join(epoch_path, 'checkpoints')
        checkpoint_path = os.path.join(results_path, '../checkpoints')
        os.makedirs(epoch_path)
        shutil.copytree(checkpoint_path, check_epoch_path)

        num_samples = 10000

        # Setup HDF5 file.
        hdf5_path = os.path.join(epoch_path, 'hdf5_epoch_%s_projected_images.h5' % epoch)
        hdf5_file = h5py.File(hdf5_path, mode='w')
        img_storage  = hdf5_file.create_dataset(name='images',           shape=[num_samples, data.patch_h, data.patch_w, data.n_channels], dtype=np.float32)
        conv_storage = hdf5_file.create_dataset(name='conv_features',    shape=[num_samples] + self.conv_space_t1.shape.as_list()[1:],     dtype=np.float32)
        h_storage    = hdf5_file.create_dataset(name='h_representation', shape=[num_samples] + self.h_rep_t1.shape.as_list()[1:],          dtype=np.float32)
        z_storage    = hdf5_file.create_dataset(name='z_representation', shape=[num_samples] + self.z_rep_t1.shape.as_list()[1:],          dtype=np.float32)

        ind = 0
        while ind<num_samples:
            images_batch = data.training.images[self.selected_indx[ind: ind+batch_size], :, :, :]
            transf_real_images_1, transf_real_images_2 = self.build_model(images_batch)
            conv_space_out, h_rep_out, z_rep_out = self.encoder(images=transf_real_images_1, is_train=True, reuse=False,
                                                                init=self.init, name='contrastive_encoder')

            img_storage[ind: ind+batch_size, :, : ,:]  = images_batch
            conv_storage[ind: ind+batch_size, :] = conv_space_out
            h_storage[ind: ind+batch_size, :]    = h_rep_out
            z_storage[ind: ind+batch_size, :]    = z_rep_out
            ind += batch_size
        try:
            conv_path, label_conv_path = report_progress_latent(epoch=epoch, w_samples=conv_storage, img_samples=img_storage, img_path=hdf5_path.split('/hdf5')[0], storage_name='conv_lat', metric='euclidean')
            h_path   , label_h_path    = report_progress_latent(epoch=epoch, w_samples=h_storage,    img_samples=img_storage, img_path=hdf5_path.split('/hdf5')[0], storage_name='h_lat',    metric='euclidean')
            z_path   , label_z_path    = report_progress_latent(epoch=epoch, w_samples=z_storage,    img_samples=img_storage, img_path=hdf5_path.split('/hdf5')[0], storage_name='z_lat',    metric='euclidean')
            if self.wandb_flag:
                wandb.log({"Conv Space": wandb.Image(conv_path), "H Space":    wandb.Image(h_path), "Z Space":    wandb.Image(z_path)})
        except Exception as ex:
            print('Issue printing latent space images. Epoch', epoch)
            if hasattr(ex, 'message'):
                print('\t\tException', ex.message)
            else:
                print('\t\tException', ex)
        finally:
            os.remove(hdf5_path)


    # Training function.
    def training(self, epochs, data_out_path, data, restore, print_epochs=10, n_images=25, checkpoint_every=None, report=False): # uses data

        if self.wandb_flag:
            train_config = save_model_config(self, data)
            run_name = self.model_name + '-' + data.dataset
            wandb.init(project='SSL Pathology', entity='adalbertocquiros', name=run_name, config=train_config)

        run_epochs = 0
        torch.save(self.state_dict(), 'model_weights.pth')

        try:
            checkpoints, csvs = setup_output(data_out_path=data_out_path, model_name=self.model_name, restore=restore)
            losses = ['Redundancy Reduction Loss Train', 'Redundancy Reduction Loss Validation']
            setup_csvs(csvs=csvs, model=self, losses=losses)
            report_parameters(self, epochs, restore, data_out_path)

            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            # Set GPU memory management similar to TensorFlow
            torch.backends.cudnn.benchmark = True  # Optimizes performance for fixed input sizes
            torch.backends.cudnn.enabled = True # Enables cuDNN acceleration (if available)
            # Prevent PyTorch from allocating all GPU memory at once (equivalent to TF's allow_growth)
            torch.cuda.empty_cache()  # Clears unused cached memory

            print('Starting run.')

            # Training session
            if self.wandb_flag: wandb.tensorflow.log(tf.summary.merge_all()) # todo: return to

            # Restore previous session.
            if restore:
                check = get_checkpoint(data_out_path)
                self.load_state_dict(torch.load('model_weights.pth', weights_only=True)) # might be wrong
                print('Restored model: %s' % check)


            # Example of augmentation images.
            batch_images, batch_labels = data.training.next_batch(n_images)
            data.training.reset()

            # transform the images
            real_images_1_t1, real_images_1_t2 = self.build_model(batch_images)
            write_sprite_image(filename=os.path.join(data_out_path, 'images/transformed_1.png'), data=real_images_1_t1, metadata=False)
            write_sprite_image(filename=os.path.join(data_out_path, 'images/transformed_2.png'), data=real_images_1_t2, metadata=False)

            if self.wandb_flag:
                dict_ = {"transformed_1": wandb.Image(os.path.join(data_out_path, 'images/transformed_1.png')), "transformed_2": wandb.Image(os.path.join(data_out_path, 'images/transformed_2.png'))}
                wandb.log(dict_)

            # Epoch Iteration.
            for epoch in range(0, epochs+1):

                # Batch Iteration.
                for batch_images, batch_labels in data.training:
                    # set it to training mode
                    self.train()

                    ################################# TRAINING ENCODER #################################################
                    # Update discriminator.
                    transf_real_images_1, transf_real_images_2 = self.build_model(batch_images)
                    #  Optimizer
                    opt = torch.optim.Adam(params=self.parameters(), lr=self.learning_rate_e, betas=(self.beta_1, 0.999)) #todo: define weight decay
                    # Learning rate decay.
                    lr_decayed_fn = torch.optim.lr_scheduler.PolynomialLR(optimizer=opt, total_iters=3*self.num_samples, power= 0.5) # todo: look at diff in parameters
                    # Encoder Training.
                    conv_space_t1, h_rep_t1, z_rep_t1 = self.encoder(images=transf_real_images_1, is_train=True, reuse=False, init=self.init, name='contrastive_encoder')
                    conv_space_t2, h_rep_t2, z_rep_t2 = self.encoder(images=transf_real_images_2, is_train=True, reuse=True,  init=self.init, name='contrastive_encoder')

                    ################### LOSS & OPTIMIZER ##############################################################################################################################
                    # Losses.
                    loss_contrastive = self.loss(rep_t1=z_rep_t1, rep_t2=z_rep_t2)
                    opt.zero_grad()
                    loss_contrastive.backward() # compute gradients
                    opt.step() # update params
                    lr_decayed_fn.step()

                    ####################################################################################################
                    # Print losses and Generate samples.
                    self.eval() # set in validation mode
                    if run_epochs % print_epochs == 0:
                        epoch_outputs = loss_contrastive.item()

                        with torch.no_grad():
                            for batch_images, batch_labels in data.validation: # todo: unsure if validation is corrrect
                                transf_real_images_1, transf_real_images_2 = self.build_model(batch_images)

                                conv_space_t1, h_rep_t1, z_rep_t1 = self.encoder(images=transf_real_images_1, is_train=True, reuse=False, init=self.init, name='contrastive_encoder')
                                conv_space_t2, h_rep_t2, z_rep_t2 = self.encoder(images=transf_real_images_2, is_train=True, reuse=True,  init=self.init, name='contrastive_encoder')

                                loss_contrastive = self.loss(rep_t1=z_rep_t1, rep_t2=z_rep_t2)

                                val_outputs = loss_contrastive


                                update_csv(model=self, file=csvs[0], variables=[epoch_outputs[0], val_outputs[0]], epoch=epoch, iteration=run_epochs, losses=losses)
                                if self.wandb_flag: wandb.log({'Redundancy Reduction Loss Train': epoch_outputs[0], 'Redundancy Reduction Loss Validation': val_outputs[0],})
                                break
                # Save model.
                torch.save(self.state_dict(), 'model_weights.pth')
                data.training.reset()

                ############################### FID TRACKING ##################################################
                # Save checkpoint and generate images for FID every X epochs.
                if (checkpoint_every is not None and epoch % checkpoint_every == 0) or (epochs==epoch):
                    self.project_subsample(data=data, epoch=epoch, data_out_path=data_out_path, report=report)



        except RuntimeError as e:
            if "CUDA out of memory" in str(e):
                print("Out of Memory! Reducing batch size may help.")
                torch.cuda.empty_cache()  # Clears unused cached memory
            else:
                raise e  # Re-raise other exceptions
        if self.wandb_flag:
            wandb.finish()
