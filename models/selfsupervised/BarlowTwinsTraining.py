import torch
import tensorflow as tf
import numpy as np
wandb_flag = False

# try:
#     import wandb
#     from models.wandb_utils import *
#     wandb_flag = True
# except:
#     wandb_flag = False
#     print('Not using W&B')

# Evaluation and Visualization lib.
from models.evaluation.latent_space import *
from models.evaluation.features import *

# Data/Folder Manipulation lib.
from torch.utils.data import DataLoader
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

from models.selfsupervised import BarlowTwins
import torch.cuda

class BarlowTwinsTraining():
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

        ### Hyper-parameters.
        self.power_iterations  = power_iterations # unsure use
        self.regularizer_scale = regularizer_scale # unsure use
        self.learning_rate_e   = learning_rate_e
        self.beta_1            = beta_1
        self.temperature       = temperature # unsure use
        self.conv_space_t1 = None
        self.h_rep_t1 = None
        self.z_rep_t1 = None

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

    # Self-supervised inputs.
    def model_inputs(self, data):
        # Image input for transformation.
        print(type(data))
        real_images_1 = torch.tensor(data, dtype=torch.float32)
        real_images_2 = torch.tensor(data, dtype=torch.float32)
        print("data shape")
        print(data.shape)
        # todo: fix shape
        real_images_1 = real_images_1.permute(0,3,1,2)
        real_images_2 = real_images_2.permute(0,3,1,2)
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
            images_trans = tf_wrapper_rb_stain(images_trans)

        # Sobel filter.
        if sobel_filter:
            images_trans = map_func(random_sobel_filter, images_trans)
        # Make sure the image batch is in the right format.
        images_trans = torch.reshape(images_trans, [-1, self.image_channels, self.image_height, self.image_width])
        images_trans = torch.clamp(images_trans, 0.0, 1.0)
        # images_trans = images_trans.permute(0,3,1,2)
        # images_trans.cuda()
        print(images_trans.shape)
        return images_trans




    # Loss Function.
    def loss(self, rep_t1, rep_t2):
        loss = cross_correlation_loss(z_a=rep_t1, z_b=rep_t2, lambda_=self.lambda_)
        return loss

    # Load data into tensors and augment the data
    def data_loading(self, data, device):

        ################### INPUTS & DATA AUGMENTATION #####################################################################################################################################
        # Inputs.
        real_images_1, real_images_2 = self.model_inputs(data)

        # Data augmentation layer.
        real_images_1_t = self.data_augmentation_layer(images=real_images_1, crop=self.crop, rotation=self.rotation, flip=self.flip, g_blur=self.g_blur, g_noise=self.g_noise,
                                                       color_distort=self.color_distort, sobel_filter=self.sobel_filter)
        real_images_2_t = self.data_augmentation_layer(images=real_images_2, crop=self.crop, rotation=self.rotation, flip=self.flip, g_blur=self.g_blur, g_noise=self.g_noise,
                                                       color_distort=self.color_distort, sobel_filter=self.sobel_filter)
        print(real_images_1_t.device, real_images_2_t.device)
        real_images_1_t.to(device)
        real_images_2_t.to(device)
        return real_images_1_t, real_images_2_t


    def project_subsample(self, device, model, data, epoch, data_out_path, report, batch_size=50):
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
        conv_storage = hdf5_file.create_dataset(name='conv_features',    shape=[num_samples] + list(self.conv_space_t1.shape[1:]),     dtype=np.float32) # set these
        h_storage    = hdf5_file.create_dataset(name='h_representation', shape=[num_samples] + list(self.h_rep_t1.shape[1:]),          dtype=np.float32)
        z_storage    = hdf5_file.create_dataset(name='z_representation', shape=[num_samples] + list(self.z_rep_t1.shape[1:]),          dtype=np.float32)

        ind = 0
        while ind<num_samples:
            images_batch = data.training.images[self.selected_indx[ind: ind+batch_size], :, :, :]
            real_images_1_t1, _ = self.data_loading(images_batch, device)

            # Model forward pass
            conv_space_out, h_rep_out, z_rep_out = model.forward(real_images_1_t1, True)

            img_storage[ind: ind+batch_size, :, : ,:]  = images_batch
            conv_storage[ind: ind+batch_size, :] = conv_space_out.detach()
            h_storage[ind: ind+batch_size, :]    = h_rep_out.detach()
            z_storage[ind: ind+batch_size, :]    = z_rep_out.detach()
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
    def training_func(self, epochs, data_out_path, data, restore, print_epochs=10, n_images=25, checkpoint_every=None, report=False): # uses data
        if self.wandb_flag:
            wandb.login(key="c13906296738f8d607f36930faa0617abbc65dc9")
            train_config = save_model_config(self, data)
            run_name = self.model_name + '-' + data.dataset
            wandb.init(project='HPL', name=run_name, config=train_config)

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(device)
        # Set GPU memory management similar to TensorFlow
        torch.backends.cudnn.benchmark = True  # Optimizes performance for fixed input sizes
        torch.backends.cudnn.enabled = True # Enables cuDNN acceleration (if available)
        # Prevent PyTorch from allocating all GPU memory at once (equivalent to TF's allow_growth)
        torch.cuda.empty_cache()  # Clears unused cached memory

        # Create Model
        model = BarlowTwins.RepresentationsPathology(z_dim=self.z_dim, beta_1=self.beta_1, learning_rate_e=self.learning_rate_e, temperature=self.temperature,
                                                     spectral=self.spectral, layers=self.layers, attention=self.attention, power_iterations=self.power_iterations,
                                                     init=self.init, regularizer_scale=self.regularizer_scale, model_name=self.model_name)
        model_params = list(model.parameters())
        model.to(device)

        run_epochs = 0
        torch.save(model, 'model_weights.pth')


        try:
            checkpoints, csvs = setup_output(data_out_path=data_out_path, model_name=self.model_name, restore=restore)
            losses = ['Redundancy Reduction Loss Train', 'Redundancy Reduction Loss Validation']
            setup_csvs(csvs=csvs, model=self, losses=losses)
            report_parameters(self, epochs, restore, data_out_path)

            # Define optimizer and loss
            #  Optimizer
            opt = torch.optim.Adam(params=model_params, lr=self.learning_rate_e, betas=(self.beta_1, 0.999)) #todo: define weight decay
            # Learning rate decay.
            lr_decayed_fn = torch.optim.lr_scheduler.PolynomialLR(optimizer=opt, total_iters=3*self.num_samples,
                                                                  power= 0.5) # todo: look at diff in parameters

            print('Starting run.')

            # Training session
            if self.wandb_flag:
                wandb.watch(model, log_freq=100)

            # Restore previous session.
            if restore:
                check = get_checkpoint(data_out_path)
                model.load_state_dict(torch.load('model_weights.pth', weights_only=True)) # might be wrong
                print('Restored model: %s' % check)


            # Example of augmentation images.
            # batch_images, batch_labels = data.training.next_batch(n_images)
            # data.training.reset()





            train_dataloader = DataLoader(data.training, batch_size=self.batch_size, shuffle=False, num_workers=4, pin_memory=True) # todo: look at shuffle and num workers

            i = 0

            # Epoch Iteration.
            # for epoch in range(0, epochs+1): #todo: uncomment this
            for epoch in range(0, 1):
                # Batch Iteration.
                for batch_images, batch_labels in train_dataloader:
                    # set it to training mode
                    model.train()
                    ################################# TRAINING ENCODER #################################################
                    # Update discriminator.
                    images_1, images_2 = self.data_loading(batch_images, device)

                    # show first transformation of images
                    if epoch == 0:
                        if i == 0:
                            # transform the images # todo: put in if in first loop
                            real_images_1_t1 = images_1.permute(0, 2, 3, 1)
                            real_images_1_t2 = images_2.permute(0, 2, 3, 1)
                            write_sprite_image(filename=os.path.join(data_out_path, 'images/transformed_1.png'),
                                               data=real_images_1_t1, metadata=False)
                            write_sprite_image(filename=os.path.join(data_out_path, 'images/transformed_2.png'),
                                               data=real_images_1_t2, metadata=False)
                            if self.wandb_flag:  # todo: fix me
                                dict_ = {"transformed_1": wandb.Image(
                                    os.path.join(data_out_path, 'images/transformed_1.png')),
                                         "transformed_2": wandb.Image(
                                             os.path.join(data_out_path, 'images/transformed_2.png'))}
                                wandb.log(dict_)
                        i+=1

                    # Model forward pass
                    self.conv_space_t1, self.h_rep_t1, self.z_rep_t1 = model.forward(images_1, True)
                    conv_space_t2, h_rep_t2, z_rep_t2 = model.forward(images_2, True)

                    ################### LOSS & OPTIMIZER ###############################################################
                    # Losses.
                    # Compute loss
                    loss_contrastive = self.loss(rep_t1=self.z_rep_t1, rep_t2=z_rep_t2)
                    # Optimization step
                    opt.zero_grad()
                    loss_contrastive.backward() # compute gradients
                    opt.step() # update params
                    # Learning rate decay step
                    lr_decayed_fn.step()
                    ####################################################################################################
                    # Print losses and Generate samples.
                    # model.eval() # set in validation mode
                    if run_epochs % print_epochs == 0:
                        epoch_outputs = loss_contrastive.item()

                        with torch.no_grad():
                            val_dataloader = DataLoader(data.taining, batch_size=self.batch_size, shuffle=False, num_workers=4, pin_memory=True) # todo: look at shuffle and num workers

                            for batch_images, batch_labels in val_dataloader: # todo: unsure if validation is corrrect
                                    eval_images_1, eval_images_2 = self.data_loading(batch_images, device)

                                    # Model forward pass
                                    self.conv_space_t1, self.h_rep_t1, self.z_rep_t1 =  model.forward(eval_images_1, False) # todo: unsure if false is correct
                                    conv_space_t2, h_rep_t2, z_rep_t2 =  model.forward(eval_images_2, False)

                                    loss_contrastive = self.loss(rep_t1=self.z_rep_t1, rep_t2=z_rep_t2)

                                    val_outputs = loss_contrastive

                                    update_csv(model=self, file=csvs[0], variables=[epoch_outputs, val_outputs], epoch=epoch, iteration=run_epochs, losses=losses)
                                    if self.wandb_flag: wandb.log({'Redundancy Reduction Loss Train': epoch_outputs, 'Redundancy Reduction Loss Validation': val_outputs})
                                    break
                # Save model.
                torch.save(model.state_dict(), 'model_weights.pth')
                data.training.reset()

                ############################### FID TRACKING ##################################################
                # Save checkpoint and generate images for FID every X epochs.
                if (checkpoint_every is not None and epoch % checkpoint_every == 0) or (epochs==epoch):
                    self.project_subsample(device=device, model=model, data=data, epoch=epoch, data_out_path=data_out_path, report=report)



        except RuntimeError as e:
            if "CUDA out of memory" in str(e):
                print("Out of Memory! Reducing batch size may help.")
                torch.cuda.empty_cache()  # Clears unused cached memory
            else:
                raise e  # Re-raise other exceptions
