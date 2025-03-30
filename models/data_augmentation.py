from configparser import Interpolation

import torch
import torchvision.transforms
from sympy.parsing.sympy_parser import transformations
from torchvision.transforms import v2, functional as f, InterpolationMode, RandomResizedCrop
from skimage import color, io
import tensorflow as tf
import numpy as np
import functools
import random

from torchvision.transforms.v2 import GaussianBlur


# Transforms elems by applying fn to each element unstacked on axis 0 - copying tf.map_fn functionality
def map_func(func, tensor):
    unstacked_tensor = torch.unbind(tensor, dim=0)
    tensor_to_stack = []
    for slice in unstacked_tensor:
        t = func(slice)
        tensor_to_stack.append(t)
    transformed_tensor = torch.stack(tensor_to_stack, dim=0)
    return transformed_tensor


# Random sampling to recide if the transformation is applied.
def random_apply(func, p, x):
    if torch.less(torch.rand(()), float(p)):
        if func == torch.rot90:
            x = func(x, dims=(1,2))
        else:
            x = func(x)
        return x
    else:
        return x


############### COLOR ###############
# Two version of random change to brightness: Addition/Multiplicative.
def random_brightness(image, max_delta, impl='simclrv2'):
    if impl == 'simclrv2':
        factor = torch.rand(size=()) * (2 * max_delta) + (1 - max_delta)
        image = image * factor
    elif impl == 'simclrv1':
        image = f.adjust_brightness(image, random.randint(-max_delta, max_delta))
    else:
        raise ValueError('Unknown impl {} for random brightness.'.format(impl))
    return image

# Random color transformations: Bright, Contrast, Saturation, and Hue.
def color_jitter_rand(image, brightness=0, contrast=0, saturation=0, hue=0, impl='simclrv2'):
    # with tf.name_scope('distort_color'): # todo:
    def apply_transform(i, x):
        def brightness_foo():
            if brightness == 0:
                return x
            else:
                return random_brightness(x, max_delta=brightness, impl=impl)
        def contrast_foo():
            if contrast == 0:
                return x
            else:
                return f.adjust_contrast(x, random.uniform(1-contrast, 1+contrast))
        def saturation_foo():
            if saturation == 0:
                return x
            else:
                return f.adjust_saturation(x, random.uniform(1-contrast, 1+contrast))
        def hue_foo():
            if hue == 0:
                return x
            else:
                return f.adjust_hue(x, random.uniform(-hue, hue))
        if torch.less(i, 2):
            if torch.less(i, 1):
                x = brightness_foo()
            else:
                x = contrast_foo()
        else:
            if torch.less(i, 3):
                x = saturation_foo()
            else:
                x = hue_foo()
        return x

    perm = torch.randperm(4) # todo: check this # https://stackoverflow.com/questions/44738273/torch-how-to-shuffle-a-tensor-by-its-rows
    for i in range(4):
        image = apply_transform(perm[i], image)
        image = torch.clamp(image, 0.0, 1.0)
    return image

# No random color transformations: 1st Bright, 2nd Contrast, 3rd Saturation, and 4th Hue.
def color_jitter_nonrand(image, brightness=0, contrast=0, saturation=0, hue=0, impl='simclrv2'):
    # with tf.name_scope('distort_color'): # todo: look at
    def apply_transform(i, x, brightness, contrast, saturation, hue):
        if brightness != 0 and i == 0:
            x = random_brightness(x, max_delta=brightness, impl=impl)
        elif contrast != 0 and i == 1:
            x = f.adjust_contrast(x, random.uniform(1-contrast, 1+contrast))
        elif saturation != 0 and i == 2:
            x = f.adjust_saturation(x, random.uniform(1-contrast, 1+contrast))
        elif hue != 0:
            x = f.adjust_hue(x, random.uniform(-hue, hue))
        return x

    for i in range(4):
        image = apply_transform(i, image, brightness, contrast, saturation, hue)
        image = torch.clamp(image, 0., 1.)
    return image

# Color transformation on image: Random or not random order.
def color_jitter(image, strength, random_order=True, impl='simclrv2'):
    brightness = 0.8 * strength
    contrast = 0.8 * strength
    saturation = 0.8 * strength
    hue = 0.2 * strength
    if random_order:
        image = color_jitter_rand(image, brightness, contrast, saturation, hue, impl=impl)
        return image
    else:
        image = color_jitter_nonrand(image, brightness, contrast, saturation, hue, impl=impl)
        return image

# Image RGB to Grayscale.
def to_grayscale(image, keep_channels=True):
    if keep_channels:
        transform = torchvision.transforms.Grayscale(num_output_channels=3)
        transform(image)
    else:
        transform = torchvision.transforms.Grayscale(num_output_channels=1)
        transform(image)
    return image

# Color transformation on image.
def random_color_jitter(image, p=1.0, impl='simclrv2'):
    def transformation(image):
        color_jitter_t = functools.partial(color_jitter, strength=0.5, impl=impl)
        image = random_apply(color_jitter_t, p=0.8, x=image)
        return random_apply(to_grayscale, p=0.2, x=image)
    return random_apply(transformation, p=p, x=image)

# Color transformation on image.
def random_color_jitter_1p0(image, p=1.0, impl='simclrv2'):
    def transformation(image):
        color_jitter_t = functools.partial(color_jitter, strength=1.0, impl=impl)
        image = random_apply(color_jitter_t, p=0.8, x=image)
        image = random_apply(to_grayscale, p=0.2, x=image)
        return image
    return random_apply(transformation, p=p, x=image)


############### SPATIAL: Cropping and Resizing ###############

# Crop.
def distorted_bounding_box_crop(image, bbox, min_object_covered=0.1, aspect_ratio_range=(0.75, 1.33), area_range=(0.05, 1.0), max_attempts=100, scope=None):
    # get dimentions from bbox
    bbox_dimentions = bbox[0,0]
    y_min = bbox_dimentions[0]
    x_min = bbox_dimentions[1]
    y_max = bbox_dimentions[2]
    x_max = bbox_dimentions[3]
    # object_region = image[:, y_min:y_max, x_min:x_max]
    bbox_height = int(y_max-y_min)
    bbox_width = int(x_max - x_min)

    # Resizes and crops based on bbox parameters
    # perm_image = torch.permute(image, (2,0,1))
    transformation = torchvision.transforms.RandomResizedCrop(size=(bbox_height, bbox_width), ratio=aspect_ratio_range, scale=(min_object_covered, 1.0))
    image = transformation(image)
    # image = torch.permute(image, (1,2,0))
    return image

# Crop and resize image.
def crop_and_resize(image, height, width, area_range):

    bbox = torch.tensor([[[0.0, 0.0, 1.0, 1.0]]], dtype=torch.float32)
    aspect_ratio = width / height
    # [224, 224, 3]
    image = distorted_bounding_box_crop(image, bbox, min_object_covered=0.1, aspect_ratio_range=(3./4*aspect_ratio, 4./3.*aspect_ratio), area_range=area_range, max_attempts=100, scope=None)
    # [1,1,3])
    # image = torch.permute(image, (2,0,1))
    image = torch.nn.functional.interpolate(image.unsqueeze(0), size=(height, width), mode="bicubic", align_corners=False)[0]
    # [1, 224, 224]
    return image



# Random crop and resize.
def random_crop_and_resize(image, prob=1.0):
    height, width, channels = list(image.shape)
    def transformation(image):
        images = crop_and_resize(image=image, height=height, width=width, area_range=(0.08, 1.0))
        return images

    return random_apply(func=transformation, p=prob, x=image)


def random_crop_and_resize_p075(image, prob=1.0):
    channels, height, width = list(image.shape)
    def transformation(image):
        images = crop_and_resize(image=image, height=height, width=width, area_range=(0.75, 1.0))
        return images
    x = random_apply(func=transformation, p=prob, x=image)
    return x


# Random crop and resize SwAV Global view.
def random_crop_and_resize_global(image, prob=1.0):
    height, width, channels = image.shape.as_list()
    def transformation(image):
        images = crop_and_resize(image=image, height=height, width=width, area_range=(0.14, 1.0))
        return images
    return random_apply(func=transformation, p=prob, x=image)

# Random crop and resize SwAV Local view.
def random_crop_and_resize_local(image, prob=1.0):
    height, width, channels = image.shape.as_list()
    def transformation(image):
        images = crop_and_resize(image=image, height=height, width=width, area_range=(0.05, 0.14))
        return images
    return random_apply(func=transformation, p=prob, x=image)


############### SPATIAL: Cutout ##############################


############### SPATIAL: Rotation and Flipping ###############

def random_rotate(image, p=0.5):
    return random_apply(torch.rot90, p, image)

def random_flip(image):
    horizontal_flip = v2.RandomHorizontalFlip(p=0.5)
    vertical_flip = v2.RandomVerticalFlip(p=0.5)
    image = horizontal_flip(image)
    image = vertical_flip(image)
    return image


############### BLUR ###############

def gaussian_blur(image, kernel_size, sigma, padding='same'):
    transformations = GaussianBlur(kernel_size=kernel_size, sigma=sigma)
    blurred = transformations(image)


    # radius = int(kernel_size/2)
    # kernel_size = radius * 2 + 1 # number
    # x = float(torch.arange(-radius, radius + 1)) # shape = [1,2,3,4,4,54,45,3]
    # blur_filter = torch.exp(-torch.pow(x, 2.0) / (2.0 * torch.pow(float(sigma), 2.0))) # an exponant number
    # blur_filter /= torch.sum(blur_filter) # a number
    #
    # # One vertical and one horizontal filter.
    # blur_v = torch.reshape(blur_filter, [kernel_size, 1, 1, 1])
    # blur_h = torch.reshape(blur_filter, [1, kernel_size, 1, 1])
    # num_channels = torch.Tensor.size(image)[-1]
    # blur_h = torch.tile(blur_h, [1, 1, num_channels, 1])
    # blur_v = torch.tile(blur_v, [1, 1, num_channels, 1])
    # expand_batch_dim = image.shape.ndims == 3
    # if expand_batch_dim:
    #     # Tensorflow requires batched input to convolutions, which we can fake with
    #     # an extra dimension.
    #     image = image.expand(1, list(image.size())[0]) # todo: could be wrong
    # blurred = torch.nn.functional.conv2d(image, blur_h, stride=(1,1,1,1), padding=padding)
    # blurred = torch.nn.functional.conv2d(blurred, blur_v, stride=(1,1,1,1), padding=padding)
    # if expand_batch_dim:
    #     blurred = torch.squeeze(blurred, dim=0)
    return blurred

def random_blur(image, p=0.5):
    channels, height, width = image.shape.as_list()
    del width
    def _transform(image):
        sigma = (2.0-0.1)*torch.rand(()) + 0.1
        return gaussian_blur(image, kernel_size=height//10, sigma=sigma, padding='same')
    return random_apply(_transform, p=p, x=image)


############### GAUSSIAN NOISE ###############

# Adds gaussian noise to an image.
def add_gaussian_noise(image):
    # image must be scaled in [0, 1]
    # noise =  torch.empty(size=torch.Tensor.size(image), dtype=torch.float32).normal_(mean=0.0,std=(50)/(255))
    #
    #
    # noise_img = image + noise
    # noise_img = torch.clamp(noise_img, 0.0, 1.0)

    translation = v2.GaussianNoise(mean=0.0, sigma=(50)/(255), clip=True)
    noise_img = translation(image)

    return noise_img

# Adds gaussian noise randomly to image.
def random_gaussian_noise(image, p=0.5):
    return random_apply(add_gaussian_noise, p, image)


############### SOBEL FILTER ###############

def random_apply_sobel(func, p, x):
    if torch.less(torch.rand((), dtype=torch.float32), float(p)):
        return torch.mean(func(x), dim=-1)
    else:
        return x

def random_sobel_filter(image, p=0.5):
    height, width, channels = image.shape.as_list()
    image = torch.reshape(image, (1, height, width, channels))
    applied = random_apply_sobel(tf.image.sobel_edges, p, image)
    applied = torch.mean(applied, dim=0)
    return applied


############### DATA AUG WRAPPER ###############
def data_augmentation(images, crop, rotation, flip, g_blur, g_noise, color_distort, sobel_filter, img_size, num_channels):
    images_trans = images
    # Spatial transformations.
    if crop:
        images_trans = tf.map_fn(random_crop_and_resize, images_trans)
    if rotation:
        images_trans = tf.map_fn(random_rotate, images_trans)
    if flip:
        images_trans = random_flip(images_trans)
    # Gaussian blur and noise transformations.
    if g_blur:
        images_trans = tf.map_fn(random_blur, images_trans)
    if g_noise:
        images_trans = tf.map_fn(random_gaussian_noise, images_trans)
    # Color distorsions.
    if color_distort:
        images_trans = tf.map_fn(random_color_jitter, images_trans)
    # Sobel filter.
    if sobel_filter:
        images_trans = tf.map_fn(random_sobel_filter, images_trans)
    # Make sure the image batch is in the right format.
    images_trans = torch.reshape(images_trans, [-1, img_size, img_size, num_channels])
    images_trans = torch.clamp(images_trans, 0., 1.)

    return images_trans


############### DATA AUG WRAPPER INVARIABILITY STAIN COLOR ###############
def get_mean_std_patches(imgs):
    means_ch_0 = list()
    means_ch_1 = list()
    means_ch_2 = list()

    stds_ch_0 = list()
    stds_ch_1 = list()
    stds_ch_2 = list()

    for i in range(imgs.shape[0]):
        if np.max(imgs[i]) <= 1:
            arr = np.array(imgs[i]* 255, dtype=np.uint8)
        else:
            arr = np.array(imgs[i])
        lab = lab = color.rgb2lab(arr)
        means_ch_0.append(np.mean(lab[:,:,0]))
        means_ch_1.append(np.mean(lab[:,:,1]))
        means_ch_2.append(np.mean(lab[:,:,2]))

        stds_ch_0.append(np.std(lab[:,:,0]))
        stds_ch_1.append(np.std(lab[:,:,1]))
        stds_ch_2.append(np.std(lab[:,:,2]))

    return [means_ch_0, means_ch_1, means_ch_2], [stds_ch_0, stds_ch_1, stds_ch_2]

def random_renorm(imgs, means, stds):

    batch_size, height, width, channels = imgs.shape
    processed_img = np.zeros((batch_size, height, width, channels), dtype=np.uint8)

    random_indeces = list(range(batch_size))
    random.shuffle(random_indeces)

    for j in range(batch_size):
        if np.max(imgs[j]) <= 1:
            arr = np.array(imgs[j]* 255, dtype=np.uint8)
        else:
            arr = np.array(imgs[j])
        lab = color.rgb2lab(arr)
        p = random_indeces[j]

        # Each channel
        for i in range(3):

            new_mean = means[i][p]
            new_std  = stds[i][p]

            t_mean = np.mean(lab[:,:,i])
            t_std  = np.std(lab[:,:,i])
            tmp = ( (lab[:,:,i] - t_mean) * (new_std / t_std) ) + new_mean
            if i == 0:
                tmp[tmp<0] = 0
                tmp[tmp>100] = 100
                lab[:,:,i] = tmp
            else:
                tmp[tmp<-128] = 128
                tmp[tmp>127] = 127
                lab[:,:,i] = tmp

        processed_img[j] = (color.lab2rgb(lab) * 255).astype(np.uint8)

    return processed_img/255.

def random_batch_renormalization(batch_images):
    means, stds  = get_mean_std_patches(imgs=batch_images)
    proc_images  = random_renorm(imgs=batch_images, means=means, stds=stds)
    return proc_images

def tf_wrapper_rb_stain(batch_images):
    out = random_batch_renormalization([batch_images.numpy()]) # might need to detach to minimise errors
    out_trans = torch.from_numpy(out)
    return out_trans

def data_augmentation_stain_variability(images, img_size, num_channels):
    images_trans = images
    # images_trans = tf.map_fn(random_crop_and_resize, images_trans)
    # images_trans = tf.map_fn(random_rotate, images_trans)
    # images_trans = random_flip(images_trans)
    images_trans = tf_wrapper_rb_stain(images_trans)

    # Make sure the image batch is in the right format.
    images_trans = torch.reshape(images_trans, [-1, img_size, img_size, num_channels])
    images_trans = torch.clamp(images_trans, 0., 1.)
    return images_trans

def data_augmentation_color(images, img_size, num_channels):
    images_trans = images
    images_trans = tf.map_fn(random_crop_and_resize, images_trans)
    images_trans = tf.map_fn(random_rotate, images_trans)
    images_trans = random_flip(images_trans)
    images_trans = tf.map_fn(random_color_jitter, images_trans)

    # Make sure the image batch is in the right format.
    images_trans = torch.reshape(images_trans, [-1, img_size, img_size, num_channels])
    images_trans = torch.clamp(images_trans, 0., 1.)
    return images_trans