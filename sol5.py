# sol5.py


from skimage import color
from skimage import io
from scipy.ndimage.filters import convolve
import numpy as np
from . import sol5_utils


from tensorflow.python.keras import layers, models
from tensorflow.python.keras.optimizers import Adam


GREYSCALE_CODE = 1
MAX_PIXEL_VAL = 255
SUBTRACT_FROM_PATCH = 0.5
TRAIN_VALIDATION_RATIO = 0.8


class Cache:
    """
    a class for caching the images
    """
    def __init__(self):
        """
        constructor
        """
        self.cache = {}
        
    def add_item(self, filename, array):
        """
        adding item to the dictionary - filename:image
        """
        if filename not in self.cache:
            self.cache[filename] = array
            
    def get_cache(self):
        """
        getter for the cache
        """
        return self.cache


def read_image(filename, representation):
    """
    this function reads an image file and returns it in a given representation
    filename is the image
    representation code: 1 is greyscale, 2 is RGB
    returns an image
    """
    final_img = io.imread(filename).astype(np.float64)
    if (representation == GREYSCALE_CODE):
        final_img = color.rgb2gray(final_img)
    final_img /= MAX_PIXEL_VAL
    return final_img.astype(np.float64)


def get_patch_idx(img, crop_size):
    """
    returns random patch indexes from the given image
    """
    rand_row = np.random.randint(0, high=(img.shape[0]-crop_size[0]+1))
    rand_col = np.random.randint(0, high=(img.shape[1]-crop_size[1]+1))
    return rand_row, rand_col


def get_patch_by_idx(img, rand_row, rand_col, crop_size):
    """
    given an image and indexes - returns a patch from it
    """
    return img[rand_row:(rand_row+crop_size[0]), rand_col:(rand_col+crop_size[1])]


def load_dataset(filenames, batch_size, corruption_func, crop_size):
    """
    returns data_generator which is a Generator that outputs random tuples 
    of the form (source_batch, target_batch), when source_batch contains
    corrupted patches and target_batch contains the original clean patches
    """
    cache = Cache()
    height = crop_size[0]
    width = crop_size[1]  
    while (True):
        source_batch = np.zeros((batch_size, height, width, 1))
        target_batch = np.zeros((batch_size, height, width, 1))
        for i in range(batch_size):
            filename_idx = np.random.randint(0, high=len(filenames))
            if (filenames[filename_idx] in cache.get_cache()):
                img = cache.get_cache()[filenames[filename_idx]]
            else:
                img = read_image(filenames[filename_idx], GREYSCALE_CODE)
                cache.add_item(filenames[filename_idx], img)
            rand_row, rand_col = get_patch_idx(img, (height*3, width*3))
            tmp_patch = get_patch_by_idx(img, rand_row, rand_col, crop_size)
            corrupted_tmp_patch = corruption_func(tmp_patch.copy())
            rand_row, rand_col = get_patch_idx(tmp_patch, crop_size)
            source_batch[i,:,:,:] = (get_patch_by_idx(corrupted_tmp_patch, rand_row, rand_col, crop_size)
                                     -SUBTRACT_FROM_PATCH).reshape((height, width, 1))
            target_batch[i,:,:,:] = (get_patch_by_idx(tmp_patch, rand_row, rand_col, crop_size)
                                     -SUBTRACT_FROM_PATCH).reshape((height, width, 1))
        yield (source_batch, target_batch)


def resblock(input_tensor, num_channels):
    """
    returns the basic unit of the ResNet network - the residual block
    """
    conv1 = layers.Conv2D(num_channels, kernel_size=(3, 3), padding='same')(input_tensor)
    act = layers.Activation('relu')(conv1)
    conv2 = layers.Conv2D(num_channels, kernel_size=(3, 3), padding='same')(act)
    add = layers.Add()([conv2, input_tensor])
    return layers.Activation('relu')(add)


def build_nn_model(height, width, num_channels, num_res_blocks):
    """
    returns the complete Neural-Network model
    """
    in1 = layers.Input(shape=(height, width, 1))
    conv1 = layers.Conv2D(num_channels, kernel_size=(3, 3), padding='same')(in1)
    act = layers.Activation('relu')(conv1)
    for i in range(num_res_blocks):
        act = resblock(act, num_channels)
    conv2 = layers.Conv2D(1, kernel_size=(3, 3), padding='same')(act)
    add = layers.Add()([conv2, in1])
    return models.Model(inputs=in1, outputs=add)


def train_model(model, images, corruption_func, batch_size,
                steps_per_epoch, num_epochs, num_valid_samples):
    """
    creates the data-set and trains the model
    """
    images_arr = np.array(images)
    shuffle_idx = np.random.choice(len(images_arr),
                                   np.round(len(images_arr)).astype(np.int), replace=False)
    train_size = np.round(TRAIN_VALIDATION_RATIO*len(images_arr)).astype(np.int)
    train_list = list(images_arr[shuffle_idx][:train_size])
    validation_list = list(images_arr[shuffle_idx][train_size:])
    train_set = load_dataset(train_list, batch_size, corruption_func,
                             model.input_shape[1:3])
    validation_set = load_dataset(validation_list, batch_size, corruption_func,
                                  model.input_shape[1:3])
    model.compile(loss='mean_squared_error', optimizer=Adam(beta_2=0.9))
    model.fit_generator(train_set, steps_per_epoch=steps_per_epoch, epochs=num_epochs,
                        validation_data=validation_set, validation_steps=(num_valid_samples/batch_size))


def restore_image(corrupted_image, base_model):
    """
    restores full size image, not only in the patch size
    """
    height = corrupted_image.shape[0]
    width = corrupted_image.shape[1]
    in1 = layers.Input(shape=(height, width, 1))
    model = base_model(in1)
    new_model = models.Model(inputs=in1, outputs=model)
    corrupted_image -= SUBTRACT_FROM_PATCH
    prediction = new_model.predict(corrupted_image.reshape((1, height, width, 1)))[0]
    prediction += SUBTRACT_FROM_PATCH
    prediction = np.reshape(prediction, corrupted_image.shape)
    return np.clip(prediction, 0, 1).astype(np.float64)


def add_gaussian_noise(image, min_sigma, max_sigma):
    """
    implementation of random gaussian noise
    """
    sigma = np.random.uniform(min_sigma, max_sigma)
    gaussian_random_variable = np.random.normal(0, sigma, image.shape)
    image += gaussian_random_variable
    image = np.around(image * MAX_PIXEL_VAL) / MAX_PIXEL_VAL
    return np.clip(image, 0, 1).astype(np.float64)


def learn_denoising_model(num_res_blocks=5, quick_mode=False):
    """
    returns a trained denoising model
    """
    patch_size = (24, 24)
    num_of_channels = 48
    sigma_range = (0, 0.2)
    corruption_func = lambda im: add_gaussian_noise(im, sigma_range[0], sigma_range[1])
    images_paths = sol5_utils.images_for_denoising()
    batch_size = 100
    steps_per_epoch = 100
    epochs_overall = 5
    num_of_validation_samples = 1000
    if (quick_mode):
        batch_size = 10
        steps_per_epoch = 3
        epochs_overall = 2
        num_of_validation_samples = 30
    model = build_nn_model(patch_size[0], patch_size[1], num_of_channels, num_res_blocks)
    train_model(model, images_paths, corruption_func, batch_size,
                steps_per_epoch, epochs_overall, num_of_validation_samples)
    return model


def add_motion_blur(image, kernel_size, angle):
    """
    simulates motion blur on a given image
    """
    kernel = sol5_utils.motion_blur_kernel(kernel_size, angle)
    return convolve(image, kernel)


def random_motion_blur(image, list_of_kernel_sizes):
    """
    simulates random motion blur on a given image
    """
    angle = np.random.uniform(0, np.pi)
    kernel_idx = np.random.randint(0, len(list_of_kernel_sizes))
    kernel_size = list_of_kernel_sizes[kernel_idx]
    image = add_motion_blur(image, kernel_size, angle)
    image = np.around(image * MAX_PIXEL_VAL) / MAX_PIXEL_VAL
    return np.clip(image, 0, 1).astype(np.float64)


def learn_deblurring_model(num_res_blocks=5, quick_mode=False):
    """
    returns a trained deblurring model
    """
    images_paths = sol5_utils.images_for_deblurring()
    patch_size = (16, 16)
    num_of_channels = 32
    kernel_size = 7
    corruption_func = lambda im: random_motion_blur(im, [kernel_size])
    batch_size = 100
    steps_per_epoch = 100
    epochs_overall = 10
    num_of_validation_samples = 1000
    if (quick_mode):
        batch_size = 10
        steps_per_epoch = 3
        epochs_overall = 2
        num_of_validation_samples = 30
    model = build_nn_model(patch_size[0], patch_size[1], num_of_channels, num_res_blocks)
    train_model(model, images_paths, corruption_func, batch_size,
                steps_per_epoch, epochs_overall, num_of_validation_samples)
    return model
