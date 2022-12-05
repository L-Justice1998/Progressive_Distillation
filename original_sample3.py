import os
import time
import requests
import functools
import jax
from jax.config import config
import jax.numpy as jnp
import flax
from matplotlib import pyplot as plt
import numpy as onp
import tensorflow.compat.v2 as tf
tf.enable_v2_behavior()
import diffusion_distillation
# from evaluate.fid_score import calculate_fid_given_paths

#sample
# create cifar model
config = diffusion_distillation.config.cifar_distill.get_config()
model = diffusion_distillation.model.Model(config)

# load original model 
loaded_params = diffusion_distillation.checkpoints.restore_from_path('./checkpoints/diffusion-distillation_cifar_original', target=None)['ema_params']

# fix possible flax version errors
ema_params = jax.device_get(model.make_init_state()).ema_params
loaded_params = flax.core.unfreeze(loaded_params)
loaded_params = jax.tree_map(
    lambda x, y: onp.reshape(x, y.shape) if hasattr(y, 'shape') else x,
    loaded_params,
    flax.core.unfreeze(ema_params))
loaded_params = flax.core.freeze(loaded_params)
del ema_params

# sample from the model
# imagenet_classes = {'malamute': 249, 'siamese': 284, 'great_white': 2,
#                     'speedboat': 814, 'reef': 973, 'sports_car': 817,
#                     'race_car': 751, 'model_t': 661, 'truck': 867}
# labels = imagenet_classes['truck'] * jnp.ones((4,), dtype=jnp.int32)
for i in range(41710,50000):
    samples = model.samples_fn(rng=jax.random.PRNGKey(i), params=loaded_params, num_steps=8192)
    samples = jax.device_get(samples).astype(onp.uint8)

    # visualize samples
    padded_samples = onp.pad(samples, ((0,0), (1,1), (1,1), (0,0)), mode='constant', constant_values=255)
    nrows = int(onp.sqrt(padded_samples.shape[0]))
    ncols = padded_samples.shape[0]//nrows
    _, height, width, channels = padded_samples.shape
    img_grid = padded_samples.reshape(nrows, ncols, height, width, channels).swapaxes(1,2).reshape(height*nrows, width*ncols, channels)
    img = plt.imsave('./cifar10_original_generation/'+ (str)(i) +'.png',img_grid)

# fid_stats_dir="fid_stats/fid_stats_cifar10_train_pytorch.npz"
# fid = calculate_fid_given_paths((fid_stats_dir,  ), 
#                                 batch_size=64, device='cuda:0', dims=2048, num_workers=8)
