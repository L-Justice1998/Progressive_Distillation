#@title Licensed under the Apache License, Version 2.0 (the "License");
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# # Download the diffusion_distillation repository 
# !apt-get -qq install subversion
# !svn checkout https://github.com/google-research/google-research/trunk/diffusion_distillation
# !pip install -r diffusion_distillation/diffusion_distillation/requirements.txt --quiet

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
from diffusion_distillation import diffusion_distillation

# # configure JAX to use the TPU
# if 'TPU_DRIVER_MODE' not in globals():
#   url = 'http://' + os.environ['COLAB_TPU_ADDR'].split(':')[0] + ':8475/requestversion/tpu_driver_nightly'
#   resp = requests.post(url)
#   time.sleep(5)
#   TPU_DRIVER_MODE = 1
# config.FLAGS.jax_xla_backend = "tpu_driver"
# config.FLAGS.jax_backend_target = "grpc://" + os.environ['COLAB_TPU_ADDR']
# print(config.FLAGS.jax_backend_target) 

# create model
config = diffusion_distillation.config.cifar_base.get_config()
model = diffusion_distillation.model.Model(config)

# init params 
state = jax.device_get(model.make_init_state())
state = flax.jax_utils.replicate(state)

# JIT compile training step
train_step = functools.partial(model.step_fn, jax.random.PRNGKey(0), True)
train_step = functools.partial(jax.lax.scan, train_step)  # for substeps
train_step = jax.pmap(train_step, axis_name='batch', donate_argnums=(0,))

# build input pipeline
total_bs = config.train.batch_size
device_bs = total_bs // jax.device_count()
train_ds = model.dataset.get_shuffled_repeated_dataset(
    split='train',
    batch_shape=(
        jax.local_device_count(),  # for pmap
        config.train.substeps,  # for lax.scan over multiple substeps
        device_bs,  # batch size per device
    ),
    local_rng=jax.random.PRNGKey(0),
    augment=True)
train_iter = diffusion_distillation.utils.numpy_iter(train_ds)

# run training
for step in range(10):
  batch = next(train_iter)
  state, metrics = train_step(state, batch)
  metrics = jax.device_get(flax.jax_utils.unreplicate(metrics))
  metrics = jax.tree_map(lambda x: float(x.mean(axis=0)), metrics)
  print(metrics)

# create model
config = diffusion_distillation.config.cifar_distill.get_config()
model = diffusion_distillation.model.Model(config)

# load the teacher params
model.load_teacher_state(config.distillation.teacher_checkpoint_path)

# init student state
init_params = diffusion_distillation.utils.copy_pytree(model.teacher_state.ema_params)
optim = model.make_optimizer_def().create(init_params)
state = diffusion_distillation.model.TrainState(
    step=model.teacher_state.step,
    optimizer=optim,
    ema_params=diffusion_distillation.utils.copy_pytree(init_params),
    num_sample_steps=model.teacher_state.num_sample_steps//2)

# build input pipeline
total_bs = config.train.batch_size
device_bs = total_bs // jax.device_count()
train_ds = model.dataset.get_shuffled_repeated_dataset(
    split='train',
    batch_shape=(
        jax.local_device_count(),  # for pmap
        config.train.substeps,  # for lax.scan over multiple substeps
        device_bs,  # batch size per device
    ),
    local_rng=jax.random.PRNGKey(0),
    augment=True)
train_iter = diffusion_distillation.utils.numpy_iter(train_ds)

steps_per_distill_iter = 10  # number of distillation steps per iteration of progressive distillation
end_num_steps = 4  # eventual number of sampling steps we want to use 
while state.num_sample_steps >= end_num_steps:

  # compile training step
  train_step = functools.partial(model.step_fn, jax.random.PRNGKey(0), True)
  train_step = functools.partial(jax.lax.scan, train_step)  # for substeps
  train_step = jax.pmap(train_step, axis_name='batch', donate_argnums=(0,))

  # train the student against the teacher model
  print('distilling teacher using %d sampling steps into student using %d steps'
        % (model.teacher_state.num_sample_steps, state.num_sample_steps))
  state = flax.jax_utils.replicate(state)
  for step in range(steps_per_distill_iter):
    batch = next(train_iter)
    state, metrics = train_step(state, batch)
    metrics = jax.device_get(flax.jax_utils.unreplicate(metrics))
    metrics = jax.tree_map(lambda x: float(x.mean(axis=0)), metrics)
    print(metrics)

  # student becomes new teacher for next distillation iteration
  model.teacher_state = jax.device_get(
      flax.jax_utils.unreplicate(state).replace(optimizer=None))

  # reset student optimizer for next distillation iteration
  init_params = diffusion_distillation.utils.copy_pytree(model.teacher_state.ema_params)
  optim = model.make_optimizer_def().create(init_params)
  state = diffusion_distillation.model.TrainState(
      step=model.teacher_state.step,
      optimizer=optim,
      ema_params=diffusion_distillation.utils.copy_pytree(init_params),
      num_sample_steps=model.teacher_state.num_sample_steps//2)

# list all available distilled checkpoints
!gsutil ls gs://gresearch/diffusion-distillation

# create imagenet model
config = diffusion_distillation.config.imagenet64_base.get_config()
model = diffusion_distillation.model.Model(config)

# load distilled checkpoint for 8 sampling steps
loaded_params = diffusion_distillation.checkpoints.restore_from_path('gs://gresearch/diffusion-distillation/imagenet_8', target=None)['ema_params']

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
imagenet_classes = {'malamute': 249, 'siamese': 284, 'great_white': 2,
                    'speedboat': 814, 'reef': 973, 'sports_car': 817,
                    'race_car': 751, 'model_t': 661, 'truck': 867}
labels = imagenet_classes['truck'] * jnp.ones((4,), dtype=jnp.int32)
samples = model.samples_fn(rng=jax.random.PRNGKey(0), labels=labels, params=loaded_params, num_steps=8)
samples = jax.device_get(samples).astype(onp.uint8)

# visualize samples
padded_samples = onp.pad(samples, ((0,0), (1,1), (1,1), (0,0)), mode='constant', constant_values=255)
nrows = int(onp.sqrt(padded_samples.shape[0]))
ncols = padded_samples.shape[0]//nrows
_, height, width, channels = padded_samples.shape
img_grid = padded_samples.reshape(nrows, ncols, height, width, channels).swapaxes(1,2).reshape(height*nrows, width*ncols, channels)
img = plt.imshow(img_grid)
plt.axis('off')
