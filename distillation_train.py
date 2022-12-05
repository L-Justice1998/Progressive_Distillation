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
    diffusion_distillation.checkpoints.save_checkpoint('./checkpoints',model.teacher_state, state.num_sample_steps)
    # reset student optimizer for next distillation iteration
    init_params = diffusion_distillation.utils.copy_pytree(model.teacher_state.ema_params)
    optim = model.make_optimizer_def().create(init_params)
    state = diffusion_distillation.model.TrainState(
        step=model.teacher_state.step,
        optimizer=optim,
        ema_params=diffusion_distillation.utils.copy_pytree(init_params),
        num_sample_steps=model.teacher_state.num_sample_steps//2)
    
