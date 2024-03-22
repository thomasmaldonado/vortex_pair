### DERIVATIVE DEFINITION ###

import jax.numpy as jnp 
from jax import jit 
import jax.scipy as jscipy
import jax
#jax.config.update('jax_platform_name', 'cpu')

@jit
def _apply_boundaries(f, boundary_left, boundary_right):
    return jnp.pad(f, (1, 1), constant_values = (boundary_left, boundary_right))

@jit
def _dx(f, boundary_left, boundary_right, ker):
    f_pad = jnp.pad(f, (1, 1), constant_values = (boundary_left, boundary_right))

@jit
def _dx(f, boundary_left, boundary_right, ker):
    fp = _apply_boundaries(f, boundary_left, boundary_right)
    return jscipy.signal.convolve(fp, ker, mode='valid')

@jit 
def d_dx1(f, h, boundary_left, boundary_right):
    return _dx(f, boundary_left, boundary_right, jnp.array([1/2, 0, -1/2]))/(h)

@jit
def d_dx2(f, h, boundary_left, boundary_right):
    return _dx(f, boundary_left, boundary_right, jnp.array([1, -2, 1]))/(h**2)