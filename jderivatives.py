### DERIVATIVE DEFINITION ###

import jax.numpy as jnp 
from jax import jit 
import jax.scipy as jscipy
import jax
jax.config.update('jax_platform_name', 'cpu')

@jit
def _apply_boundaries(f, boundary_left, boundary_right):
    bl, br = jnp.reshape(boundary_left, (f.shape[0], 1)), jnp.reshape(boundary_right, (f.shape[0], 1))
    return jnp.concatenate((bl, f, br), axis = 1)

def _d(f, boundary_left, boundary_right, ker, axis):
    if axis == 0:
        fp = _apply_boundaries(f, boundary_left, boundary_right)
        return jscipy.signal.convolve2d(fp, ker, mode='valid')
    elif axis == 1:    
        fp = _apply_boundaries(f.T, boundary_left, boundary_right)
        return jscipy.signal.convolve2d(fp, ker, mode='valid').T

@jit
def _dy(f, boundary_left, boundary_right, ker):
    return _d(f, boundary_left, boundary_right, ker, axis=0)

@jit
def _dx(f, boundary_left, boundary_right, ker):
    return _d(f, boundary_left, boundary_right, ker, axis=1)

@jit 
def d_dx1(f, h, boundary_left, boundary_right):
    return _dx(f, boundary_left, boundary_right, jnp.array([[1/2, 0, -1/2]]))/(h)

@jit
def d_dx2(f, h, boundary_left, boundary_right):
    return _dx(f, boundary_left, boundary_right, jnp.array([[1, -2, 1]]))/(h**2)

@jit
def d_dy1(f, h, boundary_left, boundary_right):
    return _dy(f, boundary_left, boundary_right, jnp.array([[1/2, 0, -1/2]]))/(h)

@jit
def d_dy2(f, h, boundary_left, boundary_right):
    return _dy(f, boundary_left, boundary_right, jnp.array([[1, -2, 1]]))/(h**2)