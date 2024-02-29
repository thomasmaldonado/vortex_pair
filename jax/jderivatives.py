### DERIVATIVE DEFINITION ###
# Notes(Alessio): This jax implementation works, 

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
def d1x(f, h, boundary_left, boundary_right):
    return _dx(f, boundary_left, boundary_right, jnp.array([[1/2, 0, -1/2]]))/(h)

@jit
def d2x(f, h, boundary_left, boundary_right):
    return _dx(f, boundary_left, boundary_right, jnp.array([[1, -2, 1]]))/(h**2)

@jit
def d1y(f, h, boundary_left, boundary_right):
    return _dy(f, boundary_left, boundary_right, jnp.array([[1/2, 0, -1/2]]))/(h)

@jit
def d2y(f, h, boundary_left, boundary_right):
    return _dy(f, boundary_left, boundary_right, jnp.array([[1, -2, 1]]))/(h**2)


if __name__ == '__main__':
    import sys 
    sys.path.append('../')
    import derivatives 
    TEST_DERIVATIVES = 1 
    JIT_BEFORE = 1
    import time 
    import numpy as np 
    # Note: it is not surprising for jax to be slower at individually called operations! 
    # This is because calling jax from Python is slow, and this part is not optimized in jax. 
    # This is ok, because in numpy/numba you call numpy from python for every call, whereas in jax you do it once per program

    print("Testing derivatives:")

    @jit
    def compute_test_derivatives(f, h, boundary_left, boundary_right):
        d1_x = d1x(f, h, boundary_left, boundary_right)
        d2_x = d2x(f, h, boundary_left, boundary_right)
        d1_y = d1y(f, h, boundary_left, boundary_right)
        d2_y = d2y(f, h, boundary_left, boundary_right)
        return d1_x, d2_x, d1_y, d2_y 
    
    def test_derivatives(output=True):
        f = jnp.ones((3,3))
        boundary_left = jnp.array([1,2,3])
        boundary_right = jnp.array([4,5,6])
        fnp = np.ones((3,3))
        boundary_left_np = np.array([1,2,3])
        boundary_right_np = np.array([4,5,6])
        h = 0.01
        t = time.time()
        d1_x, d2_x, d1_y, d2_y = compute_test_derivatives(f, h, boundary_left, boundary_right)
        if output: print("Elapsed time for jax functions: ", time.time()-t)

        t = time.time()
        d1_x_og = derivatives.d(fnp, h, 1, boundary_left_np, boundary_right_np, axis=0)
        d2_x_og = derivatives.d(fnp, h, 2, boundary_left_np, boundary_right_np, axis=0)
        d1_y_og = derivatives.d(fnp, h, 1, boundary_left_np, boundary_right_np, axis=1)
        d2_y_og = derivatives.d(fnp, h, 2, boundary_left_np, boundary_right_np, axis=1)
        if output: print("Elapsed time for original functions: ", time.time()-t)

        assert jnp.allclose(d1_x, d1_x_og), 'Failed d1_x'
        assert jnp.allclose(d2_x, d2_x_og), 'Failed d2_x'
        assert jnp.allclose(d1_y, d1_y_og), 'Failed d1_y'
        assert jnp.allclose(d2_y, d2_y_og), 'Failed d2_y'
        if output: print("Passed derivative tests.")

    if JIT_BEFORE:
        test_derivatives(False)
    
    if TEST_DERIVATIVES:
        test_derivatives()

