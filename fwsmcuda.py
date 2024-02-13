# This uses cuda for factorizing numbers shorter than in64 as this is the current
# limit of numba. The exception thing is that this factors those numbers in 10ms
# rougly 45-50x faster on a GTX 3070 laptop.

# To use: from fwsmcuda import factorise

from numba import cuda, njit
import numpy as np

@njit    
def isqrt(n):
    if n > 0:
        x = n
        y = (x + 1) // 2
        while y < x:
            x = y
            y = (x + n // x) // 2
        return x
    elif n == 0:
        return 0
    else:
        raise ValueError("square root not defined for negative numbers")
        
@cuda.jit
def factorise_cuda(N, Nsqrt, result):
    x = cuda.grid(1)
    if x < Nsqrt:
        root = Nsqrt + x
        Nmodcongruence = abs(root**2 - N)

        Nsqrt_mod = int(isqrt(Nmodcongruence))  
        a = root - Nsqrt_mod
        b = N
        while b:
            a, b = b, a % b
        if a != 1 and a != N:  # Check if a is a non-trivial factor
            result[0] = a  

def factorise(N):

  result = cuda.device_array(1, dtype=np.uint64)  # Using uint64 for larger numbers
  Nsqrt = isqrt(N)
  threadsperblock = 256
  # Not sure what block size to us but it's adjustable
  blockspergrid = min(65535, (Nsqrt + (threadsperblock - 1)) // threadsperblock)

  factorise_cuda[blockspergrid, threadsperblock](N, Nsqrt, result)

  result_cpu = result.copy_to_host()
  return result_cpu

# In [9]: factorise(100973253376521343)
# Out[9]: array([247711], dtype=uint64)
