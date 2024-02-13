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
    factors = []  # List to store all factors
    to_factor = [N]  # List of numbers to be factored
    
    while to_factor:
        current_N = to_factor.pop()  # Get the last number to factor
        
        # Check if the number is prime or 1 before trying to factorize it further
        if current_N == 1 or np.all(np.array([current_N]) % np.arange(2, int(isqrt(current_N)) + 1) != 0):
            if current_N != 1:
                factors.append(current_N)
            continue

        result = cuda.device_array(1, dtype=np.int64)
        Nsqrt = isqrt(current_N)
        threadsperblock = 256
        blockspergrid = min(65535, (Nsqrt + (threadsperblock - 1)) // threadsperblock)

        factorise_cuda[blockspergrid, threadsperblock](current_N, Nsqrt, result)
        cuda.synchronize()  # Ensure the kernel has finished
        factor = result.copy_to_host()[0]

        if factor and factor != current_N:
            factors.append(factor)  # Add the found factor to the list
            to_factor.append(factor)  # Add the factor for potential further factoring
            to_factor.append(current_N // factor)  # Add the quotient for potential further factoring
        else:
            # In case the CUDA method fails to find a factor (should not happen with proper implementation), add the number as is.
            factors.append(current_N)

    return np.unique(factors)  # Return the unique factors as a numpy array

# In [9]: factorise(100973253376521343)
# Out[9]: array([247711], dtype=uint64)
