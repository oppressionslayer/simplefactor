# simplefactor

There are two versoins use either numba or gmpy2. 
The numba version is currently limited to int64 numbers

Added a Cuda version which can factor an int64 number in
about 15 ms on a laptop 3070 Nivida with numba cuda. This
is currently optimized for odd factor factorization


```
from simplefactor import factorise
In [16]: factorise(1009732533765211)
Out[16]: mpz(11344301)

from simplenumbafactor import factorise
In [16]: factorise(1009732533765211)
Out[16]: mpz(11344301)

from simplegmpy2factor import factorise
In [16]: factorise(1009732533765211)
Out[16]: mpz(11344301)
```
# CUDA TIMING PERFORMANCE

If you are just interested in CUDA Factoring Performance, check out this test code

```
import timeit

setup_code = """
from numba import cuda, njit
import numpy as np
import math

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
            factors.append(factor)  
            to_factor.append(factor)  
            to_factor.append(current_N // factor)  
        else:

            factors.append(current_N)

    return np.unique(factors)  # Return the unique factors as a numpy array

# Preparation for execution
N = 1009732533765211
result = cuda.device_array(1, dtype=np.int64)
Nsqrt = isqrt(N)
threadsperblock = 256
blockspergrid = min(65535, (Nsqrt + (threadsperblock - 1)) // threadsperblock)
"""

test_code = """
factorise_cuda[blockspergrid, threadsperblock](N, Nsqrt, result)
cuda.synchronize()  # Wait for the kernel to complete
result_cpu = result.copy_to_host()
"""

# Measure execution time
execution_time = timeit.timeit(stmt=test_code, setup=setup_code, number=100, globals=globals()) / 100
print(f"Average execution time: {execution_time} seconds")

# Average execution time: 0.01580274557100006 seconds

```
