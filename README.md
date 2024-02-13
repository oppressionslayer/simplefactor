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
We can factor not in just 500 ms but in 279 usec!

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
    factors = []
    to_factor = [N]

    while to_factor:
        current_N = to_factor.pop()

        if current_N == 1 or np.all(np.array([current_N]) % np.arange(2, int(isqrt(current_N)) + 1) != 0):
            if current_N != 1:
                factors.append(current_N)
            continue

        result = cuda.device_array(1, dtype=np.uint64)
        Nsqrt = isqrt(current_N)
        threadsperblock = 256
        blockspergrid = (Nsqrt + (threadsperblock - 1)) // threadsperblock

        factorise_cuda[blockspergrid, threadsperblock](current_N, Nsqrt, result)
        cuda.synchronize()
        factor = int(result.copy_to_host()[0])  # Explicit casting to int

        if factor and factor != current_N:
            factors.extend([factor, current_N // factor])
        else:
            factors.append(current_N)

    return np.array(np.unique(factors), dtype=np.int64)  # Ensure output is integer type



# Preparation for execution
N = 1009732533765211
result = cuda.device_array(1, dtype=np.int64)
Nsqrt = isqrt(N)
threadsperblock = 8
blockspergrid = min(64**2-1, (Nsqrt + (threadsperblock - 1)) // threadsperblock)
"""

test_code = """
factorise_cuda[blockspergrid, threadsperblock](N, Nsqrt, result)
cuda.synchronize()  # Wait for the kernel to complete
result_cpu = result.copy_to_host()
"""

# Measure execution time
execution_time_us = timeit.timeit(stmt=test_code, setup=setup_code, number=100, globals=globals()) / 100 * 1_000_000  # Convert to microseconds
print(f"Average execution time: {execution_time_us} μs")

# Average execution time: 279.54018999935215 μs

```
