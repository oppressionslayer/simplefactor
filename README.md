# simplefactor

There are two versoins use either numba or gmpy2. 
The numba version is currently limited to int64 numbers

Added a Cuda version which can factor an int64 number in
about 10 ms on a laptop 3070 Nivida with numba cuda


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

@cuda.jit
def factorise_cuda(N, Nsqrt, result):
    x = cuda.grid(1)
    if x < Nsqrt:
        root = Nsqrt + x
        Nmodcongruence = abs(root**2 - N)
        Nsqrt_mod = int(math.sqrt(Nmodcongruence))  # Simplified for example
        a = root - Nsqrt_mod
        b = N
        while b:
            a, b = b, a % b
        if a != 1 and a != N:  # Check if a is a non-trivial factor
            result[0] = a  # Assuming you want to store the factor found

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
execution_time = timeit.timeit(stmt=test_code, setup=setup_code, number=10, globals=globals()) / 10
print(f"Average execution time: {execution_time} seconds")

#Average execution time: 0.010021003099973313 seconds
```
