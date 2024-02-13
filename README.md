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
