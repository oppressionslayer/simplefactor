# if you have numba use: 
# from simplenumberfactor factorise

import numba

@nb.njit
def get_mod_congruence(root, N):
    return root - N

@nb.njit
def gcd(a, b):
    while b:
        a, b = b, a % b
    return a

@nb.njit
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

@nb.njit
def factorise(N):
    if N % 2 == 0:
        return 2
    Nsqrt = isqrt(N)
    x = 1
    while True and x < Nsqrt:
        root = Nsqrt + x
        Nmodcongruence = get_mod_congruence(root**2, N)
        Nsqrt_mod = isqrt(Nmodcongruence)
        Ngcd = gcd(root - Nsqrt_mod, N)
        if Ngcd != 1 and Ngcd != N:  # Ensure non-trivial factor
            break
        x += 1
    return Ngcd
