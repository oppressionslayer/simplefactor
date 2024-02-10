def factorise(N):
  from sympy import isprime

  if N%2 == 0:
    return 2
  Nsqrt = math.isqrt(N)
  x=1
  while True and x < Nsqrt:
    Nmodcongruence = get_mod_congruence((Nsqrt+x)**2, N)
    Ngcd = math.gcd(Nsqrt+x-math.isqrt(Nmodcongruence), N)
    if Ngcd != 1:
      break;
    x+=1
  if not isprime(Ngcd):
    print(f"Factor is not prime, you can factor the number {Ngcd} again")
  return Ngcd

# In [13]: factorise(1009732533765211)
# Out[13]: mpz(11344301)
