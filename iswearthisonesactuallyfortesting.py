from statistics import stdev
from random import randint

x = []
for i in range(100000):
    i = randint(0,1)
    x.append(i)

print(stdev(x))