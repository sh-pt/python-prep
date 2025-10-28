import random
from src import Stream

windows = [5, 1_000, 15_000]
t = 0
data = []

for _ in range(200_000):
    t += random.randint(1,30)
    data.append([t, random.uniform(1,200)])
s = Stream(windows)
for d in data:
    s.add(*d)