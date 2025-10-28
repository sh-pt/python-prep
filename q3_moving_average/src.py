# A class will take in price data and output current moving average
# input is [time, price]
# output is [avg1, avg2, ... ] moving average for required time length

# import numpy as np
# from collections import deque

try:
    profile
except NameError:
    def profile(func):
        return func

class Stream:
    def __init__(self, windows:list):
        self.windows = windows
        self.acc = [0.0] * len(windows) # use for pre sum for each window size
        self.cnt = [0] * len(windows)
        self.records = []
        self.header = [0] * len(windows)

    @profile
    def add(self, time, price):

        self.records.append([time, price])

        for i in range(len(self.windows)):
            self.cnt[i] += 1
            self.acc[i] += price

        for i in range(len(self.windows)):
            while self.records[self.header[i]][0] < time - self.windows[i]:
                self.acc[i] -= self.records[self.header[i]][1]
                self.cnt[i] -= 1
                self.header[i] += 1

        first_header = min(self.header)
        self.records = self.records[first_header:]

        for i in range(len(self.windows)):
            self.header[i] -= first_header

        return [self.acc[i] / self.cnt[i] for i in range(len(self.cnt))]


if __name__ == '__main__':

    windows = [5,10,15]

    data = [[1,10],
            [3,20],
            [6,30],
            [9,40],
            [10,50],
            [14,60],
            [15,70],
            [20,80],
            [21,90],
            [25,100],
            [30,110],
            [44,100],
            ]

    Stream = Stream(windows = windows)
    for d in data:
        print(Stream.add(*d))




