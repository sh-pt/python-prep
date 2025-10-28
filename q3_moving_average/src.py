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

        records = self.records
        header = self.header
        acc = self.acc
        cnt = self.cnt
        windows = self.windows
        nw = len(windows)

        records.append([time, price])

        for i in range(nw):
            cnt[i] += 1
            acc[i] += price

        nrec = len(records)

        for i in range(nw):
            cutoff = time - windows[i]
            idx = header[i]
            while idx < nrec and records[idx][0] < cutoff:
                acc[i] -= records[idx][1]
                cnt[i] -= 1
                idx += 1
            header[i] = idx

        if nrec >= 50_000:
            first_header = min(header)
            new_records = records[first_header:]
            for i in range(nw):
                header[i] -= first_header
            self.records = new_records

        return [(acc[i] / cnt[i]) if cnt[i] else float('nan') for i in range(nw)]


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




