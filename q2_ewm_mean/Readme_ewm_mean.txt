>>> How EWMA is calculated in the code?
- I assume that every two consecutive data points within the same group are separated by one unit of time.
- A half-life of n means the influence of a past data point drops by half after n time units.
- (1/2) = alpha ** n → alpha = exp(-ln(2) / n)
- alpha represents how quickly the importance of the current data decays as time passes.
- n is half life
When different weights, we track a weighted running sum (S) and a weighted total weight (W):
- W = alpha * w + (1 - alpha) * W_prev
- S = alpha * (w * x) + (1 - alpha) * S_prev
Then the new EWMA is ( S/W )

>>> How to deal with N/A?
- If there is an N/A in the input data, it is skipped.
- But the passed data importance will still decay as time goes.
- N/A in weights will be fill as 0, meaning skip that data


