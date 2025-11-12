=======================================================================================================================
Custom EWM Mean Project
=======================================================================================================================
This project is the custom version of an EWM (Exponentially Weighted Moving) mean function
The main goal was to make it fast using Cython and make it handle grouping efficiently using factorization


=======================================================================================================================
WHAT IT DOES
=======================================================================================================================
The main function is ewm_mean(input, halflife, group, weight).

input:    A Pandas Series or DataFrame.

halflife: A float for the decay

group:    (Optional) Can be:
- None: Just runs over the whole Series.
- A string: The name of an index level (like "stock").
- A Series: A separate Series to use for grouping.

weight:   (Optional) A Series of weights.


=======================================================================================================================
THE FORMULA
=======================================================================================================================
The code keeps track of a weighted sum (s) and a total weight (w) for every single group.

Alpha Calculation:
alpha = 1.0 - exp(-log(2.0) / halflife)

How each new value is added (for a specific group):
s_prev = (the group's last 's' value)
w_prev = (the group's last 'w' value)

s_now = alpha * current_weight * current_value + (1 - alpha) * s_prev
w_now = alpha * current_weight + (1 - alpha) * w_prev

The output for that row is:
out[i] = s_now / w_now

How NaNs are Handled:
If a value is NaN or its weight is 0, it's skipped. The 's' and 'w' for that group are just decayed by alpha,
and the output for that row is whatever the EWM was before this row.

s_now = s_prev * (1.0 - alpha)
w_now = w_prev * (1.0 - alpha)


=======================================================================================================================
FILE STRUCTURE
=======================================================================================================================
src.py:
main Python file to import from. It has the ewm_mean function.
For DataFrames, it uses threads to run the columns in parallel.

ewm_fast.pyx:
It has the fast for loop doing the math.
The with nogil: part is releasing gil for multi threading

setup.py & pyproject.toml:
Build script.

test_src.py:
pytest unitest


=======================================================================================================================
BUILD AND TEST INSTRUCTIONS
=======================================================================================================================
Need C compiler
Need to install pandas numpy cython pytest
run
pip install -e .
to build the Cython file
and then run the test by
pytest test_src.py