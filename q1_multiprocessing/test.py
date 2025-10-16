import pytest
import time
import concurrent.futures
from src import MyAsyncMapper, foo


# Test correct input as tuple
def test_correct_tuple():
    inputs = [(3, 4)]
    expected = [7/3]
    with MyAsyncMapper(2) as mapper:
        results = mapper.map(foo, inputs)
    assert results == expected, 'Tuple returns correct results'


# Test correct input as list of tuple
def test_correct_list_of_tuple():
    inputs = [(1, 2), (2, 3), (3, 4)]
    expected = [3, 2.5, 7/3]
    with MyAsyncMapper(2) as mapper:
        results = mapper.map(foo, inputs)
    assert results == expected, "List of tuple returns correct results"


# Test correct input as list of mapping
def test_correct_list_of_dict():
    inputs = [{'x': 1, 'y': 2}, {'x': 2, 'y': 3}, {'x': 3, 'y': 4}]
    expected = [3, 2.5, 7 / 3]
    with MyAsyncMapper(2) as mapper:
        results = mapper.map(foo, inputs)
    assert results == expected, "List of dict returns correct results"


# Test correct input as one mapping
def test_correct_one_dict():
    inputs = {'x': 3, 'y': 4}
    expected = [7/3]
    with MyAsyncMapper(2) as mapper:
        results = mapper.map(foo, inputs)
    assert results == expected, "One dict returns correct results"


# Test zero division
def test_zero_division():
    inputs = [(0, 1), (3, 4)]
    with MyAsyncMapper(2) as mapper:
        results = mapper.map(foo, inputs)
    assert isinstance(results[0], ZeroDivisionError), "ZeroDivisionError handle correctly"
    assert results[1] == 7/3, 'Other value handled correctly while ZeroDivisionError'


# Test incorrect input as list of tuple with unmatched args
def test_incorrect_input():
    inputs = [(1, 2, 3), (3, 4), 'string']
    with MyAsyncMapper(2) as mapper:
        results = mapper.map(foo, inputs)
    assert isinstance(results[0], TypeError), "Unmatched input handle correctly"
    assert results[1] == 7/3, 'Other value handled correctly while TypeError'
    assert isinstance(results[2], TypeError), "Wrong type input handle correctly"


# Test multiprocessing should be faster than normal run
def test_speed():
    inputs = [(x, x+1) for x in range(1, 11)]
    normal_start = time.perf_counter()
    r1 = [foo(x, y) for x, y in inputs]
    normal_end = time.perf_counter()
    normal_time = normal_end - normal_start
    mapper_start = time.perf_counter()
    with MyAsyncMapper(10) as mapper:
        results = mapper.map(foo, inputs)
    mapper_end = time.perf_counter()
    mapper_time = mapper_end - mapper_start
    assert mapper_time < normal_time / 2, "Speed test passed"
