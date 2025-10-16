import time
import concurrent.futures


class MyAsyncMapper:
    def __init__(self, max_workers):
        self.max_workers = max_workers
        self.executor = None  # save time, only create executor when WITH starts

    def __enter__(self):
        self.executor = concurrent.futures.ProcessPoolExecutor(max_workers=self.max_workers)
        return self  # return for WITH ... AS ... mapper to assign mapper

    def __exit__(self, exc_type, exc_val, exc_tb):  # even not used, need to keep type val and tb
        if self.executor:
            self.executor.shutdown(wait=True)  # make sure all process finished

    def map(self, func, args):
        futures = []
        if isinstance(args, (tuple, list)):
            for arg in args:
                if isinstance(arg, dict):
                    futures.append(self.executor.submit(func, **arg))
                else:
                    futures.append(self.executor.submit(func, *arg))
        elif isinstance(args, dict):
            futures.append(self.executor.submit(func, **args))
        else:
            raise TypeError("Args need to be list, tuple or dict, and foo() only takes 2 inputs")

        results = []
        for future in concurrent.futures.as_completed(futures):
            try:
                results.append(future.result())
            except Exception as e:
                results.append(e)
        return results


def foo(x, y):
    time.sleep(1)
    return (x + y) / x


if __name__ == "__main__":

    inputs = [(x, x+1) for x in range(10)]

    with MyAsyncMapper(10) as mapper:
        start = time.perf_counter()
        ans = mapper.map(foo, inputs)
        end = time.perf_counter()
        print(f'Results: {ans}')
        print(f'Time taken: {round(end - start, 2)} seconds')
