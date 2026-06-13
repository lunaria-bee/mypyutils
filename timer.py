'''Code timing utilities.'''

from contextlib import contextmanager
import time


class Timer:
    '''Class providing a context manager for measuring code execution time.'''

    def __init__(self):
        self.process_times: list[float] = []
        self.thread_times: list[float] = []
        self.clock_times: list[float] = []

    @contextmanager
    def time(self):
        '''Context manager that will measure execution time of the managed
        context, then save that time to ``times``.'''
        process_start = time.process_time()
        thread_start = time.thread_time()
        clock_start = time.time()
        try:
            yield
        finally:
            self.process_times.append(time.process_time() - process_start)
            self.thread_times.append(time.thread_time() - thread_start)
            self.clock_times.append(time.time() - clock_start)
