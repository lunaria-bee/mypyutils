'''Code timing utilities.'''

from contextlib import contextmanager
import time


class Timer:
    '''Class providing a context manager for measuring code execution time.'''

    def __init__(self, timefn=time.process_time):
        self.timefn = timefn
        '''Function that will be used to measure the start and end times of the
        managed context.'''

        self.times = []
        '''Measured time of each managed context, in order of when each context
        ended.'''

    @contextmanager
    def time(self):
        '''Context manager that will measure execution time of the managed
        context, then save that time to ``times``.'''
        start = self.timefn()
        try:
            yield
        finally:
            self.times.append(self.timefn() - start)
