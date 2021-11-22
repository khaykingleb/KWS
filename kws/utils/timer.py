import time

class Timer:
    def __init__(self, verbose):
        self.verbose = verbose
        self.fix_time = None

    def __enter__(self):
        self.t = time.time()
        return self
    
    def get_time(self):
        return time.time() - self.t

    def __exit__(self, type, value, traceback):
        self.t = time.time() - self.t

        if self.verbose:
            print(f"Elapsed time: {self.t:.2f}.")
