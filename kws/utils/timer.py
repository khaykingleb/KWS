import time

class Timer:
    def __init__(self, name: str, verbose):
        self.name = name
        self.verbose = verbose

    def __enter__(self):
        self.t = time.time()
        return self

    def __exit__(self, type, value, traceback):
        self.t = time.time() - self.t

        if self.verbose:
            print(f"{self.name.capitalize()} model:")
            print(f"Elapsed time = {self.t:.2f} seconds.")
