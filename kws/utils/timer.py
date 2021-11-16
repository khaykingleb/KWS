import time

class Timer:
    def __init__(self, name: str, config):
        self.name = name
        self.config = config

    def __enter__(self):
        self.t = time.time()
        return self

    def __exit__(self, type, value, traceback):
        self.t = time.time() - self.t

        if self.config.verbose:
            print(f"{self.name.capitalize()} model's elapsed time: {self.t:.2f}")
