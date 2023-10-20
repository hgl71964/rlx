import os
import random
import time

from absl import app
from absl import flags

import numpy as np

FLAGS = flags.FLAGS

flags.DEFINE_integer("seed", 0, "")
flags.DEFINE_integer("len", 100, "")


def main(_):
    # set random seeds for reproducability
    random.seed(FLAGS.seed)
    np.random.seed(FLAGS.seed)

    ops = ["+", '-', '*', '/', 'sin', 'cos', 'pow', 'sqrt']
    times = []
    for op in ops:
        t = []
        for _ in range(FLAGS.len):
            a, b = np.random.rand(100), np.random.rand(100)
            if op == '+':
                t1 = time.perf_counter()
                c = np.add(a, b)
                t2 = time.perf_counter()
                t.append(t2 - t1)
            elif op == '-':
                t1 = time.perf_counter()
                c = np.subtract(a, b)
                t2 = time.perf_counter()
                t.append(t2 - t1)
            elif op == '*':
                t1 = time.perf_counter()
                c = np.multiply(a, b)
                t2 = time.perf_counter()
                t.append(t2 - t1)
            elif op == '/':
                t1 = time.perf_counter()
                c = np.divide(a, b)
                t2 = time.perf_counter()
                t.append(t2 - t1)
            elif op == 'sin':
                t1 = time.perf_counter()
                c = np.sin(a)
                t2 = time.perf_counter()
                t.append(t2 - t1)
            elif op == 'cos':
                t1 = time.perf_counter()
                c = np.cos(a)
                t2 = time.perf_counter()
                t.append(t2 - t1)
            elif op == 'pow':
                t1 = time.perf_counter()
                c = np.power(a, 2)
                t2 = time.perf_counter()
                t.append(t2 - t1)
            elif op == 'sqrt':
                t1 = time.perf_counter()
                c = np.sqrt(a)
                t2 = time.perf_counter()
                t.append(t2 - t1)
            else:
                raise RuntimeError(f"unsupport op{op}")
        times.append(sum(t) / len(t))

    print("unnormalised: ", times)
    norm = times[0]
    t = []
    for i, item in enumerate(times):
        if i == 0:
            t.append(1)
        else:
            print(ops[i], item / norm)
            t.append(item / norm)
    print("normalise: ", t)


if __name__ == "__main__":
    app.run(main)
