import time
import numpy as np
import tensorflow as tf;

X = np.random.random((32, 10))
y = np.random.random((10,))

print(y)

y = np.expand_dims(y, axis=1)

print(y)

Y = np.concatenate([y]*3, axis=1)

print(Y)