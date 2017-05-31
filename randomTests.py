import numpy as np

b = np.array([1,-2,3])


def relu(x):
    if x < 0:
        return 0
    else:
        return x
print(b.T)
print(np.apply_along_axis(relu, -1, b))
