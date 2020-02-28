import numpy as np
x1 = np.load("0-50209.npy")
x2 = np.load("100420-150629.npy")
x3 = np.load("100420-150629.npy")
x4 = np.load("150630-200839.npy")
x = np.concatenate([x1,x2,x3,x4])
np.save("train",x)