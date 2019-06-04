import numpy as np
from skimage.util.shape import view_as_windows
A = np.arange(4*4*4).reshape(4,4,4)
B=np.arange(10).reshape(10,1)
print(A)
C=B[0:9]
print(B[9])

print(A[1,0,0])
window_shape = (1, 2, 2)
B = view_as_windows(A, window_shape)
print(B.shape)
print(B)
B=np.reshape(B,(9,4,2,2))
print(B.shape)
print(B)
