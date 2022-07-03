import numpy as np

a = [
    [0,0,0,1],
    [1,0,0,0],
    [0,1,0,0],
    [0,1,0,0]
]

a = [np.argmax(i) for i in a]
print(type(a))
