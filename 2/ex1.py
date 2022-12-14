#Allocate an array a from the python list [[0.5, -1],[-1, 2]] with dtype float 32bit.

import numpy as np

mylist = [[0.5, -1],[-1, 2]]

a = np.array(mylist[:], dtype=np.float32)


# Verify its shape, dimension and create a deep copy b using a flatten shape.

type(a)
print(f'a-shape:     {a.shape}')
print(f'a-ndim:      {a.ndim}')
print(f'a-dtype:     {a.dtype}')
print(f'a-flattened: {a.flatten()}')

b = a.flatten().copy()
c = np.copy(a.flatten())

print(f'b:           {b}')
print(f'c:           {c}')
print(f'is b=c?:     {b==c}')
b[2]=0
c[2]=5
print(a)
print(b)
print(c)

# Assign zero to elements in b with even indexes.
