import numpy as np



def main():
    #add_lists()
    #add_numpy_arrays()
    #multidimensional_array()
    #array_slicing()
    #array_reshaping()
    #marray_slicing()
    #manipulate_array_shape()
    #matrix_ops()
    #matmult()
    handvstack()
    
    
def handvstack():
    x = np.arange(4).reshape(2, 2)
    print(x)
    y = np.arange(4, 8).reshape(2, 2)
    print(y)
    z = np.vstack((x, y))
    print(z) 

def add_lists():
    ''' Adds two lists together '''
    list_one = list(range(1, 4))
    list_two = list(range(4, 7))
    list_sum = []
    for i in range(3):
        list_sum.append(list_one[i]+list_two[i])
    print(list_sum)
    

def add_numpy_arrays():
    array_one = np.arange(1, 4)
    array_two = np.arange(4, 7)
    print(array_one + array_two)
    np.power(np.array([1, 2, 3, 4]), 4)
    np.negative(np.array[1, 2, 3])
    # np.exp
    #np.log
    #np.sin
    
    
def multidimensional_array():
    x = np.arange(3)
    y = np.arange(3)
    z = np.arange(3)
    multi_array = np.array([x, y, z], dtype=np.int8)
    print(multi_array.dtype)
    print(multi_array)
    print(multi_array.shape)
    w = np.linspace(1, 10, 3, False)
    print(w)
    
    
def manipulate_array_shape():
    x = np.arange(9).reshape(3, 3)
    print(x)
    ravelled_array = x.ravel()
    print(ravelled_array)
    y = np.arange(12).reshape(3, 4)
    print(y)
    flattened_array = y.flatten()
    flattened_array[2] = 100000
    print(flattened_array)
    print(y)
    
    
def matrix_ops():
    y = np.arange(9).reshape(3,3)
    print(y)
    print(y.T)
    print(np.eye(3))
    print(np.zeros((3,2), dtype=np.int8))
    print(np.ones((3,2), dtype=np.int8))
    print(np.random.rand(4,4))
    
    
def array_slicing():
    x = np.arange(1, 10)
    print(x)
    print(x[2:])
    

def array_reshaping():
    x = np.arange(18)
    print(x)
    x = x.reshape(3, 3, 2)
    print(x)
    
    
def marray_slicing():
    x = np.arange(9).reshape(3, 3)
    print(x[2 ,1])
    x = np.arange(18).reshape(3, 2, 3)
    print(x[1, :,:])
    print(x.max())
    print(x.min())
    
    
def matmult():
    mat_a = np.matrix([0, 1, 2, 3, 4, 5]).reshape(2, 3)
    print(mat_a)
    mat_b = np.matrix([3, 1, 2, 3, 4, 5]).reshape(3, 2)
    print(mat_b)
    product = np.matmul(mat_a, mat_b)
    print(product)
    
    
if __name__ == "__main__":
    main()

