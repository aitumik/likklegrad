import numpy as np 

BYTES_PER_FLOAT32 = 4
KB_TO_BYTES = 1024
MB_TO_BYTES = 1024 * 1024 

class Tensor:
    """
    Tensor, the foundation of machine learning computation. It carries the 
    computation graph of the entire network
    """

    def __init__(self,data):
        self.data  = np.array(data,dtype=np.float32)
        self.shape = self.data.shape 
        self.size  = len(self.data)
        self.dtype = self.data.dtype

    def __repr__(self):
        """
        String representation of the tensor for debugging
        """
        return f"Tensor(data={self.data},shape={self.shape})"

    def __str__(self):
        """
        Human readable string representation of the tensor
        """
        return f"Tensor(data={self.data})"

    def numpy(self):
        """
        Return the underlying np array.
        """
        return self.data

    def memory_footprint(self):
        """
        Calculate exact memory usage in bytes
        """
        return self.data.nbytes

    def __add__(self,other):
        """
        Adds two tensors elementwise with broadcasting support
        """
        if isinstance(other,Tensor): 
            return Tensor(self.data + other.data)
        else:
            return Tensor(self.data + other)

    def __sub__(self,other):
        """
        Subtract two tensors elementwise with broadcasting support
        """
        if isinstance(other,Tensor): 
            return Tensor(self.data - other.data)
        else:
            return Tensor(self.data - other)

    def __mul__(self,other):
        """
        Subtract two tensors elementwise(not matmul)
        """
        if isinstance(other,Tensor): 
            return Tensor(self.data * other.data)
        else:
            return Tensor(self.data * other)

    def __truediv__(self,other):
        """
        Divide two tensors element-wise
        """
        if isinstance(other,Tensor): 
            return Tensor(self.data / other.data)
        else:
            return Tensor(self.data / other)

    def matmul(self,other):
        """
        Matrix multiplication of two tensors
        """
        pass

    def __matmul__(self,other):
        """Enable @ operator for matrix multiplication"""
        return self.matmul(other)

    def __getitem__(self,key):
        """
        Enable indexing and slicing operations on Tensors
        """

    def reshape(self,*shape):
        """
        Reshape tensor to new dimensions
        """
        pass

    def transpose(self,dim0=None,dim1=None):
        """
        Transpose tensor dimensions
        """
        pass

    def sum(self,axis=None,keepdims=False):
        """
        Sum tensor along specified axis
        """
        pass
    def mean(self,axis=None,keepdims=False):
        """
        Compute mean of a tensor along specified axis
        """
        pass

    def max(self,axis=None,keepdims=False):
        """
        Find maximum values along specified axis
        """
        pass


if __name__ == "__main__":
    # do something here
    print("testing tensors")
