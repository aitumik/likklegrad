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
        Multiply two tensors elementwise(not matmul). Also known as 
        hadamard product. 
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
        if isinstance(other,Tensor):
            return Tensor(np.matmul(self.data,other.data))
        else:
            return Tensor(np.matmul(self.data,other))

    def __matmul__(self,other):
        """Enable @ operator for matrix multiplication"""
        return self.matmul(other)

    def __getitem__(self,key):
        """
        Enable indexing and slicing operations on Tensors
        """
        pass

    def reshape(self,*shape):
        """
        Reshape tensor to new dimensions
        """
        return Tensor(self.data.reshape(*shape))

    def transpose(self,dim0=None,dim1=None):
        """
        Transpose tensor dimensions. Swap rows with columns
        """

        if dim0 is None and dim1 is None:
            if len(self.data.shape) < 2:
                return Tensor(self.data.copy())
            else:
                axes = list(range(len(self.data.shape)))
                axes[-2],axes[-1] = axes[-1],axes[-2]
                transposed_data = np.transpose(self.data,axes)
        else:
            if dim0 is None or dim1 is None:

                provided = f"dim0={dim0}" if dim1 is None else f"dim1={dim1}"
                missing  = f"dim1" if dim1 is None else f"dim0"

                raise ValueError(
                    f"Transpose requires both dimensions to be specified\n"
                    f"Got {provided} but {missing} is None\n"
                    f"Either both dim0 and dim1 must be specified, or neither (default swaps last two)\n"
                    f"Use transpose({dim0 if dim0 is not None else 0},{dim1 if dim1 is not None else 1}) or just transpose()"
                )

            else:
                axes = list(range(len(self.data.shape)))
                axes[dim1],axes[dim0] = axex[dim0],axes[dim1]
                transposed_data = np.transpose(self.data,axes)

        return Tensor(transposed_data)


    def sum(self,axis=None,keepdims=False):
        """
        Sum tensor along specified axis
        """
        result = np.sum(self.data,axis=axis,keepdims=keepdims)
        return Tensor(result)

    def mean(self,axis=None,keepdims=False):
        """
        Compute mean of a tensor along specified axis
        """
        result = np.mean(self.data,axis=axis,keepdims=keepdims)
        return Tensor(result)

    def max(self,axis=None,keepdims=False):
        """
        Find maximum values along specified axis
        """
        result = np.max(self.data,axis=axis,keepdims=keepdims)
        return Tensor(result)


if __name__ == "__main__":
    # do something here
    print("testing tensors")
