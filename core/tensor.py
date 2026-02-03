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

if __name__ == "__main__":
    # do something here
    print("testing tensors")
