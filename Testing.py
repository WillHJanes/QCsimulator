import qiskit
import numpy as np 
import QuantumCircuit 
class Tensor_Dot_Testing:
    def __init__(self, part):
        self.part = str(part)
        #self.seed = np.random.seed()
    def test(self):
        return exec('self.'+self.part+'()')
    def get_random_matrices(self):
        W = np.random.randint(1,50,(np.random.randint(1,10),np.random.randint(1,10)))
        V = np.random.randint(1,50,np.shape(W))
        return V, W
    def tensor_product(self):
        V, W = self.get_random_matrices()
        tensor_product_test = qiskit.aqua.utils.tensorproduct(np.array(V),np.array(W))
        tensor_product_func = QuantumCircuit.QuantumCircuit(2).tensor_product(np.array(V),np.array(W))
        assert not np.any(tensor_product_func-tensor_product_test) == True, "The tensor products do not match"
    def dot_product(self):
        V, W = self.get_random_matrices()
        dot_product_test = np.matmul(np.array(V), np.array(W))
        dot_product_func = QuantumCircuit.QuantumCircuit(2).dot_product(np.array(V), np.array(W))
        all_zeros = not np.any(np.array([dot_product_func-dot_product_test]))
        assert not np.any(dot_product_func-dot_product_test) == True, "The dot products do not match"
    


