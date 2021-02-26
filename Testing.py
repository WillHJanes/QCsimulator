#!/usr/bin/env python
# coding: utf-8

# In[1]:


import qiskit
import numpy as np 
import QuantumCircuit 
class Testing:
    def __init__(self, part):
        self.part = str(part)
        #self.seed = np.random.seed()
    def test(self):
        return exec('self.'+self.part+'()')
    def get_random_matrices(self):
        n = np.random.randint(1,10)
        W = []
        V = []
        for i in range(n):
            V_row = []
            W_row = []
            for j in range(n):
                V_row.append([np.random.randint(0, 50)])
                W_row.append([np.random.randint(0, 50)])
            V.append(V_row)
            W.append(W_row)
        return V, W
    def tensor_product(self):
        V, W = get_random_matrices()
        tensor_product_test = qiskit.aqua.utils.tensorproduct(np.array(V),np.array(W))
        tensor_product_func = QuantumCircuit.QuantumCircuit(2).tensor_product(np.array(V),np.array(W))
        all_zeros = not np.any(tensor_product_func.reshape(n**2,n**2)-tensor_product_test.reshape(n**2,n**2))
        if (all_zeros == True):
            print('Success')
        else:
            print('Fail')
    def dot_product(self):
        V, W = get_random_matrices()
        dot_product_test = np.matmul(np.array(V), np.array(W))
        dot_product_func = QuantumCircuit.QuantumCircuit(2).dot_product(np.array(V), np.array(W))
        all_zeros = not np.any(np.array([dot_product_func-dot_product_test]))
        if (all_zeros == True):
            print('Success')
        else:
            print('Fail')
            
# In[ ]:

    


