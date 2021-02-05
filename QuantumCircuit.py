#!/usr/bin/env python
# coding: utf-8


import numpy as np
import math
import QuantumGate as QG
from matplotlib import pyplot as plt 

class QuantumCircuit():
    def __init__(self, qubit_number):
        assert (qubit_number>0), 'Qubit number should be more than 0'
        self.qn = qubit_number
        self.state = self.get_initial_state()
    
    def tensor_product(self, V, W):
        result = None
        M_list = []
        for V_row_index in range(V.shape[0]):
            R_list = []
            for V_col_index in range(V.shape[1]):
                temp = np.zeros(W.shape)
                V_entry = V[V_row_index][V_col_index]
                for W_row_index in range(W.shape[0]):
                    for W_col_index in range(W.shape[1]):
                        temp[W_row_index][W_col_index] = V_entry*W[W_row_index][W_col_index]  
                if len(R_list) == 0:
                    R_list = temp
                else:
                    R_list = np.concatenate((R_list, temp),axis=1)
            M_list.append(R_list)

        result = M_list[0]
        for i in range(1, len(M_list)):
            result = np.concatenate((result,M_list[i]),axis=0)
        return result
    
    def get_initial_state(self):
        state = np.zeros(2**self.qn)
        state[0] = 1
        return state.reshape(len(state),1)
        
    def apply_hardmard(self, wire_index):
        assert -1<wire_index<self.qn, 'Input argument should be between wire 0 to ' + str(self.qn)
        if self.qn == 1:
            self.state = np.dot(QG.H,self.state)
        else:
            gate_list = []
            for i in range(self.qn):
                if i == wire_index:
                    gate_list.append(QG.H)
                else:
                    gate_list.append(QG.I)

            gate_M = gate_list[0]
            for i in range(1, self.qn):
                gate_M = self.tensor_product(gate_M, gate_list[i])
            self.state = np.dot(gate_M,self.state)
            
    def apply_pauliX(self, wire_index):
        assert -1<wire_index<self.qn, 'Input argument should be between wire 0 to ' + str(self.qn)
        if self.qn == 1:
            self.state = np.dot(QG.PX,self.state)  
        else:
            gate_list = []
            for i in range(self.qn):
                if i == wire_index:
                    gate_list.append(QG.PX)
                else:
                    gate_list.append(QG.I)

            gate_M = gate_list[0]
            for i in range(1, self.qn):
                gate_M = self.tensor_product(gate_M, gate_list[i])
            self.state = np.dot(gate_M,self.state)
    
    def apply_pauliY(self, wire_index):
        assert -1<wire_index<self.qn, 'Input argument should be between wire 0 to ' + str(self.qn)
        if self.qn == 1:
            self.state = np.dot(QG.PY,self.state)  
        else:
            gate_list = []
            for i in range(self.qn):
                if i == wire_index:
                    gate_list.append(QG.PY)
                else:
                    gate_list.append(QG.I)

            gate_M = gate_list[0]
            for i in range(1, self.qn):
                gate_M = self.tensor_product(gate_M, gate_list[i])
            self.state = np.dot(gate_M,self.state)
    
    def apply_pauliZ(self, wire_index):
        assert -1<wire_index<self.qn, 'Input argument should be between wire 0 to ' + str(self.qn)
        
        if self.qn == 1:
            self.state = np.dot(QG.PZ,self.state)
        else:
            gate_list = []
            for i in range(self.qn):
                if i == wire_index:
                    gate_list.append(QG.PZ)
                else:
                    gate_list.append(QG.I)
            gate_M = gate_list[0]
            for i in range(1, self.qn):
                gate_M = self.tensor_product(gate_M, gate_list[i])
            self.state = np.dot(gate_M,self.state)
            
    def apply_swap(self, wire_index1, wire_index2):
        assert self.qn>1, 'The curcuit does not have enough qubit to do SWAP gate'
        assert wire_index1<self.qn or wire_index2<self.qn, 'Input argument should be between wire 0 to ' + str(self.qn)
        
        if self.qn == 2:
            self.state = np.dot(QG.SWAP,self.state)
        else:
            if wire_index1 < wire_index2:
                a = wire_index1
            else:
                a = wire_index2
            gate_list = []
            for i in range(self.qn-1):
                if i == a:
                    gate_list.append(QG.SWAP)
                else:
                    gate_list.append(QG.I)
            gate_M = gate_list[0]
            for i in range(1, self.qn-1):
                gate_M = self.tensor_product(gate_M, gate_list[i])
            self.state = np.dot(gate_M,self.state)
    
    def apply_rotation(self, wire_index, angel=0):
        pass
    
    def apply_measurement(self, wire_index):
        pass
    
    def apply_cnot(self, wire_idnex1, wire_index2):
        pass
    
    def plot_pr(self):
        temp_x = range(1,2**self.qn + 1)
        x = []
        for elem in temp_x:
            x.append(str(elem))
        y = []
        for i in range(self.state.shape[0]):
            y.append((self.state[i][0])**2)
        plt.style.use('seaborn')
        plt.bar(x, y, width=0.5)
        plt.tick_params(axis='both', labelsize=15)
        plt.ylim(0, 1)
        plt.ylabel('Probability', fontsize=15) 
        plt.xlabel('State', fontsize=15)
        plt.show()





