#!/usr/bin/env python
# coding: utf-8

from QuantumCircuit import QuantumCircuit as QC
import numpy as np

# my_qc = QC(3)
# print("Initial State:" + str(np.transpose(my_qc.state)))
# my_qc.apply_hardmard(0)
# print("Qubit state after apply a hardamard gate on the first qubit: \n" + str(np.transpose(my_qc.state)))
#uncomment the line below to plot the probabilitis of each state
# my_qc.plot_pr()

'''
an expamle of grove search with 3 qubit
This link show how it works: https://qiskit.org/textbook/ch-algorithms/grover.html
'''
# grover_qc = QC(3)
# grover_qc.apply_hardmard(0)
# grover_qc.apply_hardmard(1)
# grover_qc.apply_hardmard(2)

# #oracle
# grover_qc.apply_controlZ(2,0)
# grover_qc.apply_controlZ(1,0)

# #amplification
# grover_qc.apply_hardmard(0)
# grover_qc.apply_hardmard(1)
# grover_qc.apply_hardmard(2)
# grover_qc.apply_pauliX(0)
# grover_qc.apply_pauliX(1)
# grover_qc.apply_pauliX(2)
# grover_qc.apply_controlZ([1,2],0)
# grover_qc.apply_pauliX(0)
# grover_qc.apply_pauliX(1)
# grover_qc.apply_pauliX(2)
# grover_qc.apply_hardmard(0)
# grover_qc.apply_hardmard(1)
# grover_qc.apply_hardmard(2)

# grover_qc.show_state()
# grover_qc.plot_pr()

'''
Try new method apply_grover_oracle() and apply_amplification()
'''
my_qc = QC(5)
my_qc.apply_hardmard(0)
my_qc.apply_hardmard(1)
my_qc.apply_hardmard(2)
my_qc.apply_hardmard(3)
my_qc.apply_hardmard(4)

my_qc.apply_grover_oracle(2)
my_qc.apply_amplification()
my_qc.apply_grover_oracle(2)
my_qc.apply_amplification()
my_qc.apply_grover_oracle(2)
my_qc.apply_amplification()
my_qc.apply_grover_oracle(2)
my_qc.apply_amplification()
my_qc.apply_grover_oracle(2)
my_qc.apply_amplification()
my_qc.plot_pr()