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
grove_qc = QC(3)
grove_qc.apply_hardmard(0)
grove_qc.apply_hardmard(1)
grove_qc.apply_hardmard(2)

#oracle
grove_qc.apply_controlZ(2,0)
grove_qc.apply_controlZ(1,0)

#amplification
grove_qc.apply_hardmard(0)
grove_qc.apply_hardmard(1)
grove_qc.apply_hardmard(2)
grove_qc.apply_pauliX(0)
grove_qc.apply_pauliX(1)
grove_qc.apply_pauliX(2)
grove_qc.apply_controlZ([1,2],0)
grove_qc.apply_pauliX(0)
grove_qc.apply_pauliX(1)
grove_qc.apply_pauliX(2)
grove_qc.apply_hardmard(0)
grove_qc.apply_hardmard(1)
grove_qc.apply_hardmard(2)

grove_qc.show_state()
grove_qc.plot_pr()