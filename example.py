#!/usr/bin/env python
# coding: utf-8

from QuantumCircuit import QuantumCircuit as QC
import numpy as np

my_qc = QC(3)
print("Initial State:" + str(np.transpose(my_qc.state)))
my_qc.apply_hardmard(0)
print("Qubit state after apply a hardamard gate on the first qubit: \n" + str(np.transpose(my_qc.state)))
#uncomment the line below to plot the probabilitis of each 
# my_qc.plot_pr()