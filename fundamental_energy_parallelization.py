#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu May  8 09:26:13 2025

@author: gabriel
"""

from analytic_energy import GetAnalyticEnergies
import numpy as np
from pathlib import Path
import multiprocessing

L_x = 400  #2500
L_y = 400   #2500
Delta = 0.2 # meV  0.2 ###############Normal state

t = 10
Delta_0 = 0.2#t/5     
Lambda_R = 0.56
Lambda_D = 0
phi_angle = np.pi/2  #np.pi/2 #np.pi/16#np.pi/2
theta = np.pi/2 #np.pi/2   #np.pi/2
B = 2*Delta_0   #2*Delta_0
B_x = B * np.sin(theta) * np.cos(phi_angle)
B_y = B * np.sin(theta) * np.sin(phi_angle)
B_z = B * np.cos(theta)
mu = -3.9*t#-2*t



k_x_values = 2*np.pi/L_x*np.arange(0, L_x)
k_y_values = 2*np.pi/L_y*np.arange(0, L_y)
phi_x_values = np.linspace(-np.pi/30, np.pi/30, 100)
phi_y_values = [0]

n_cores = 8
params = {"L_x": L_x, "L_y": L_y, "t": t,
          "mu": mu, "Delta_0": Delta_0, "theta": theta,
           "Lambda_R": Lambda_R, "phi_angle": phi_angle,
          "k_x_values": k_x_values,
          "k_y_values": k_y_values,
          "Lambda_D": Lambda_D, "phi_x_values": phi_x_values,
          "phi_y_values": phi_y_values
          }


def integrate(B):
    B_x = B * np.sin(theta) * np.cos(phi_angle)
    B_y = B * np.sin(theta) * np.sin(phi_angle)
    E = GetAnalyticEnergies(k_x_values, k_y_values, phi_x_values, phi_y_values, t,
                            mu, Delta, B_x, B_y, Lambda_R, Lambda_D)
    return E

if __name__ == "__main__":
    B_values = Delta_0 * np.array([1/2, 1, 3/2, 2])
    with multiprocessing.Pool(n_cores) as pool:
        results_pooled = pool.map(integrate, B_values)
    E_vs_B = np.array(results_pooled)
    data_folder = Path("Data/")
    
    name = f"fundamental_energy_B={B_values}_mu={mu}_L_x={L_x}.npz"
    file_to_open = data_folder / name
    np.savez(file_to_open , E_vs_B=E_vs_B, B_values=B_values,
             **params)
    print("\007")    