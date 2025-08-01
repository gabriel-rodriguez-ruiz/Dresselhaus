#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu May  8 09:26:13 2025

@author: gabriel
"""

from analytic_energy import GetSumOfPositiveAnalyticEnergy
import numpy as np
from pathlib import Path
import multiprocessing

L_x = 2500#400  #2500
L_y = 2500#400   #2500

w_0 = 100
Delta_0 = 0.08#t/5     
Lambda_R = 0#0.56
Lambda_D = 0
phi_angle = np.pi/2  #np.pi/2 #np.pi/16#np.pi/2
theta = np.pi/2 #np.pi/2   #np.pi/2
mu = -3.49*w_0#-2*t
beta = 1000
T = False
h = 1e-1
Nphi = 10
phi_x_values = np.linspace(-h, h, Nphi)
g_xx = 1
g_yy = 1
g_xy = 0
g_yx = 0

k_x_values = 2*np.pi/L_x*np.arange(0, L_x)
k_y_values = 2*np.pi/L_y*np.arange(0, L_y)

n_cores = 16
points = 1 * n_cores

params = {"L_x": L_x, "L_y": L_y, "w_0": w_0,
          "mu": mu, "Delta_0": Delta_0, "theta": theta,
           "Lambda_R": Lambda_R, "phi_angle": phi_angle,
          "k_x_values": k_x_values,
          "k_y_values": k_y_values,
          "Lambda_D": Lambda_D, "phi_x_values": phi_x_values,
          "h": h, "Nphi":Nphi
          }

def Fermi_function_efficiently(energy, beta, T=False):
    if T==False:
        return np.zeros_like(energy)
    else:
        if energy <= 0:
            Fermi = 1 / (np.exp(beta*energy) + 1)
        else:
            Fermi = np.exp(-beta*energy) / (1 + np.exp(-beta*energy))
            return Fermi

def get_fundamental_energy_efficiently(L_x, L_y, w_0, mu, Delta_0, B_x, B_y, Lambda_R, Lambda_D, phi_values, beta, T):
    #k_x_values = 2*np.pi/L_x*np.arange(0, L_x)
    #k_y_values = 2*np.pi/L_y*np.arange(0, L_y)
    #phi_x_values = [-h, 0, h]
    #phi_y_values = [-h, 0, h]
    k_x = 2*np.pi/L_x*np.arange(0, L_x)
    k_y = 2*np.pi/L_y*np.arange(0, L_y)
    fundamental_energy = np.zeros(len(phi_x_values))
    for k, phi_x in enumerate(phi_x_values):
        for i, k_x in enumerate(k_x_values):
            for j, k_y in enumerate(k_y_values):
                positive_energy = GetSumOfPositiveAnalyticEnergy(k_x, k_y, phi_x, 0, w_0, mu, Delta_0, B_x, B_y, Lambda_R, Lambda_D)
                fundamental_energy[k] += -1/2*positive_energy + positive_energy * Fermi_function_efficiently(positive_energy, beta, T)
    return fundamental_energy

def integrate(B):
    B_x = B * np.sin(theta) * np.cos(phi_angle)
    B_y = B * np.sin(theta) * np.sin(phi_angle)
    E = get_fundamental_energy_efficiently(L_x, L_y, w_0, mu, Delta_0, B_x, B_y, Lambda_R, Lambda_D, phi_x_values, beta, T)
    return E

if __name__ == "__main__":
    # B_values = Delta_0 * np.array([1/2, 1, 3/2, 2])
    B_values = Delta_0 * np.linspace(0*Delta_0, 2*Delta_0, points)
    with multiprocessing.Pool(n_cores) as pool:
        results_pooled = pool.map(integrate, B_values)
    E_vs_B = np.array(results_pooled)
    data_folder = Path("Data/")
    
    name = f"fundamental_energy_L_x={L_x}_L_y={L_y}_phi_x_in_({np.min(phi_x_values)},{np.round(np.max(phi_x_values),3)}))_B_y_in_({np.min(B_values)}-{np.round(np.max(B_values),3)})_Delta={Delta_0}_lambda_R={np.round(Lambda_R, 2)}_lambda_D={Lambda_D}_g_xx={g_xx}_g_xy={g_xy}_g_yy={g_yy}_g_yx={g_yx}_theta={np.round(theta,2)}_phi_angle={np.round(phi_angle,2)}_points={points}_beta={beta}_T={T}_Nphi={Nphi}.npz"
    file_to_open = data_folder / name
    np.savez(file_to_open , E_vs_B=E_vs_B, B_values=B_values,
             **params)
    print("\007")    