#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed May  7 08:22:02 2025

@author: gabriel
"""

from analytic_energy import GetAnalyticEnergies
import numpy as np
import matplotlib.pyplot as plt
from pauli_matrices import tau_0, sigma_0, tau_z, sigma_x, sigma_y, tau_y, tau_x

L_x = 400
L_y = 400   
Delta = 0.2 # meV  0.2 ###############Normal state

t = 10
Delta_0 = 0.2#t/5     
Lambda_R = 0.56
Lambda_D = 0
phi_angle = np.pi/4 #np.pi/16#np.pi/2
theta = np.pi/2 #np.pi/2   #np.pi/2
B = 2*Delta_0   #2*Delta_0
B_x = B * np.sin(theta) * np.cos(phi_angle)
B_y = B * np.sin(theta) * np.sin(phi_angle)
B_z = B * np.cos(theta)
mu = -3.8*t#-4*t



k_x_values = 2*np.pi/L_x*np.arange(0, L_x)
k_y_values = 2*np.pi/L_y*np.arange(0, L_y)
phi_x_values = np.linspace(-0.1, 0.1, 100)
phi_y_values = [0]

def get_Hamiltonian(k_x, k_y, phi_x, phi_y, w_0, mu, Delta, B_x, B_y, Lambda):
    """ Periodic Hamiltonian in x and y with flux.
    """
    H = (
        -2*w_0*((np.cos(k_x)*np.cos(phi_x) + np.cos(k_y)*np.cos(phi_y))
               * np.kron(tau_z, sigma_0)
               - (np.sin(k_x)*np.sin(phi_x) + np.sin(k_y)*np.sin(phi_y))
               * np.kron(tau_0, sigma_0)) - mu * np.kron(tau_z, sigma_0)
        + 2*Lambda*(np.sin(k_x)*np.cos(phi_x) * np.kron(tau_z, sigma_y)
                    + np.cos(k_x)*np.sin(phi_x) * np.kron(tau_0, sigma_y)
                    - np.sin(k_y)*np.cos(phi_y) * np.kron(tau_z, sigma_x)
                    - np.cos(k_y)*np.sin(phi_y) * np.kron(tau_0, sigma_x))
        - B_x*np.kron(tau_0, sigma_x) - B_y*np.kron(tau_0, sigma_y)
        + Delta*np.kron(tau_x, sigma_0)
            ) * 1/2
    return H

E = GetAnalyticEnergies(k_x_values, k_y_values, phi_x_values, phi_y_values, t,
                        mu, Delta, B_x, B_y, Lambda_R, Lambda_D)

# phi_y = 0
# eigenvalues = np.zeros((len(k_x_values), len(k_y_values),
#                         len(phi_x_values), len(phi_y_values), 4))
# for i, k_x in enumerate(k_x_values):
#     for j, k_y in enumerate(k_y_values):
#         for l, phi_x in enumerate(phi_x_values):
#             for m, phi_y in enumerate(phi_y_values):
#                 H = get_Hamiltonian(
#                     k_x, k_y, phi_x, phi_y, t, mu, Delta_0, B_x, B_y, Lambda_R
#                     )
#                 eigenvalues[i, j, l, m] = np.linalg.eigvalsh(
#                     get_Hamiltonian(
#                         k_x, k_y, phi_x, phi_y, t, mu, Delta, B_x,
#                         B_y, Lambda_R
#                         )
#                     )


E_positive = np.where(E > 0, E, np.zeros_like(E))
fundamental_energy = -np.sum(E_positive, axis=(0, 1, 3, 4))

# E_positive_2 = np.where(eigenvalues > 0, eigenvalues, np.zeros_like(eigenvalues))
# fundamental_energy_2 = -np.sum(E_positive_2, axis=(0, 1, 3, 4))

#%% Plot
fig, ax = plt.subplots()
ax.plot(phi_x_values, fundamental_energy)
# ax.plot(phi_x_values, fundamental_energy_2)

ax.set_xlabel(r"$\phi_x$")
ax.set_ylabel(r"$E_0(\phi_x)$")
plt.title(
    r"$B=$" + f"{B}" + r"; $\theta =$" + f"{np.round(theta, 2)}" +
    r"; $\varphi=$" + f"{np.round(phi_angle, 2)}"
         )

#%%

import matplotlib.pyplot as plt
from pathlib import Path
import numpy as np
import os

abspath = os.path.abspath(__file__)
dname = os.path.dirname(abspath)
os.chdir(dname)

# data_folder = Path(r"/home/gabriel/OneDrive/Doctorado-DESKTOP-JBOMLCA/Archivos/Data_19_06_25/Data")
data_folder = Path(r"./Data")

file_to_open = data_folder / "fundamental_energy_L_x=400_L_y=400_phi_x_in_(-0.01,0.01))_B_y_in_(0.0-0.013)_Delta=0.08_lambda_R=0_lambda_D=0_g_xx=1_g_xy=0_g_yy=1_g_yx=0_theta=1.57_phi_angle=1.57_points=16_beta=1000_T=False_Nphi=10.npz"
# file_to_open = data_folder / "fundamental_energy_L_x=5000_L_y=5000_mu=-349.0_phi_x_in_(-0.0001,0.0))_B_y_in_(0.0-0.4)_Delta=0.08_lambda_R=0.56_lambda_D=0_g_xx=1_g_xy=0_g_yy=1_g_yx=0_theta=1.57_phi_angle=1.57_points=16_beta=1000_T=False_Nphi=10.npz"

Data = np.load(file_to_open)

fundamental_energy = Data["E_vs_B"]
phi_x_values = Data["phi_x_values"]
# phi_x_values = np.linspace(-0.02, 0.02, 50)
B_values = Data["B_values"]
mu = Data["mu"]
phi_angle = Data["phi_angle"]
theta = Data["theta"]
Delta_0 = Data["Delta_0"]
L_x = Data["L_x"]
L_y = Data["L_y"]
Lambda_R = Data["Lambda_R"]
w_0 = Data["w_0"]
# h = 2e-4
h = Data["h"]
# Nphi = 30
Nphi = Data["Nphi"]

from scipy.optimize import curve_fit
from numpy.polynomial import Polynomial

def cuartic_function(x, a, b, c, d, e):
    return a*x**4 + b*x**3 + c*x**2 + d*x + e 

def parabola(x, c, b, a):
    return a*x**2 + b*x + c

superfluid_density = np.zeros_like(B_values)
superfluid_density_error = np.zeros_like(B_values)

superfluid_density_polinomial = np.zeros_like(B_values)
n_s_xx = np.zeros_like(B_values)

phi_x_values_fit = slice(0, 30)

initial_parameters = [1e6, 1e3, 1e7]
fig, ax = plt.subplots()
for i, B in enumerate(B_values):
    # popt, pcov = curve_fit(parabola, phi_x_values,
    #                        fundamental_energy[i, :],
    #                        p0=initial_parameters)
    p = Polynomial.fit(phi_x_values[phi_x_values_fit], fundamental_energy[i, phi_x_values_fit], 2)
    ax.plot(phi_x_values, fundamental_energy[i, :], label=r"$B/\Delta=$"
            + f"{np.round(B/Delta_0, 2)}")
    # ax.plot(phi_x_values, parabola(phi_x_values, *popt), "r--")
    ax.plot(p.linspace()[0], p.linspace()[1], "b--")
    # superfluid_density[i] = popt[2]/(2*(w_0*L_x*L_y))
    superfluid_density_polinomial[i] = p.convert().coef[2]/(2*(w_0*L_x*L_y))
    # superfluid_density_error[i] = np.sqrt(np.diag(pcov))[2]/(2*(w_0*L_x*L_y))
    # h = 0.001
    n_s_xx[i] = 1/w_0 * 1/(L_x*L_y) * ( fundamental_energy[i, -1] - 2*fundamental_energy[i,5] + fundamental_energy[i,0]) / h**2
# ax.plot(phi_x_values, fundamental_energy_2)

ax.set_xlabel(r"$\phi_x$")
ax.set_ylabel(r"$E_0(\phi_x)$")
# plt.legend()
plt.title("Fundamental energy "
          + r"$E_0(\phi) = \sum_{\epsilon_k(\phi) < 0} \epsilon_k(\phi)$"
          + "\n"
          + r" $\mu=$" + f"{mu}"
          + r"; $\varphi=$" + f"{np.round(phi_angle, 2)}"
          + r"; $\theta=$" + f"{np.round(theta, 2)}"
          + "\n"
          + r"$L_x=$" + f"{L_x}"
          + r"; $L_y=$" + f"{L_y}"
          + r"; $\lambda_R=$" + f"{Lambda_R}")
ax.set_box_aspect(2)
ax.legend(loc='center left', bbox_to_anchor=(1, 0.5))
plt.tight_layout()

fig, ax = plt.subplots()
# ax.errorbar(B_values/Delta_0, superfluid_density,yerr=superfluid_density_error,
#             marker="o",
#             markersize=3)
ax.plot(B_values/Delta_0, superfluid_density_polinomial, "-ob")
ax.plot(B_values/Delta_0, n_s_xx, "-or")
ax.set_title("Superfluid density "
          + "\n"
          + r" $\mu=$" + f"{mu}"
          + r"; $\varphi=$" + f"{np.round(phi_angle, 2)}"
          + r"; $\theta=$" + f"{np.round(theta, 2)}"
          + "\n"
          + r"$L_x=$" + f"{L_x}"
          + r"; $L_y=$" + f"{L_y}"
          + r"; $\lambda_R=$" + f"{Lambda_R}"
          + f"; h={h}"
          + r"; $N_\varphi=$" + f"{Nphi}")