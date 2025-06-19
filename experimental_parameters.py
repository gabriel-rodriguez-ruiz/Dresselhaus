#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Apr  3 13:06:03 2025

@author: gabriel
"""

import numpy as np
import matplotlib.pyplot as plt

E_F = 0.0506 # eV
Delta = 0.08   #0.2 # meV induced gap
g = 100
n = 8.5e11 # 1/cm**2

hbar = 6.58e-16 # eV
m_e = 5.1e5 / (3e10)**2 # eV
a = 3.08e-07   #3.08e-07   6e-8 # cm
m = 0.0403 * m_e
mu_B = 5.79e-2 # eV/T

t = hbar**2 / (2*a**2*m) # eV
k_F = np.sqrt(2*np.pi*n) # 1/cm
mu = E_F - 4*t
Lambda_R = 5*Delta/k_F /a # meV
Lambda_D = 0

#%% Free electron and tight-binding
L_x = 1000
k_values = np.pi/(L_x*a)*np.arange(-L_x, L_x)

def free_electron(k):
    return hbar**2*k**2/(2*m)

def free_electron_SOC(k):
    return [hbar**2*k**2/(2*m) + Lambda_R*k,
            hbar**2*k**2/(2*m) - Lambda_R*k]

def tight_binding(k):
    return -2*t*np.cos(k*a)+2*t

def SOC_tight_binding(k):
    return [-2*t*np.cos(k*a)+2*t + Lambda_R * 1/a*np.sin(k*a),
            -2*t*np.cos(k*a)+2*t - Lambda_R * 1/a*np.sin(k*a)]
fig, ax = plt.subplots()

# ax.plot(k_values, [free_electron(k) for k in k_values])
ax.plot(k_values, [free_electron(k) for k in k_values])
# ax.plot(k_values, [free_electron_SOC(k) for k in k_values])

ax.plot(k_values, [E_F for k in k_values])
ax.plot(k_values, [tight_binding(k) for k in k_values], ".")
# ax.plot(k_values, [t*a**2*k**2 for k in k_values], ".")
ax.axvline(x=k_F, ls='--', color="blue")
ax.axvline(x=-k_F, ls='--', color="blue")

ax.set_xlabel(r"$k[cm^{-1}]$")
ax.set_ylabel(r"$E_k[eV]$")

plt.show()
#%% Energy bands

L_x = 100
L_y = 100
B_x = 0.1     # T
B_y = 0     # eV/T

B_c = 0.07 # T
g = 2*Delta / (mu_B * B_c)
Delta_Z_x = g*mu_B*B_x/2   # meV
Delta_Z_y = g*mu_B*B_y/2   # meV


#k_y_values = 2*np.pi/L_x*np.arange(-L_x/2, L_x/2)
# k_y_values = 2*np.pi/L_y*np.arange(0, L_y)

k_y_values = np.linspace(-0.25, 0.25, L_x)  #np.linspace(k_F*a, 1.1*k_F*a, L_x)
k_x_values = np.linspace(-0.25, 0.25, L_x)  #np.linspace(k_F*a, 1.1*k_F*a, L_x)

# k_y_values = np.linspace(k_F*a, 1.05*k_F*a, L_y)

#%% 1D-plot of pockets
from analytic_energy import GetAnalyticEnergies

E = GetAnalyticEnergies([0], k_y_values, [0], [0], w_0=t, mu=mu, Delta=Delta,
                        B_x=Delta_Z_x, B_y=Delta_Z_y, Lambda_R=Lambda_R, Lambda_D=Lambda_D)

fig, ax = plt.subplots()
ax.plot(k_y_values, E[0, :, 0, 0, 1], ".")
ax.plot(k_y_values, E[0, :, 0, 0, 2], ".")
# ax.plot(k_y_values, E[0, :, 0, 0, 0], ".")
# ax.plot(k_y_values, E[0, :, 0, 0, 3], ".")
ax.grid()

ax.set_xlabel(r"$k_y a $")
ax.set_ylabel(r"$E_k[meV]$")
plt.show()
#%%
E = GetAnalyticEnergies(k_x_values, k_y_values, [0], [0], w_0=t, mu=mu, Delta=Delta,
                        B_x=Delta_Z_x, B_y=Delta_Z_y, Lambda_R=Lambda_R, Lambda_D=Lambda_D)

fig, ax = plt.subplots(subplot_kw={"projection": "3d"})

# Make data.
X = k_x_values
Y = k_y_values
X, Y = np.meshgrid(X, Y)

# Plot the surface.
# ax.plot_surface(X, Y, eigenvalues[:, :, 0],
#                 linewidth=0, antialiased=False,
#                 alpha=0.2, cmap='PuOr', vmin=-1, vmax=1)
ax.plot_surface(X, Y, E[:, :, 0, 0, 1]/Delta,
                linewidth=0, antialiased=True,
                alpha=0.3, cmap='PuOr', vmin=-1, vmax=1, edgecolor='red', lw=0.5, rstride=4, cstride=4)
ax.plot_surface(X, Y, E[:, :, 0, 0, 2]/Delta,
                linewidth=0, antialiased=True,
                alpha=0.3, cmap='PuOr', vmin=-1, vmax=1, edgecolor='blue', lw=0.5, rstride=4, cstride=4)
# ax.plot_surface(X, Y, eigenvalues[:, :, 3],
#                 linewidth=0, antialiased=False,
#                 alpha=0.2, cmap='PuOr', vmin=-1, vmax=1)

ax.set_xlabel(r'$k_y$')
ax.set_ylabel(r'$k_x$')

# ax.set(xlim=(min(1.3*k_y_values/np.pi), max(1.3*k_y_values/np.pi)), ylim=(min(1.3*k_x_values/np.pi), max(1.3*k_x_values/np.pi)), zlim=(-6, 6), #zlim=(-15, 15),
#        xlabel='X', ylabel='Y', zlabel='Z')
#%%
# plt.contour(X, Y, eigenvalues[:, :, 0], levels=np.array([0]), colors=["k"])
ax.contour(X, Y, 2*E[:, :, 0, 0, 1]/Delta, levels=np.array([0]), colors=["k"])
ax.contour(X, Y, 2*E[:, :, 0, 0, 2]/Delta, levels=np.array([0]), colors=["k"])
# plt.contour(X, Y, eigenvalues[:, :, 3], levels=np.array([0]), colors=["k"])

ax.contour(X, Y, 2*E[:, :, 0, 0, 2]/Delta, zdir='x', offset=ax.get_xlim()[0], colors='blue', levels=np.array([0]))
ax.contour(X, Y, 2*E[:, :, 0, 0, 2]/Delta, zdir='y', offset=ax.get_ylim()[1], colors='blue', levels=np.array([0]))

ax.contour(X, Y, 2*E[:, :, 0, 0, 1]/Delta, zdir='x', offset=ax.get_xlim()[0], colors='red', levels=np.array([0]))
ax.contour(X, Y, 2*E[:, :, 0, 0, 1]/Delta, zdir='y', offset=ax.get_ylim()[1], colors='red', levels=np.array([0]))

ax.contour(X, Y, np.zeros_like(2*E[:, :, 0, 0, 2]/Delta), zdir='y', offset=ax.get_ylim()[1], levels=np.array([0]), colors="k", linestyles='dashed')
ax.contour(X, Y, np.zeros_like(2*E[:, :, 0, 0, 2]/Delta), zdir='x', offset=ax.get_xlim()[0], levels=np.array([0]), colors="k", linestyles='dashed')
# ax.contour(X, Y, np.zeros_like(eigenvalues[:, :, 1]), zdir='z', offset=ax.get_zlim()[0], levels=10, colors="k", linestyles='dashed')

ax.contour(X, Y, np.zeros_like(2*E[:, :, 0, 0, 1]/Delta), zdir='y', offset=ax.get_ylim()[1], levels=np.array([0]), colors="k", linestyles='dashed')
ax.contour(X, Y, np.zeros_like(2*E[:, :, 0, 0, 1]/Delta), zdir='x', offset=ax.get_xlim()[0], levels=np.array([0]), colors="k", linestyles='dashed')

# ax.contour(X, Y, Z, zdir='y', offset=40, cmap='coolwarm')

ax.set_xlabel(r'$k_y$')
ax.set_ylabel(r'$k_x$')
ax.set_zlabel(r'$E(k_x, k_y)/\Delta$')

plt.tight_layout()
