#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Mar 16 20:32:59 2025

@author: gabriel
"""

import matplotlib.pyplot as plt
import numpy as np
from analytic_energy import GetAnalyticEnergies
from mpl_toolkits.mplot3d import axes3d

w_0 = 10
mu = -40
Delta = 0.2
phi_angle = 0
theta = np.pi/2   #np.pi/2
B = 2*Delta   #2*Delta_0
B_x = B * np.sin(theta) * np.cos(phi_angle)
B_y = B * np.sin(theta) * np.sin(phi_angle)
B_z = B * np.cos(theta)
Lambda_R = 0.56
Lambda_D = 0
k_x_values = np.linspace(-np.pi, np.pi, 400)
k_y_values = np.linspace(-np.pi, np.pi, 400)

E = GetAnalyticEnergies(k_x_values, k_y_values, [0], [0], w_0, mu, Delta, B_x, B_y, Lambda_R, Lambda_D)

fig, ax = plt.subplots(subplot_kw={"projection": "3d"})

# Make data.
X, Y = np.meshgrid(k_x_values, k_y_values)
Z = E[:, :, 0, 0, 0]

# Plot the surface.
surf = ax.plot_surface(X, Y, Z,
                       linewidth=0, antialiased=True, alpha=0.5)
Z = E[:, :, 0, 0, 2]
surf = ax.plot_surface(X, Y, Z,
                       linewidth=0, antialiased=True, alpha=0.5)
# ax.set(xlim=(min(k_x_values), max(k_x_values)), ylim=(min(k_y_values), max(k_y_values)), zlim=(-2, 2),
#        xlabel='X', ylabel='Y', zlabel='Z')

#%%
k_x_values = np.linspace(-np.pi/10, np.pi/10, 400)
k_y_values = np.linspace(-np.pi/10, np.pi/10, 400)
w_0 = 10
mu = -40
Delta = 0.2
B_x = 2*Delta
B_y = 0
Lambda_R = 0.56
Lambda_D = 0
E_2 = GetAnalyticEnergies(k_x_values, k_y_values, [0], [0], w_0, mu, Delta, B_x, B_y, Lambda_R, Lambda_D)

#%%

fig, ax = plt.subplots(subplot_kw={"projection": "3d"})

# Make data.
X, Y = np.meshgrid(k_x_values, k_y_values)
ax = plt.figure().add_subplot(projection='3d')

# Plot the 3D surface
# ax.plot_surface(X, Y, E[:, :, 0, 0, 0], edgecolor='royalblue', lw=0.5, rstride=8, cstride=8,
#                 alpha=0.3)

Z = E_2[:, :, 0, 0, 2]
ax.plot_surface(X, Y, E_2[:, :, 0, 0, 2], edgecolor='royalblue', lw=0.5,
                alpha=0.3, antialiased=False, axlim_clip=True, rstride=8, cstride=8)
# ax.plot_surface(X, Y, Z,
#                        linewidth=0, antialiased=True, alpha=0.3)
Z2 = E_2[:, :, 0, 0, 0]
ax.plot_surface(X, Y, E_2[:, :, 0, 0, 0], edgecolor='royalblue', lw=0.5,
                alpha=0.3, antialiased=False, axlim_clip=True, rstride=8, cstride=8)
# surf = ax.plot_surface(X, Y, Z2,
#                        linewidth=0, antialiased=True, alpha=0.3)

# Plot projections of the contours for each dimension.  By choosing offsets
# that match the appropriate axes limits, the projected contours will sit on
# the 'walls' of the graph.
# ax.contour(X, Y, Z, zdir='z', offset=0, cmap='coolwarm', levels=[0, 2])
# ax.contour(X, Y, Z, zdir='x', offset=-0.02, cmap='coolwarm')
# ax.contour(X, Y, Z, zdir='y', offset=0.02, cmap='coolwarm')

# ax.contour(X, Y, Z2, zdir='z', offset=0, cmap='coolwarm', levels=[0, 2])
# ax.contour(X, Y, Z2, zdir='x', offset=-0.02, cmap='coolwarm')
# ax.contour(X, Y, Z2, zdir='y', offset=0.02, cmap='coolwarm')


# ax.set(xlim=(-0.02, 0.02), ylim=(-0.02, 0.02), zlim=(-1, 1),
#        xlabel='X', ylabel='Y', zlabel='Z')

plt.show()