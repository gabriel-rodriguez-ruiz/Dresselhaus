#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jun  5 17:23:07 2024

@author: gabriel
"""

import matplotlib.pyplot as plt
from pathlib import Path
import numpy as np

data_folder = Path("Data/")
file_to_open = data_folder / "n_theta_mu_-349.0_L=2500_h=0.001_theta_in_(0.0-1.571)B=0.29_Delta=0.2_lambda_R=1.5454059201910981_lambda_D=0_g_xx=1_g_xy=0_g_yy=1_g_yx=0_points=16.npz"
Data = np.load(file_to_open, allow_pickle=True)

n_theta = Data["n_theta"]
n_theta_0_90 = np.append(
        np.append(
            np.append(
                  n_theta, np.flip(n_theta, axis=0), axis=0), 
                    n_theta, axis=0),
                        np.flip(n_theta, axis=0), axis=0)


# 45°
n_theta_45 = np.append(
        np.append(
            np.append(
                  n_theta, np.flip(-n_theta, axis=0), axis=0), 
                    n_theta, axis=0),
                        np.flip(-n_theta, axis=0), axis=0)


        
theta_values = Data["theta_values"]
theta_values = np.append(np.append(np.append(theta_values, np.pi/2 + theta_values), np.pi + theta_values), 3/2*np.pi + theta_values)

Lambda_R = Data["Lambda_R"]
Lambda_D = Data["Lambda_D"]
Delta = float(Data["Delta"])
w_0 = Data["w_0"]
mu = Data["mu"]
L_x = Data["L_x"]
B = Data["B"]
g_xx = Data["g_xx"]
g_yy = Data["g_yy"]
g_xy = Data["g_xy"]
g_yx = Data["g_yx"]

# fig, ax = plt.subplots(subplot_kw={'projection': 'polar'})
fig, ax = plt.subplots()

# ax.plot(theta_values, n_theta_0_90[:,0], "-o",  label=r"$n_{s,xx}$")
ax.plot(theta_values, n_theta_0_90[:,1], "-o",  label=r"$n_{s,yy}$")
# ax.plot(theta_values, n_theta_45[:,2], "-o",  label=r"$n_{s,xy}$")
# ax.plot(theta_values, n_theta_45[:,3], "-o",  label=r"$n_{s,yx}$")
# ax.plot(theta_values, (n_theta[:,0]+n_theta[:,1]+n_theta[:,2]+n_theta[:,3])/2, "-o",  label=r"$n_{s,xx}+n_{s,yx}+n_{s,xy}n_{s,yy}$")

ax.set_title(r"$\lambda_R=$" + f"{Lambda_R:.2}"
             +r"; $\Delta=$" + f"{Delta}"
             +r"; $B=$" + f"{B:.3}"
             + r"; $\mu$"+f"={mu}"
             +r"; $w_0$"+f"={w_0}"
             +r"; $L_x=$"+f"{L_x}"+ "\n"
             +r"$\lambda_D=$" + f"{np.round(Lambda_D,2)}"
             + r"; $g_{xx}=$" + f"{g_xx}"
             + r"; $g_{yy}=$" + f"{g_yy}"
             + r"; $g_{xy}=$" + f"{g_xy}"
             + r"; $g_{yx}=$" + f"{g_yx}")

ax.set_ylabel(r"$n_s(\theta)$")
ax.legend()
plt.tight_layout()
plt.show()