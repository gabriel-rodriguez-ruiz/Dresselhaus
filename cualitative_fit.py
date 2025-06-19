#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jun 18 18:21:48 2025

@author: gabriel
"""

import matplotlib.pyplot as plt
from pathlib import Path
import numpy as np
import os

abspath = os.path.abspath(__file__)
dname = os.path.dirname(abspath)
os.chdir(dname)

data_folder = Path("Data/")
# file_to_open = data_folder / "n_By_mu_-34.900000000000006_L=1500_h=0.001_B_y_in_(0.0-0.8)_Delta=0.08_lambda_R=0.56_lambda_D=0_g_xx=1_g_xy=0_g_yy=1_g_yx=0_theta=0.79_points=24_beta=False_T=False.npz"
file_to_open = data_folder / "n_By_mu_-34.900000000000006_L=2000_h=0.001_B_y_in_(0.0-0.8)_Delta=0.08_lambda_R=0.56_lambda_D=0.19_g_xx=1_g_xy=1_g_yy=1_g_yx=-1_theta=0.79_points=24_beta=False_T=False.npz"


Data = np.load(file_to_open)

n_B_y = Data["n_B_y"]
B_values = Data["B_values"]
Lambda_R = Data["Lambda_R"]
Lambda_D = Data["Lambda_D"]
Delta = Data["Delta"]
theta = Data["theta"]
w_0 = Data["w_0"]
mu = Data["mu"]
L_x = Data["L_x"]
h = Data["h"]
g_xx = Data["g_xx"]
g_yy = Data["g_yy"]
g_xy = Data["g_xy"]
g_yx = Data["g_yx"]
beta = Data["beta"]

fig, ax = plt.subplots()
# ax.plot(B_values/Delta, n_B_y[:,0], "-*g",  label=r"$n_{s,xx}$")
# ax.plot(B_values/Delta, n_B_y[:,1], "-or",  label=r"$n_{s,yy}$")
# ax.plot(B_values/Delta, n_B_y[:,2], "-o",  label=r"$n_{s,xy}$")
# ax.plot(B_values/Delta, n_B_y[:,3], "-o",  label=r"$n_{s,yx}$")
ax.plot(B_values/Delta, 1/2*(n_B_y[:,0] + n_B_y[:,1] + n_B_y[:,2] + n_B_y[:,3]), "-og",  label=r"$n_{s,x'x'}(\theta=\pi/4)$")
ax.plot(B_values/Delta, n_B_y[:,0], "-*g",  label=r"$n_{s,x'x'}(\theta=\pi/4)$")

# ax.plot(B_values/Delta, n_B_y[:,0], "-or", label=r"$n_{s,x'x'}(\theta=0)$")
# ax.plot(B_values/Delta, n_B_y[:,1], "-ok",  label=r"$n_{s,x'x'}(\theta=\pi/2)$")
# ax.plot(B_values/Delta, 1/2*(n_B_y[:,0] + n_B_y[:,1] - n_B_y[:,2] - n_B_y[:,3]),
#         "-o", label=r"$n_{s,x'x'}(\theta=3\pi/4)$", color="darkviolet")


data_folder = Path("Data/")
file_to_open = data_folder / "n_By_mu_-34.900000000000006_L=2000_h=0.001_B_y_in_(0.0-0.8)_Delta=0.08_lambda_R=0.56_lambda_D=0.19_g_xx=1_g_xy=1_g_yy=1_g_yx=-1_theta=2.36_points=24_beta=False_T=False.npz"
Data = np.load(file_to_open)

n_B_y = Data["n_B_y"]
B_values = Data["B_values"]
ax.plot(B_values/Delta, 1/2*(n_B_y[:,0] + n_B_y[:,1] + n_B_y[:,2] + n_B_y[:,3]),
        "-o",  label=r"$n_{s,x'x'}(\theta=3\pi/4)$",
        color="darkviolet")
ax.plot(B_values/Delta, n_B_y[:,0], "-*",  label=r"$n_{s,x'x'}(\theta=3\pi/4)$",
        color="darkviolet")


data_folder = Path("Data/")
file_to_open = data_folder / "n_By_mu_-34.900000000000006_L=2000_h=0.001_B_y_in_(0.0-0.8)_Delta=0.08_lambda_R=0.56_lambda_D=0.19_g_xx=1_g_xy=1_g_yy=1_g_yx=-1_theta=0_points=24_beta=False_T=False.npz"
Data = np.load(file_to_open)

n_B_y = Data["n_B_y"]
B_values = Data["B_values"]
ax.plot(B_values/Delta, 1/2*(n_B_y[:,0] + n_B_y[:,1] + n_B_y[:,2] + n_B_y[:,3]),
        "-o",  label=r"$n_{s,x'x'}(\theta=0)$",
        color="red")
ax.plot(B_values/Delta, n_B_y[:,0], "-*",  label=r"$n_{s,x'x'}(\theta=0)$",
        color="red")

data_folder = Path("Data/")
file_to_open = data_folder / "n_By_mu_-34.900000000000006_L=2000_h=0.001_B_y_in_(0.0-0.8)_Delta=0.08_lambda_R=0.56_lambda_D=0.19_g_xx=1_g_xy=1_g_yy=1_g_yx=-1_theta=1.57_points=24_beta=False_T=False.npz"
Data = np.load(file_to_open)

n_B_y = Data["n_B_y"]
B_values = Data["B_values"]
ax.plot(B_values/Delta, 1/2*(n_B_y[:,0] + n_B_y[:,1] - n_B_y[:,2] - n_B_y[:,3]),
        "-o",  label=r"$n_{s,x'x'}(\theta=\pi/2)$",
        color="black")
ax.plot(B_values/Delta, n_B_y[:,0], "-*",  label=r"$n_{s,x'x'}(\theta=\pi/2)$",
        color="black")

ax.set_title(r"$\lambda_R=$" + f"{np.round(Lambda_R,2)}"
             +r"; $\Delta=$" + f"{Delta}"
             +r"; $\theta=$" + f"{np.round(theta,2)}"
             + r"; $\mu$"+f"={np.round(mu, 3)}"
             +r"; $w_0$"+f"={w_0}"
             +r"; $L_x=$"+f"{L_x}"
             +f"; h={h}" + "\n"
             + r"$\lambda_D=$" + f"{Lambda_D}"
             + r"; $g_{xx}=$" + f"{g_xx}"
             + r"; $g_{yy}=$" + f"{g_yy}"
             + r"; $g_{xy}=$" + f"{g_xy}"
             + r"$; g_{yx}=$" + f"{g_yx}"
             + r"; $\beta=$" + f"{beta}")

ax.set_xlabel(r"$\frac{B}{\Delta*}$")
ax.set_ylabel(r"$n_s$")
ax.legend()
