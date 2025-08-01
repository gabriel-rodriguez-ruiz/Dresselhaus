#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed May 29 17:05:28 2024

@author: gabriel
"""

import matplotlib.pyplot as plt
from pathlib import Path
import numpy as np
import os

abspath = os.path.abspath(__file__)
dname = os.path.dirname(abspath)
os.chdir(dname)

# data_folder = Path(r"/home/gabriel/OneDrive/Doctorado-DESKTOP-JBOMLCA/Archivos/Data_19_06_25/Data")
data_folder = Path("./Data")
# file_to_open = data_folder / "n_By_mu_-34.900000000000006_L=2500_h=0.001_B_y_in_(0.0-0.8)_Delta=0.08_lambda_R=0.56_lambda_D=0_g_xx=1_g_xy=0_g_yy=1_g_yx=0_theta=1.57_points=24_beta=1000_T=False_chi=0.79.npz"
# file_to_open = data_folder / "n_By_mu_-34.900000000000006_L=1500_h=0.001_B_y_in_(0.0-0.8)_Delta=0.08_lambda_R=0.56_lambda_D=0.19_g_xx=1_g_xy=0_g_yy=1_g_yx=0_theta=1.57_points=24_beta=False_T=False.npz"
# file_to_open = data_folder / "n_By_mu_-349.0_L=3000_h=0.001_B_y_in_(0.0-0.96)_Delta=0.08_lambda_R=10_lambda_D=0_g_xx=1_g_xy=0_g_yy=1_g_yx=0_theta=0.79_points=24_beta=1000_T=True_chi=0.npz"
file_to_open = data_folder / "n_By_mu_-34.900000000000006_L=3000_h=0.001_B_y_in_(0.064-0.104)_Delta=0.08_lambda_R=0.06_lambda_D=0_g_xx=1_g_xy=0_g_yy=1_g_yx=0_theta=1.57_points=48_beta=1000_T=False_chi=0.npz"
# file_to_open = data_folder / "n_By_mu_-34.900000000000006_L=1000_h=0.01_B_y_in_(0.0-0.96)_Delta=0.08_lambda_R=29.87_lambda_D=0_g_xx=1_g_xy=0_g_yy=1_g_yx=0_theta=0.79_points=16_beta=1000_T=True_chi=0.npz"


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
T = Data["T"]

Delta_Z_x = B_values * (g_xx*np.cos(theta))
 
# phi = np.arctan(g_yy/g_xx)

fig, ax = plt.subplots()
ax.plot(B_values/Delta, n_B_y[:,0], "-*g",  label=r"$n_{s,xx}(\theta=$"+f"{np.round(theta,2)}, 5.7GHz, L=3000)")
ax.plot(B_values/Delta, n_B_y[:,1], "-sg",  label=r"$n_{s,yy}(\theta=$"+f"{np.round(theta,2)}, \lambda_R=0.056)")
# ax.plot(B_values/Delta, n_B_y[:,2], "-o",  label=r"$n_{s,xy}(\theta=$"+f"{np.round(theta,2)}, 5.7GHz)")
# ax.plot(B_values/Delta, n_B_y[:,3], "-o",  label=r"$n_{s,yx}$")
# ax.plot(B_values/Delta, 1/2*(n_B_y[:,0] + n_B_y[:,1] + n_B_y[:,2] + n_B_y[:,3]), "-og",  label=r"$n_{s,x'x'}(\theta=\pi/4,4.9GHz)$")
# ax.plot(B_values/Delta, 1/2*(n_B_y[:,0] + n_B_y[:,1] - n_B_y[:,2] - n_B_y[:,3]), "-oc",  label=r"$n_{s,y'y'}(\theta=\pi/4,4.9GHz)$")

# ax.plot(B_values/Delta, n_B_y[:,1],
#                              "-o",  label=r"$n_{s,x'x'}(\varphi=\pi/2, 4.9GHz)$",
#                              color="black",
#                              markersize=10)
# ax.plot(B_values/Delta, 1/2*(n_B_y[:,0] + n_B_y[:,1] + n_B_y[:,2] + n_B_y[:,3]),
#                              "-*",  label=r"$n_{s,x'x'}(\varphi=\pi/2, 5.7GHz)$",
#                              color="black",
#                              markersize=10)
phi = np.arctan(-np.pi/2)
# ax.plot(B_values/Delta, np.cos(phi)**2*n_B_y[:,0] + np.sin(phi)**2*n_B_y[:,1] + np.sin(phi)*np.cos(phi)*(n_B_y[:,2] + n_B_y[:,3]),
#                              "-*",  label=r"$n_{s,x'x'}(\varphi=\pi/2, 5.7GHz)$",
#                              color="black",
#                              markersize=10)
    
# data_folder = Path(r"/home/gabriel/OneDrive/Doctorado-DESKTOP-JBOMLCA/Archivos/Data_19_06_25/Data")
data_folder = Path("./Data")
file_to_open = data_folder / "n_By_mu_-34.900000000000006_L=3000_h=0.001_B_y_in_(0.064-0.104)_Delta=0.08_lambda_R=0_lambda_D=0_g_xx=1_g_xy=0_g_yy=1_g_yx=0_theta=1.57_points=48_beta=1000_T=False_chi=0.npz"
# file_to_open = data_folder / "n_By_mu_-349.0_L=2500_h=0.001_B_y_in_(0.0-0.48)_Delta=0.08_lambda_R=0.56_lambda_D=0_g_xx=1_g_xy=0_g_yy=1_g_yx=0_theta=1.57_points=24_beta=1000_T=True.npz"

Data = np.load(file_to_open)
n_B_y = Data["n_B_y"]
B_values = Data["B_values"]

ax.plot(B_values/Delta, n_B_y[:,1],
        "-*k",  label=r"$n_{s,xx}(\theta=\pi/2,\lambda_R=0, L=3000)$")
# ax.plot(B_values/Delta, n_B_y[:,2], "-o",  label=r"$n_{s,xy}(\theta=\pi/2, 5.7GHz)$")
ax.plot(B_values/Delta, n_B_y[:,0],
        "-*r",  label=r"$n_{s,yy}(\theta=\pi/2,5.7GHz)$")
# ax.plot(B_values/Delta, 1/2*(n_B_y[:,0] + n_B_y[:,1] + n_B_y[:,2] + n_B_y[:,3]),
#                              "-ok",
#                              label=r"$n_{s,x'x'}(\theta=\pi/2,4.9GHz)$")

# file_to_open = data_folder / "n_By_mu_-34.900000000000006_L=2500_h=0.001_B_y_in_(0.064-0.104)_Delta=0.08_lambda_R=0.06_lambda_D=0_g_xx=1_g_xy=0_g_yy=1_g_yx=0_theta=1.57_points=48_beta=1000_T=False_chi=0.npz"
# Data = np.load(file_to_open)
# n_B_y = Data["n_B_y"]   
# B_values = Data["B_values"]

# ax.plot(B_values/Delta, n_B_y[:,0],
#         "-o",  label=r"$n_{s,x'x'}(\varphi=1.57, 4.9GHz,L=2500)$", color="red")
# ax.plot(B_values/Delta, 1/2*(n_B_y[:,0] + n_B_y[:,1] + n_B_y[:,2] + n_B_y[:,3]),
#                              "-*",  label=r"$n_{s,x'x'}(\varphi=0, 5.7GHz)$", color="red")
# ax.plot(B_values/Delta, np.cos(phi)**2*n_B_y[:,0] + np.sin(phi)**2*n_B_y[:,1] + np.sin(phi)*np.cos(phi)*(n_B_y[:,2] + n_B_y[:,3]),
#                              "-*",  label=r"$n_{s,x'x'}(\varphi=0, 5.7GHz)$",
#                              color="red",
#                              markersize=10)
# file_to_open = data_folder / "n_By_mu_-34.900000000000006_L=1500_h=0.001_B_y_in_(0.0-0.8)_Delta=0.08_lambda_R=0.56_lambda_D=0.19_g_xx=1_g_xy=0_g_yy=1_g_yx=0_theta=0.79_points=24_beta=False_T=False.npz"
# file_to_open = data_folder / "n_By_mu_-34.900000000000006_L=1000_h=0.001_B_y_in_(0.064-0.104)_Delta=0.08_lambda_R=0.06_lambda_D=0_g_xx=1_g_xy=0_g_yy=1_g_yx=0_theta=1.57_points=48_beta=1000_T=False_chi=0.npz"

# Data = np.load(file_to_open)
# n_B_y = Data["n_B_y"]
# B_values = Data["B_values"]

# ax.plot(B_values/Delta, n_B_y[:,0],
#         "-o",  label=r"$n_{s,x'x'}(\varphi=\pi/2, 5.7GHz, L=1000)$", color="green")
# ax.plot(B_values/Delta, 1/2*(n_B_y[:,0] + n_B_y[:,1] + n_B_y[:,2] + n_B_y[:,3]),
#                              "-*",  label=r"$n_{s,x'x'}(\varphi=\pi/4, 5.7GHz)$", color="green")
# ax.plot(B_values/Delta, np.cos(phi)**2*n_B_y[:,0] + np.sin(phi)**2*n_B_y[:,1] + np.sin(phi)*np.cos(phi)*(n_B_y[:,2] + n_B_y[:,3]),
#                              "-*",  label=r"$n_{s,x'x'}(\varphi=0, 5.7GHz)$",
#                              color="green",
#                              markersize=10)

# file_to_open = data_folder / "n_By_mu_-34.900000000000006_L=1500_h=0.001_B_y_in_(0.0-0.8)_Delta=0.08_lambda_R=0.56_lambda_D=0.19_g_xx=1_g_xy=0_g_yy=1_g_yx=0_theta=2.36_points=24_beta=False_T=False.npz"
# Data = np.load(file_to_open)
# n_B_y = Data["n_B_y"]
# B_values = Data["B_values"]
    
# ax.plot(B_values/Delta, n_B_y[:,1],
#         "-o",  label=r"$n_{s,x'x'}(\varphi=3\pi/4, 4.9GHz)$", color="darkviolet")
# ax.plot(B_values/Delta, 1/2*(n_B_y[:,0] + n_B_y[:,1] + n_B_y[:,2] +  n_B_y[:,3]),
#                              "-*",  label=r"$n_{s,x'x'}(\varphi=\pi/2, 5.7GHz)$", color="darkviolet")

# ax.plot(B_values/Delta, 1/2*( n_B_y[:,0]+n_B_y[:,1]+n_B_y[:,2]+n_B_y[:,3]), "-o",  label=r"$n_{s,x'x'}(\theta=3\pi/4)$", color="darkviolet")

# ax.plot(B_values/Delta, np.cos(phi)**2*n_B_y[:,0] + np.sin(phi)**2*n_B_y[:,1] + np.sin(phi)*np.cos(phi)*(n_B_y[:,2] + n_B_y[:,3]),
#                              "-*",  label=r"$n_{s,x'x'}(\varphi=0, 5.7GHz)$",
#                              color="darkviolet",
#                              markersize=10)
# file_to_open = data_folder / "n_By_mu_-34.900000000000006_L=1500_h=0.001_B_y_in_(0.48-0.8)_Delta=0.08_lambda_R=0.56_lambda_D=0_g_xx=1.5_g_xy=0_g_yy=1_g_yx=0_theta=0_points=16_beta=False_T=False.npz"
# Data = np.load(file_to_open)
# n_B_y = Data["n_B_y"]
# B_values = Data["B_values"]

# ax.plot(B_values/Delta, 1/2*( n_B_y[:,0]+n_B_y[:,1]+n_B_y[:,2]+n_B_y[:,3]), "-o",  label=r"$n_{s,x'x'}(\theta=0)$", color="red")

# file_to_open = data_folder / "n_By_mu_-34.900000000000006_L=1500_h=0.001_B_y_in_(0.48-0.8)_Delta=0.08_lambda_R=0.56_lambda_D=0_g_xx=1.5_g_xy=0_g_yy=1_g_yx=0_theta=1.57_points=16_beta=False_T=False.npz"
# Data = np.load(file_to_open)
# n_B_y = Data["n_B_y"]
# B_values = Data["B_values"]

# ax.plot(B_values/Delta, 1/2*( n_B_y[:,0]+n_B_y[:,1]+n_B_y[:,2]+n_B_y[:,3]), "-o",  label=r"$n_{s,x'x'}(\theta=\pi/2)$", color="black")


    

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
             + r"; $\beta=$" + f"{beta}"
             + f"; T={T}")

ax.set_xlabel(r"$\frac{B}{\Delta*}$")
ax.set_ylabel(r"$n_s$")
ax.legend()

# plt.tight_layout()
# plt.show(block=False)

