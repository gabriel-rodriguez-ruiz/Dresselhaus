# -*- coding: utf-8 -*-
"""
Created on Wed Jul  2 15:14:46 2025

@author: Gabriel
"""

import matplotlib.pyplot as plt
from pathlib import Path
import numpy as np
import os

abspath = os.path.abspath(__file__)
dname = os.path.dirname(abspath)
os.chdir(dname)

# data_folder = Path(r"C:\Users\Gabriel\OneDrive - Universidad Nacional de San Martin\Doctorado-DESKTOP-JBOMLCA\Archivos\Data_19_06_25\Data")
data_folder = Path("./Data")
file_to_open = data_folder / "n_By_mu_-34.900000000000006_L=2500_h=0.001_B_y_in_(0.0-0.8)_Delta=0.08_lambda_R=0.56_lambda_D=0_g_xx=1_g_xy=0_g_yy=1_g_yx=0_theta=0_points=24_beta=1000_T=False_chi=0.npz"

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

# phi = np.arctan(g_yy/g_xx)

fig, ax = plt.subplots()
# ax.plot(B_values/Delta, n_B_y[:,0], "-ok",  label=r"$n_{s,xx}$")
# ax.plot(B_values/Delta, n_B_y[:,1], "-or",  label=r"$n_{s,yy}$")
# ax.plot(B_values/Delta, n_B_y[:,2], "-o",  label=r"$n_{s,xy}$")
# ax.plot(B_values/Delta, n_B_y[:,3], "-o",  label=r"$n_{s,yx}$")

ax.plot(B_values/Delta, n_B_y[:,1],
                             "-o",  label=r"$n_{s,x'x'}(\varphi=0, 4.9GHz,$"+r"$\lambda_R=$"+f"{Lambda_R})",
                             color="black",
                             markersize=10)
ax.plot(B_values/Delta, 1/2*(n_B_y[:,0] + n_B_y[:,1] + n_B_y[:,2] + n_B_y[:,3]),
                             "-*",  label=r"$n_{s,x'x'}(\varphi=0, 5.7GHz$"+r"$\lambda_R=$"+f"{Lambda_R})",
                             color="black",
                             markersize=10)

data_folder = Path("./Data")
file_to_open = data_folder / "n_By_mu_-34.900000000000006_L=2500_h=0.001_B_y_in_(0.0-0.48)_Delta=0.08_lambda_R=0.1_lambda_D=0_g_xx=1_g_xy=0_g_yy=1_g_yx=0_theta=0_points=24_beta=1000_T=False_chi=0.npz"

Data = np.load(file_to_open)

n_B_y = Data["n_B_y"]
B_values = Data["B_values"]
Lambda_R = Data["Lambda_R"]

ax.plot(B_values/Delta, n_B_y[:,1],
                             "--o",  label=r"$n_{s,x'x'}(\varphi=0, 4.9GHz$"+r"$\lambda_R=$"+f"{Lambda_R})",
                             color="red",
                             markersize=10)
ax.plot(B_values/Delta, 1/2*(n_B_y[:,0] + n_B_y[:,1] + n_B_y[:,2] + n_B_y[:,3]),
                             "--*",  label=r"$n_{s,x'x'}(\varphi=0, 5.7GHz$"+"$\lambda_R=$"+f"{Lambda_R})",
                             color="red",
                             markersize=10)

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

# plt.tight_layout()
# plt.show(block=False)

