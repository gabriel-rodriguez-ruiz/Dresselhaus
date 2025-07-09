# -*- coding: utf-8 -*-
"""
Created on Mon Jun 30 10:10:22 2025

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
# file_to_open = data_folder / "n_By_mu_-34.900000000000006_L=2500_h=0.001_B_y_in_(0.0-0.8)_Delta=0.08_lambda_R=0.56_lambda_D=0_g_xx=1_g_xy=0_g_yy=1_g_yx=0_theta=1.57_points=24_beta=1000_T=True_chi=0.npz"
file_to_open = data_folder / "n_By_mu_-34.900000000000006_L=2500_h=0.001_B_y_in_(0.0-0.8)_Delta=0.08_lambda_R=0.56_lambda_D=0_g_xx=1_g_xy=0_g_yy=1_g_yx=0_theta=1.57_points=24_beta=1000_T=True_chi=0.npz"

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

Delta_Z_x = B_values * (g_xx*np.cos(theta))

# phi = np.arctan(g_yy/g_xx)

fig, axs = plt.subplots(2,1)
# ax.plot(B_values/Delta, n_B_y[:,0], "-ok",  label=r"$n_{s,xx}$")
# ax.plot(B_values/Delta, n_B_y[:,1], "-or",  label=r"$n_{s,yy}$")
# ax.plot(B_values/Delta, n_B_y[:,2], "-o",  label=r"$n_{s,xy}$")
# ax.plot(B_values/Delta, n_B_y[:,3], "-o",  label=r"$n_{s,yx}$")
# ax.plot(B_values/Delta, 1/2*(n_B_y[:,0] + n_B_y[:,1]), "-o",  label=r"$n_{s,x'x'}$")

axs[0].plot(B_values/Delta,  1/2*(n_B_y[:,0] + n_B_y[:,1] + n_B_y[:,2] + n_B_y[:,3]),
                             "-o",  label=r"$n_{s,x'x'}(\varphi=\pi/2, 4.9GHz)$",
                             color="black",
                             markersize=10)
axs[1].plot(B_values/Delta, n_B_y[:,0],
                             "-*",  label=r"$n_{s,x'x'}(\varphi=\pi/2, 5.7GHz)$",
                             color="black",
                             markersize=10)
    

# file_to_open = data_folder / "n_By_mu_-34.900000000000006_L=1500_h=0.001_B_y_in_(0.0-0.8)_Delta=0.08_lambda_R=0.56_lambda_D=0.19_g_xx=1_g_xy=0_g_yy=1_g_yx=0_theta=-0.79_points=24_beta=False_T=False.npz"
Data = np.load(file_to_open)
n_B_y = Data["n_B_y"]

# ax.plot(B_values/Delta, n_B_y[:,1],
#         "-o",  label=r"$n_{s,x'x'}(\varphi=-\pi/4, 4.9GHz)$", color="orange",
#         markersize=10)
# ax.plot(B_values/Delta, 1/2*(n_B_y[:,0] + n_B_y[:,1] + n_B_y[:,2] + n_B_y[:,3]),
#                              "-*",
#                              label=r"$n_{s,x'x'}(\varphi=-\pi/4, 5.7GHz)$", color="orange")

file_to_open = data_folder / "n_By_mu_-34.900000000000006_L=2500_h=0.001_B_y_in_(0.0-0.8)_Delta=0.08_lambda_R=0.56_lambda_D=0_g_xx=1_g_xy=0_g_yy=1_g_yx=0_theta=0_points=24_beta=1000_T=False_chi=0.npz"
Data = np.load(file_to_open)
n_B_y = Data["n_B_y"]   
B_values = Data["B_values"]

axs[0].plot(B_values/Delta, 1/2*(n_B_y[:,0] + n_B_y[:,1] + n_B_y[:,2] + n_B_y[:,3]),
        "-o",  label=r"$n_{s,x'x'}(\varphi=0, 4.9GHz)$", color="red")
axs[1].plot(B_values/Delta, n_B_y[:,0],
                             "-*",  label=r"$n_{s,x'x'}(\varphi=0, 5.7GHz)$", color="red")

data_folder = Path(r"C:\Users\Gabriel\OneDrive - Universidad Nacional de San Martin\Doctorado-DESKTOP-JBOMLCA\Archivos\Data_19_06_25\Data")
file_to_open = data_folder / "n_By_mu_-34.900000000000006_L=1500_h=0.001_B_y_in_(0.0-0.8)_Delta=0.08_lambda_R=0.56_lambda_D=0_g_xx=1_g_xy=0_g_yy=1_g_yx=0_theta=0.79_points=24_beta=False_T=False.npz"
# file_to_open = data_folder / "n_By_mu_-349.0_L=2500_h=0.001_B_y_in_(0.0-0.8)_Delta=0.08_lambda_R=29.87_lambda_D=0_g_xx=1_g_xy=0_g_yy=1_g_yx=0_theta=0.79_points=24_beta=False_T=False.npz"

Data = np.load(file_to_open)
n_B_y = Data["n_B_y"]
B_values = Data["B_values"]

axs[0].plot(B_values/Delta, 1/2*(n_B_y[:,0] + n_B_y[:,1] + n_B_y[:,2] + n_B_y[:,3]),
        "-o",  label=r"$n_{s,x'x'}(\varphi=\pi/4, 4.9GHz)$", color="green")
axs[1].plot(B_values/Delta, n_B_y[:,0],
                             "-*",  label=r"$n_{s,x'x'}(\varphi=\pi/4, 5.7GHz)$", color="green")

data_folder = Path("./Data")
file_to_open = data_folder / "n_By_mu_-34.900000000000006_L=2500_h=0.001_B_y_in_(0.0-0.8)_Delta=0.08_lambda_R=0.56_lambda_D=0_g_xx=1_g_xy=0_g_yy=1_g_yx=0_theta=2.36_points=24_beta=1000_T=False_chi=0.npz"

Data = np.load(file_to_open)
n_B_y = Data["n_B_y"]
B_values = Data["B_values"]

axs[0].plot(B_values/Delta, 1/2*(n_B_y[:,0] + n_B_y[:,1] + n_B_y[:,2] + n_B_y[:,3]),
        "-o",  label=r"$n_{s,x'x'}(\varphi=3\pi/4, 4.9GHz)$", color="darkviolet")
axs[1].plot(B_values/Delta, n_B_y[:,0],
                             "-*",  label=r"$n_{s,x'x'}(\varphi=\pi/2, 5.7GHz)$", color="darkviolet")

# ax.plot(B_values/Delta, 1/2*( n_B_y[:,0]+n_B_y[:,1]+n_B_y[:,2]+n_B_y[:,3]), "-o",  label=r"$n_{s,x'x'}(\theta=3\pi/4)$", color="darkviolet")

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




# axs[0].set_title(r"$\lambda_R=$" + f"{np.round(Lambda_R,2)}"
#              +r"; $\Delta=$" + f"{Delta}"
#              +r"; $\theta=$" + f"{np.round(theta,2)}"
#              + r"; $\mu$"+f"={np.round(mu, 3)}"
#              +r"; $w_0$"+f"={w_0}"
#              +r"; $L_x=$"+f"{L_x}"
#              +f"; h={h}" + "\n"
#              + r"$\lambda_D=$" + f"{Lambda_D}"
#              + r"; $g_{xx}=$" + f"{g_xx}"
#              + r"; $g_{yy}=$" + f"{g_yy}"
#              + r"; $g_{xy}=$" + f"{g_xy}"
#              + r"$; g_{yx}=$" + f"{g_yx}"
#              + r"; $\beta=$" + f"{beta}")

axs[0].set_xlabel(r"$\frac{B}{\Delta*}$")
axs[0].set_ylabel(r"$n_s$")
axs[0].set_title(r"4.9 GHz Resonator (45°)")
# axs[0].legend()

axs[1].set_xlabel(r"$\frac{B}{\Delta*}$")
axs[1].set_ylabel(r"$n_s$")
axs[1].set_title(r"5.7 GHz Resonator (0°)")

# axs[1].legend()
plt.tight_layout()
# plt.show(block=False)

