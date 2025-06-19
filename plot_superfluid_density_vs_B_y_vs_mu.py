#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Mar 31 13:51:13 2025

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
file_to_open = data_folder / "n_By_mu_-39_L=1000_h=0.01_B_y_in_(0.0-1.2)_Delta=0.2_lambda=0.56.npz"
Data = np.load(file_to_open)

file_to_open2 = data_folder / "n_By_mu_-38_L=1000_h=0.01_B_y_in_(0.0-1.2)_Delta=0.2_lambda_R=0.56_lambda_D=0_g_xx=1_g_xy=0_g_yy=1_g_yx=0_theta=1.57_points=24.npz"
Data2 = np.load(file_to_open2)

file_to_open3 = data_folder / "n_By_mu_-39.5_L=1000_h=0.01_B_y_in_(0.0-1.2)_Delta=0.2_lambda_R=0.56_lambda_D=0_g_xx=1_g_xy=0_g_yy=1_g_yx=0_theta=1.57_points=24.npz"
Data3 = np.load(file_to_open3)

file_to_open4 = data_folder / "n_By_mu_-40_L=1000_h=0.01_B_y_in_(0.0-1.2)_Delta=0.2_lambda=0.56.npz"
Data4 = np.load(file_to_open4)

file_to_open5 = data_folder / "n_By_mu_-37_L=1000_h=0.01_B_y_in_(0.0-1.2)_Delta=0.2_lambda_R=0.56_lambda_D=0_g_xx=1_g_xy=0_g_yy=1_g_yx=0_theta=1.57_points=24.npz"
Data5 = np.load(file_to_open5)

file_to_open6 = data_folder / "n_By_mu_-36_L=1000_h=0.01_B_y_in_(0.0-1.2)_Delta=0.2_lambda_R=0.56_lambda_D=0_g_xx=1_g_xy=0_g_yy=1_g_yx=0_theta=1.57_points=40.npz"
Data6 = np.load(file_to_open6)

file_to_open7 = data_folder / "n_By_mu_-35_L=1000_h=0.01_B_y_in_(0.0-1.2)_Delta=0.2_lambda_R=0.56_lambda_D=0_g_xx=1_g_xy=0_g_yy=1_g_yx=0_theta=1.57_points=56.npz"
Data7 = np.load(file_to_open7)

file_to_open8 = data_folder / "n_By_mu_-34_L=1000_h=0.01_B_y_in_(0.0-1.2)_Delta=0.2_lambda_R=0.56_lambda_D=0_g_xx=1_g_xy=0_g_yy=1_g_yx=0_theta=1.57_points=24.npz"
Data8 = np.load(file_to_open8)

file_to_open9 = data_folder / "n_By_mu_-33_L=1000_h=0.01_B_y_in_(0.0-1.2)_Delta=0.2_lambda_R=0.56_lambda_D=0_g_xx=1_g_xy=0_g_yy=1_g_yx=0_theta=1.57_points=24.npz"
Data9 = np.load(file_to_open9)

file_to_open10 = data_folder / "n_By_mu_-30_L=1000_h=0.01_B_y_in_(0.0-1.2)_Delta=0.2_lambda_R=0.56_lambda_D=0_g_xx=1_g_xy=0_g_yy=1_g_yx=0_theta=1.57_points=24.npz"
Data10 = np.load(file_to_open10)

n_B_y = Data["n_B_y"]
B_values = Data["B_values"]
# Lambda_R = Data["Lambda_R"]
Lambda = Data["Lambda"]

# Lambda_D = Data["Lambda_D"]
Delta = Data["Delta"]
theta = Data["theta"]
w_0 = Data["w_0"]
mu = Data["mu"]
L_x = Data["L_x"]
h = Data["h"]
# g_xx = Data["g_xx"]
# g_yy = Data["g_yy"]
# g_xy = Data["g_xy"]
# g_yx = Data["g_yx"]

n_B_y2 = Data2["n_B_y"]
B_values2 = Data2["B_values"]
Lambda_R2 = Data2["Lambda_R"]
Lambda_D2 = Data2["Lambda_D"]
Delta2 = Data2["Delta"]
theta2 = Data2["theta"]
w_02 = Data2["w_0"]
mu2 = Data2["mu"]
L_x2 = Data2["L_x"]
h2 = Data2["h"]
g_xx2 = Data2["g_xx"]
g_yy2 = Data2["g_yy"]
g_xy2 = Data2["g_xy"]
g_yx2 = Data2["g_yx"]

n_B_y3 = Data3["n_B_y"]
B_values3 = Data3["B_values"]
Lambda_R3 = Data3["Lambda_R"]
Lambda_D3 = Data3["Lambda_D"]
Delta3 = Data3["Delta"]
theta3 = Data3["theta"]
w_03 = Data3["w_0"]
mu3 = Data3["mu"]
L_x3 = Data3["L_x"]
h3 = Data3["h"]
g_xx3 = Data3["g_xx"]
g_yy3 = Data3["g_yy"]
g_xy3 = Data3["g_xy"]
g_yx3 = Data3["g_yx"]

n_B_y4 = Data4["n_B_y"]
B_values4 = Data4["B_values"]
# Lambda_R = Data["Lambda_R"]
Lambda4 = Data4["Lambda"]

# Lambda_D = Data["Lambda_D"]
Delta4 = Data4["Delta"]
theta4 = Data4["theta"]
w_04 = Data4["w_0"]
mu4 = Data4["mu"]
L_x4 = Data4["L_x"]
# h = Data["h"]
# g_xx = Data["g_xx"]
# g_yy = Data["g_yy"]
# g_xy = Data["g_xy"]
# g_yx = Data["g_yx"]

n_B_y5 = Data5["n_B_y"]
B_values5 = Data5["B_values"]
Lambda_R5 = Data5["Lambda_R"]
Lambda_D5 = Data5["Lambda_D"]
Delta5 = Data5["Delta"]
theta5 = Data5["theta"]
w_05 = Data5["w_0"]
mu5 = Data5["mu"]
L_x5 = Data5["L_x"]
h5 = Data5["h"]
g_xx5 = Data5["g_xx"]
g_yy5 = Data5["g_yy"]
g_xy5 = Data5["g_xy"]
g_yx5 = Data5["g_yx"]

n_B_y6 = Data6["n_B_y"]
B_values6 = Data6["B_values"]
Lambda_R6 = Data6["Lambda_R"]
Lambda_D6 = Data6["Lambda_D"]
Delta6 = Data6["Delta"]
theta6 = Data6["theta"]
w_06 = Data6["w_0"]
mu6 = Data6["mu"]
L_x6= Data6["L_x"]
h6 = Data6["h"]
g_xx6 = Data6["g_xx"]
g_yy6 = Data6["g_yy"]
g_xy6 = Data6["g_xy"]
g_yx6 = Data6["g_yx"]

n_B_y7 = Data7["n_B_y"]
B_values7 = Data7["B_values"]
Lambda_R7 = Data7["Lambda_R"]
Lambda_D7 = Data7["Lambda_D"]
Delta7 = Data7["Delta"]
theta7 = Data7["theta"]
w_07 = Data7["w_0"]
mu7 = Data7["mu"]
L_x7= Data7["L_x"]
h7 = Data7["h"]
g_xx7 = Data7["g_xx"]
g_yy7 = Data7["g_yy"]
g_xy7 = Data7["g_xy"]
g_yx7 = Data7["g_yx"]

n_B_y8 = Data8["n_B_y"]
B_values8 = Data8["B_values"]
Lambda_R8 = Data8["Lambda_R"]
Lambda_D8 = Data8["Lambda_D"]
Delta8 = Data8["Delta"]
theta8 = Data8["theta"]
w_08 = Data8["w_0"]
mu8 = Data8["mu"]
L_x8= Data8["L_x"]
h8 = Data8["h"]
g_xx8 = Data8["g_xx"]
g_yy8 = Data8["g_yy"]
g_xy8 = Data8["g_xy"]
g_yx8 = Data8["g_yx"]

n_B_y9 = Data9["n_B_y"]
B_values9 = Data9["B_values"]
Lambda_R9 = Data9["Lambda_R"]
Lambda_D9 = Data9["Lambda_D"]
Delta9 = Data9["Delta"]
theta9 = Data9["theta"]
w_09 = Data9["w_0"]
mu9 = Data9["mu"]
L_x9= Data9["L_x"]
h9 = Data9["h"]
g_xx9 = Data9["g_xx"]
g_yy9 = Data9["g_yy"]
g_xy9 = Data9["g_xy"]
g_yx9 = Data9["g_yx"]

n_B_y10 = Data10["n_B_y"]
B_values10 = Data10["B_values"]
Lambda_R10 = Data10["Lambda_R"]
Lambda_D10 = Data10["Lambda_D"]
Delta10 = Data10["Delta"]
theta10 = Data10["theta"]
w_010 = Data10["w_0"]
mu10 = Data10["mu"]
L_x10= Data10["L_x"]
h10 = Data10["h"]
g_xx10 = Data10["g_xx"]
g_yy10 = Data10["g_yy"]
g_xy10 = Data10["g_xy"]
g_yx10 = Data10["g_yx"]


fig, ax = plt.subplots()
# ax.plot(B_values/Delta, n_B_y[:,0], "-o",  label=r"$n_{s,xx}$")
ax.plot(B_values/Delta, n_B_y[:,0]/n_B_y[0,0], "-o",  label=r"$n_{s}/n_s(0)(\theta=\pi/2,\mu=$"+f"{mu})")
# ax.plot(B_values/Delta, n_B_y[:,2], "-o",  label=r"$n_{s,xy}$")
# ax.plot(B_values/Delta, n_B_y[:,3], "-o",  label=r"$n_{s,yx}$")
# ax.plot(B_values/Delta, (n_B_y[:,0]+n_B_y[:,1]+n_B_y[:,2]+n_B_y[:,3])/2, "-o",  label=r"$n_{s,xx}+n_{s,yy}+n_{s,xy}+n_{s,yx}$")

# ax.plot(B_values/Delta, n_B_y[:,0], "-o",  label=r"$n_{s,xx}$")
ax.plot(B_values2/Delta2, n_B_y2[:,0]/n_B_y2[0,0], "-s",  label=r"$n_{s}/n_s(0)(\theta=\pi/2,\mu=$"+f"{mu2})")
# ax.plot(B_values/Delta, n_B_y[:,2], "-o",  label=r"$n_{s,xy}$")
# ax.plot(B_values/Delta, n_B_y[:,3], "-o",  label=r"$n_{s,yx}$")
# ax.plot(B_values/Delta, (n_B_y[:,0]+n_B_y[:,1]+n_B_y[:,2]+n_B_y[:,3])/2, "-o",  label=r"$n_{s,xx}+n_{s,yy}+n_{s,xy}+n_{s,yx}$")

ax.plot(B_values3/Delta3, n_B_y3[:,0]/n_B_y3[0,0], "-v",  label=r"$n_{s}/n_s(0)(\theta=\pi/2,\mu=$"+f"{mu3})")
ax.plot(B_values4/Delta4, n_B_y4[:,0]/n_B_y4[0,0], "-D",  label=r"$n_{s}/n_s(0)(\theta=\pi/2,\mu=$"+f"{mu4})")
ax.plot(B_values5/Delta5, n_B_y5[:,0]/n_B_y5[0,0], "-p",  label=r"$n_{s}/n_s(0)(\theta=\pi/2,\mu=$"+f"{mu5})")
ax.plot(B_values6/Delta6, n_B_y6[:,0]/n_B_y6[0,0], "-^",  label=r"$n_{s}/n_s(0)(\theta=\pi/2,\mu=$"+f"{mu6})")
ax.plot(B_values7/Delta7, n_B_y7[:,0]/n_B_y7[0,0], "-d",  label=r"$n_{s}/n_s(0)(\theta=\pi/2,\mu=$"+f"{mu7})")
ax.plot(B_values8/Delta8, n_B_y8[:,0]/n_B_y8[0,0], "-d",  label=r"$n_{s}/n_s(0)(\theta=\pi/2,\mu=$"+f"{mu8})")
ax.plot(B_values9/Delta9, n_B_y9[:,0]/n_B_y9[0,0], "-.",  label=r"$n_{s}/n_s(0)(\theta=\pi/2,\mu=$"+f"{mu9})")
ax.plot(B_values10/Delta10, n_B_y10[:,0]/n_B_y10[0,0], "-.",  label=r"$n_{s}/n_s(0)(\theta=\pi/2,\mu=$"+f"{mu10})")

# ax.set_title(r"$\lambda_R=$" + f"{np.round(Lambda_R,2)}"
#              +r"; $\Delta=$" + f"{Delta}"
#              +r"; $\theta=$" + f"{np.round(theta,2)}"
#              + r"; $\mu$"+f"={mu}"
#              +r"; $w_0$"+f"={w_0}"
#              +r"; $L_x=$"+f"{L_x}"
#              +f"; h={h}" + "\n"
#              + r"$\lambda_D=$" + f"{Lambda_D}"
#              + r"; $g_{xx}=$" + f"{g_xx}"
#              + r"; $g_{yy}=$" + f"{g_yy}"
#              + r"; $g_{xy}=$" + f"{g_xy}"
#              + r"$; g_{yx}=$" + f"{g_yx}")

ax.set_xlabel(r"$\frac{\mu_BgB}{\Delta}$")
ax.set_ylabel(r"$n_s$")
ax.legend()

# plt.tight_layout()
plt.show()

#%% Save figure
saving_folder = Path("/home/gabriel/Dropbox/Figures/Fitting to superfluid density")
file_name = f'theoretical_ns_for_various_mu_lambda={np.round(Lambda,2)}_Delta={Delta}_' \
            f'L={L_x}_h={h}_B_y_in_({np.min(B_values)}-{np.round(np.max(B_values),3)})_theta={np.round(theta,2)}.png'

file_path = saving_folder / file_name
fig.savefig(file_path, dpi=300)