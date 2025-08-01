#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jul 15 10:55:34 2025

@author: gabriel
"""

import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import pandas as pd

#%% Angle dependence 5.7 GHz

data_folder = Path(r"Files/data gabriel")

file_path = data_folder / 'angle_dep_5_7_GHz_0deg.xlsx'

xls = pd.ExcelFile(file_path) # use r before absolute file path 
sheetX = xls.parse(0) #2 is the sheet number+1 thus if the file has only 1 sheet write 0 in paranthesis

angle_25_mT = sheetX['angle 25 mT']  
delta_ns_25 = sheetX["delta ns 25"]
delta_ns_err_25 = sheetX["delta ns err 25"]

angle_50_mT = sheetX['angle 50 mT']  
delta_ns_50 = sheetX["delta ns 50"]
delta_ns_err_50 = sheetX["delta ns err 50"]

angle_75_mT = sheetX['angle 75 mT']  
delta_ns_75 = sheetX["delta ns 75"]
delta_ns_err_75 = sheetX["delta ns err 75"]

angle_100_mT = sheetX['angle 100 mT']  
delta_ns_100 = sheetX["delta ns 100"]
delta_ns_err_100 = sheetX["delta ns err 100"]

fig, ax = plt.subplots()
ax.errorbar(angle_25_mT, delta_ns_25, yerr=delta_ns_err_25, label=r"$n_s(25mT)$", fmt="-o")
ax.errorbar(angle_50_mT, delta_ns_50, yerr=delta_ns_err_50, label=r"$n_s(50mT)$", fmt="-o")
ax.errorbar(angle_75_mT, delta_ns_75, yerr=delta_ns_err_75, label=r"$n_s(75mT)$", fmt="-o")
ax.errorbar(angle_100_mT, delta_ns_100, yerr=delta_ns_err_100, label=r"$n_s(100mT)$", fmt="-o")



ax.set_title("5.7 GHz Resonator, 0°")
ax.set_xlabel(r"Angle")
ax.set_ylabel(r"$n_s$")
ax.legend()
plt.show()

#%% Theoretical calculation

data_folder = Path("Data/")
file_to_open = data_folder / "n_theta_mu_-34.900000000000006_L=1000_h=0.001_theta_in_(0.0-1.571)B=0.08_Delta=0.08_lambda_R=0.056_lambda_D=0_g_xx=1_g_xy=0_g_yy=1_g_yx=0_points=16.npz"

Data = np.load(file_to_open, allow_pickle=True)

n_theta = (Data["n_theta"] - Data["n_theta"][0])

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

def interpolation_for_theory(x):
    return [
            np.interp(x, theta_values, n_theta_0_90[:, 0]),       
            np.interp(x, theta_values, n_theta_0_90[:, 1])
            ]

# fig, ax = plt.subplots(subplot_kw={'projection': 'polar'})
fig, ax = plt.subplots()


ax.plot(theta_values, n_theta_0_90[:,0], "-o",  label=r"$n_{s,xx}$", color="mediumseagreen")
ax.plot(theta_values, n_theta_0_90[:,1], "-o",  label=r"$n_{s,yy}$")
ax.plot(theta_values, 1/2*(n_theta_45[:,0]+n_theta_45[:,1]+n_theta_45[:,2]+n_theta_45[:,3]),
        "-o",  label=r"$n_{s,x'x'}$", color="yellowgreen")
ax.plot(theta_values, interpolation_for_theory(theta_values)[0])
ax.plot(theta_values, interpolation_for_theory(theta_values)[1])


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

#%% Fitting
from scipy.optimize import curve_fit

data_folder = Path("Data/")
file_to_open = [#data_folder / "n_theta_mu_-34.900000000000006_L=2000_h=0.001_theta_in_(0.0-1.571)B=0.06_Delta=0.08_lambda_R=0.056_lambda_D=0_g_xx=1_g_xy=0_g_yy=1_g_yx=0_points=16_chi_equal_theta.npz",
                #data_folder / "n_theta_mu_-34.900000000000006_L=1000_h=0.001_theta_in_(0.0-1.571)B=0.068_Delta=0.08_lambda_R=0.056_lambda_D=0_g_xx=1_g_xy=0_g_yy=1_g_yx=0_points=16.npz",
                #data_folder / "n_theta_mu_-34.900000000000006_L=1000_h=0.001_theta_in_(0.0-1.571)B=0.0704_Delta=0.08_lambda_R=0.056_lambda_D=0_g_xx=1_g_xy=0_g_yy=1_g_yx=0_points=16.npz",
                data_folder / "n_theta_mu_-34.900000000000006_L=1000_h=0.001_theta_in_(0.0-1.571)B=0.07200000000000001_Delta=0.08_lambda_R=0.056_lambda_D=0_g_xx=1_g_xy=0_g_yy=1_g_yx=0_points=16.npz",
                data_folder / "n_theta_mu_-34.900000000000006_L=1000_h=0.001_theta_in_(0.0-1.571)B=0.076_Delta=0.08_lambda_R=0.056_lambda_D=0_g_xx=1_g_xy=0_g_yy=1_g_yx=0_points=16.npz",
                data_folder / "n_theta_mu_-34.900000000000006_L=1000_h=0.001_theta_in_(0.0-1.571)B=0.0784_Delta=0.08_lambda_R=0.056_lambda_D=0_g_xx=1_g_xy=0_g_yy=1_g_yx=0_points=16.npz",
                #data_folder / "n_theta_mu_-34.900000000000006_L=1000_h=0.001_theta_in_(0.0-1.571)B=0.08_Delta=0.08_lambda_R=0.056_lambda_D=0_g_xx=1_g_xy=0_g_yy=1_g_yx=0_points=16.npz",
                #data_folder / "n_theta_mu_-34.900000000000006_L=1000_h=0.001_theta_in_(0.0-1.571)B=0.084_Delta=0.08_lambda_R=0.056_lambda_D=0_g_xx=1_g_xy=0_g_yy=1_g_yx=0_points=16.npz",
                data_folder / "n_theta_mu_-34.900000000000006_L=1000_h=0.001_theta_in_(0.0-1.571)B=0.09_Delta=0.08_lambda_R=0.056_lambda_D=0_g_xx=1_g_xy=0_g_yy=1_g_yx=0_points=16.npz",
                # data_folder / "n_theta_mu_-34.900000000000006_L=1000_h=0.001_theta_in_(0.0-1.571)B=0.1_Delta=0.08_lambda_R=0.056_lambda_D=0_g_xx=1_g_xy=0_g_yy=1_g_yx=0_points=16.npz",
                # data_folder / "n_theta_mu_-34.900000000000006_L=1000_h=0.001_theta_in_(0.0-1.571)B=0.10400000000000001_Delta=0.08_lambda_R=0.056_lambda_D=0_g_xx=1_g_xy=0_g_yy=1_g_yx=0_points=16.npz",
                # data_folder / "n_theta_mu_-34.900000000000006_L=1000_h=0.001_theta_in_(0.0-1.571)B=0.12_Delta=0.08_lambda_R=0.056_lambda_D=0_g_xx=1_g_xy=0_g_yy=1_g_yx=0_points=16.npz"
                ]
angle = [angle_25_mT, angle_50_mT, angle_75_mT, angle_100_mT]
delta_ns = [delta_ns_25, delta_ns_50, delta_ns_75, delta_ns_100]
delta_ns_err = [delta_ns_err_25, delta_ns_err_50, delta_ns_err_75, delta_ns_err_100]
name = [r"$n_s(25mT)$", r"$n_s(50mT)$", r"$n_s(75mT)$", r"$n_s(100mT)$"]
standard_deviation_5_7GHz = np.zeros(4)

fig, ax = plt.subplots()

for i, file in enumerate(file_to_open):
    Data = np.load(file, allow_pickle=True)
    n_theta = (Data["n_theta"] - Data["n_theta"][0])
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
    theta_values = theta_values * 360 / (2*np.pi)
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
    
    def interpolation_for_theory(x):
        return [
                np.interp(x, theta_values, n_theta_0_90[:, 0]),       
                np.interp(x, theta_values, n_theta_0_90[:, 1])
                ]
    def model_5_7GHz(x, a):
        return a*interpolation_for_theory(x)[0]
    
    x_model_5_7GHz = np.linspace(0, 2*np.pi, 1000) * 360 / (2*np.pi)
    initial_parameters = [1]
    fitted_angles = pd.concat([angle[i][0:4], angle[i][9:16], angle[i][20:25]])
    fitted_delta_ns = pd.concat([delta_ns[i][0:4], delta_ns[i][9:16], delta_ns[i][20:25]])
    popt_5_7GHz, pcov_5_7GHz = curve_fit(model_5_7GHz, fitted_angles, fitted_delta_ns,
                                              p0=initial_parameters)
    standard_deviation_5_7GHz[i] = np.sqrt(np.diag(pcov_5_7GHz))[0]
    ax.errorbar(angle[i], delta_ns[i], yerr=delta_ns_err[i], label=name[i], fmt="-o")
    ax.plot(x_model_5_7GHz, model_5_7GHz(x_model_5_7GHz, popt_5_7GHz[0]), "--*")

ax.set_xlabel("Angle")
ax.set_ylabel(r"$n_s$")
# plt.vlines(x=angle[0][9], ymin=min(delta_ns[3]), ymax=0)
# plt.vlines(x=angle[0][16], ymin=min(delta_ns[3]), ymax=0)
plt.axvspan(angle[0][16], angle[0][20], facecolor='b', alpha=0.2)
plt.axvspan(angle[0][4], angle[0][8], facecolor='b', alpha=0.2)
# plt.axvspan(angle[0][20], angle[0][25], facecolor='b', alpha=0.3)