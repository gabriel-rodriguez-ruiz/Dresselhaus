#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jul 23 16:15:23 2025

@author: gabriel
"""

from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit


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


def anisotropic_B_square_angle_shift(angle, alpha):
    return alpha * (np.cos(2*angle*2*np.pi/360) - 1)    # in units of B


initial_parameters = [1e-2]
popt_25_mT, pcov_25_mT = curve_fit(anisotropic_B_square_angle_shift, angle_25_mT.dropna(), delta_ns_25.dropna(),
                                          p0=initial_parameters)
angle_fit = np.linspace(0, 2*np.pi, 1000)*360/(2*np.pi)
ax.plot(angle_fit, anisotropic_B_square_angle_shift(angle_fit, popt_25_mT[0]), "-")
ax.plot(angle_fit, 4*anisotropic_B_square_angle_shift(angle_fit, popt_25_mT[0]), "-")
ax.plot(angle_fit, 8*anisotropic_B_square_angle_shift(angle_fit, popt_25_mT[0]), "-")
ax.plot(angle_fit, 16*anisotropic_B_square_angle_shift(angle_fit, popt_25_mT[0]), "-")

popt_50_mT, pcov_50_mT = curve_fit(anisotropic_B_square_angle_shift, angle_50_mT.dropna(), delta_ns_50.dropna(),
                                          p0=initial_parameters)

angle_fit = np.linspace(0, 2*np.pi, 1000)*360/(2*np.pi)
ax.plot(angle_fit, 1/4*anisotropic_B_square_angle_shift(angle_fit, popt_50_mT[0]), "--")
ax.plot(angle_fit, anisotropic_B_square_angle_shift(angle_fit, popt_50_mT[0]), "--")
ax.plot(angle_fit, 2*anisotropic_B_square_angle_shift(angle_fit, popt_50_mT[0]), "--")
ax.plot(angle_fit, 4*anisotropic_B_square_angle_shift(angle_fit, popt_50_mT[0]), "--")


#%% 5.7 GHz resonator

data_folder = Path(r"Files/data gabriel")

file_path = data_folder / 'field_dep_5_7_GHz_0deg.xlsx'

xls = pd.ExcelFile(file_path) # use r before absolute file path 
sheetX = xls.parse(0) #2 is the sheet number+1 thus if the file has only 1 sheet write 0 in paranthesis

n_s_0 = sheetX['n_s 0°'].dropna()
field_0 = sheetX["fields 0°"].dropna()
n_s_0_error = sheetX["n_s 0° err"].dropna()

n_s_45 = sheetX['n_s 45°'].dropna()
field_45 = sheetX["fields 45°"].dropna()
n_s_45_error = sheetX["n_s 45° err"].dropna()

n_s_90 = sheetX['n_s 90°'].dropna()
field_90 = sheetX["fields 90°"].dropna()
n_s_90_error = sheetX["n_s 90° err"].dropna()

n_s_135 = sheetX['n_s 135°'].dropna()
field_135 = sheetX["fields 135°"].dropna()
n_s_135_error = sheetX["n_s 135° err"].dropna()

fig, ax = plt.subplots()
ax.errorbar(field_0, n_s_0, yerr=n_s_0_error, label=r"$n_s(0°)$", color="red", fmt="*")
ax.errorbar(field_45, n_s_45, yerr=n_s_45_error, label=r"$n_s(45°)$", color="green", fmt="*")
ax.errorbar(field_90, n_s_90, yerr=n_s_90_error, label=r"$n_s(90°)$", color="black", fmt="*")
ax.errorbar(field_135, n_s_135, yerr=n_s_135_error, label=r"$n_s(135°)$", color="darkviolet", fmt="*")

def interpolation_for_theory(x):
    return np.interp(x, field_0, n_s_0)

def anisotropic_B_square_field_shift(B, alpha, angle, limit):
    return alpha * (np.cos(2*angle*2*np.pi/360) - 1) * (B/limit)**2 #+ interpolation_for_theory(B)

# field_model_25_mT = np.linspace(0, 0.025)
# field_model_50_mT = np.linspace(0, 0.05)

field_model_25_mT = np.linspace(0, 0.15)
field_model_50_mT = np.linspace(0, 0.15)

# ax.plot(field_model_25_mT, anisotropic_B_square_field_shift(field_model_25_mT,
#                                                             popt_25_mT[0],
#                                                             0, 0.025), "or")
# ax.plot(field_model_50_mT, anisotropic_B_square_field_shift(field_model_50_mT,
#                                                             popt_50_mT[0],
#                                                             0, 0.05), "or")
ax.plot(field_model_25_mT, anisotropic_B_square_field_shift(field_model_25_mT,
                                                            popt_25_mT[0],
                                                            90, 0.025), "-k")
ax.plot(field_model_50_mT, anisotropic_B_square_field_shift(field_model_50_mT,
                                                            popt_50_mT[0],
                                                            90, 0.05), "--k")
ax.plot(field_model_25_mT, anisotropic_B_square_field_shift(field_model_25_mT,
                                                            popt_25_mT[0],
                                                            45, 0.025), "-g")
ax.plot(field_model_50_mT, anisotropic_B_square_field_shift(field_model_50_mT,
                                                            popt_50_mT[0],
                                                            45, 0.05), "--g")
ax.plot(field_model_25_mT, anisotropic_B_square_field_shift(field_model_25_mT,
                                                            popt_25_mT[0],
                                                            135, 0.025), color="darkviolet")
ax.plot(field_model_50_mT, anisotropic_B_square_field_shift(field_model_50_mT,
                                                            popt_50_mT[0],
                                                            135, 0.05), linestyle="dashed", color="darkviolet")

from matplotlib.lines import Line2D
custom_lines = [Line2D([0], [0], color="blue", lw=2),
                Line2D([0], [0], color="blue", lw=2, linestyle="dashed")]

ax.axvline(0.025, linestyle="dashed")
ax.axvline(0.05, linestyle="dashed")
ax.axvline(0.075, linestyle="dashed")
ax.axvline(0.1, linestyle="dashed")

ax.set_title("5.7 GHz Resonator, 0°")
ax.set_xlabel(r"$B$ [$T$]")
ax.set_ylabel(r"$n_s$")
legend = ax.legend()
ax.legend(custom_lines, ["Angular fit until 25mT", "Angular fit until 50mT"], loc="lower left")
ax.add_artist(legend)
plt.tight_layout()
plt.show()

#%%

data_folder = Path("./Data")
file_to_open = data_folder / "n_By_mu_-34.900000000000006_L=3000_h=0.001_B_y_in_(0.0-0.24)_Delta=0.08_lambda_R=0.06_lambda_D=0_g_xx=1_g_xy=0_g_yy=1_g_yx=0_theta=1.57_points=48_beta=1000_T=False_chi=0.npz"
Data = np.load(file_to_open)

n_B_y = Data["n_B_y"]
B_values = Data["B_values"]
Delta = Data["Delta"]
# fig, ax = plt.subplots()
# ax.plot(field_model_50_mT, n_B_y[:, 0], "o")
ax.plot(B_values/Delta*0.05, n_B_y[:, 1]- n_B_y[0, 1], "or")

scale_factor_parallel = 0.001
scale_factor_perpendicular = 0.02


for i, B in enumerate(B_values/Delta*0.05):
    if B<0.05:
        ax.scatter(B, anisotropic_B_square_field_shift(B,
                                                 popt_50_mT[0],
                                                 90, 0.05))
    else:
        ax.scatter(B, anisotropic_B_square_field_shift(B,
                                                 popt_50_mT[0],
                                                 90, 0.05) + scale_factor_perpendicular*n_B_y[i, 0]/n_B_y[0, 0])



def anisotropic_B_square_parallel_shift(B, alpha):
    return  alpha*B**2

initial_parameters = [-1]
popt_parallel, pcov_parallel = curve_fit(anisotropic_B_square_parallel_shift, field_0[:10], n_s_0[:10],
                                          p0=initial_parameters)
# ax.plot(B_values/Delta*0.05, popt_parallel[0]*(B_values/Delta*0.05)**2)
# ax.plot(B_values/Delta*0.05, scale_factor*(n_B_y[:, 1]-n_B_y[0, 1]), "o")
ax.plot(B_values/Delta*0.05, scale_factor_parallel*(n_B_y[:, 1]-n_B_y[0, 1])+popt_parallel[0]*(B_values/Delta*0.05)**2, "o")

file_to_open = "superfluid_density.npz"
data = np.load(file_to_open)
superfluid_density_polinomial = data["superfluid_density_polinomial"]
B_values = data["B_values"]

scale_factor = 0.2

for i, B in enumerate(B_values/Delta*0.05):
    if B<0.05:
        ax.scatter(B, anisotropic_B_square_field_shift(B,
                                                 popt_50_mT[0],
                                                 90, 0.05))
    else:
        ax.scatter(B, anisotropic_B_square_field_shift(B,
                                                 popt_50_mT[0],
                                                 90, 0.05) + scale_factor*superfluid_density_polinomial[i]/superfluid_density_polinomial[0])


# ax.plot(B_values/Delta*0.05, scale_factor*(n_B_y[:, 1]-n_B_y[0, 1]), "o")




#%% Angle dependence 4.9 GHz

data_folder = Path(r"Files/data gabriel")

# file_path = data_folder / 'angle_dep_4_9_GHz_45deg.xlsx'
file_path = data_folder / '4_9_GHz_angle_dep_norm_45deg.xls'

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

def anisotropic_B_square_angle_shift_45(angle, alpha):
    return alpha * ( np.cos(2*angle*2*np.pi/360 + np.pi/2) + 1)    # in units of B


initial_parameters = [1e-2]
popt_25_mT, pcov_25_mT = curve_fit(anisotropic_B_square_angle_shift_45, angle_25_mT.dropna(), delta_ns_25.dropna(),
                                          p0=initial_parameters)
angle_fit = np.linspace(0, 2*np.pi, 1000)*360/(2*np.pi)
ax.plot(angle_fit, anisotropic_B_square_angle_shift_45(angle_fit, popt_25_mT[0]), "-")
ax.plot(angle_fit, 4*anisotropic_B_square_angle_shift_45(angle_fit, popt_25_mT[0]), "-")
ax.plot(angle_fit, 8*anisotropic_B_square_angle_shift_45(angle_fit, popt_25_mT[0]), "-")
ax.plot(angle_fit, 16*anisotropic_B_square_angle_shift_45(angle_fit, popt_25_mT[0]), "-")

popt_50_mT, pcov_50_mT = curve_fit(anisotropic_B_square_angle_shift_45, angle_50_mT.dropna(), delta_ns_50.dropna(),
                                          p0=initial_parameters)

angle_fit = np.linspace(0, 2*np.pi, 1000)*360/(2*np.pi)
ax.plot(angle_fit, 1/4*anisotropic_B_square_angle_shift_45(angle_fit, popt_50_mT[0]), "--")
ax.plot(angle_fit, anisotropic_B_square_angle_shift_45(angle_fit, popt_50_mT[0]), "--")
ax.plot(angle_fit, 2*anisotropic_B_square_angle_shift_45(angle_fit, popt_50_mT[0]), "--")
ax.plot(angle_fit, 4*anisotropic_B_square_angle_shift_45(angle_fit, popt_50_mT[0]), "--")

ax.set_title("4.9 GHz Resonator, 45°")
ax.set_xlabel(r"Angle")
ax.set_ylabel(r"$n_s$")
ax.legend()
plt.tight_layout()
plt.show()

#%% 4.9 GHz resonator

data_folder = Path(r"Files/data gabriel")

file_path = data_folder / 'field_dep_4_9_GHz_45deg.xlsx'

xls = pd.ExcelFile(file_path) # use r before absolute file path 
sheetX = xls.parse(0) #2 is the sheet number+1 thus if the file has only 1 sheet write 0 in paranthesis

n_s_0 = sheetX['n_s 0°'].dropna()
field_0 = sheetX["fields 0°"].dropna().dropna()
n_s_0_error = sheetX["n_s 0° err"].dropna()

n_s_45 = sheetX['n_s 45°'].dropna()
field_45 = sheetX["fields 45°"].dropna()
n_s_45_error = sheetX["n_s 45° err"].dropna()

n_s_90 = sheetX['n_s 90°'].dropna()
field_90 = sheetX["fields 90°"].dropna()
n_s_90_error = sheetX["n_s 90° err"].dropna()

n_s_135 = sheetX['n_s 135°'].dropna()
field_135 = sheetX["fields 135°"].dropna()
n_s_135_error = sheetX["n_s 135° err"].dropna()

fig, ax = plt.subplots()
ax.errorbar(field_0, n_s_0, yerr=n_s_0_error, label=r"$n_s(0°)$", color="red", fmt="o")
ax.errorbar(field_45, n_s_45, yerr=n_s_45_error, label=r"$n_s(45°)$", color="green", fmt="o")
ax.errorbar(field_90, n_s_90, yerr=n_s_90_error, label=r"$n_s(90°)$", color="black", fmt="o")
ax.errorbar(field_135, n_s_135, yerr=n_s_135_error, label=r"$n_s(135°)$", color="darkviolet", fmt="o")

ax.axvline(0.025, linestyle="dashed")
ax.axvline(0.05, linestyle="dashed")
ax.axvline(0.075, linestyle="dashed")
ax.axvline(0.1, linestyle="dashed")


def anisotropic_B_square_field_shift_45(B, alpha, angle, limit):
    return alpha * (np.cos(2*angle*2*np.pi/360 + np.pi/2) + 1) * (B/limit)**2

ax.plot(field_model_25_mT, anisotropic_B_square_field_shift_45(field_model_25_mT,
                                                            popt_25_mT[0],
                                                            90, 0.025), "-k")
ax.plot(field_model_50_mT, anisotropic_B_square_field_shift_45(field_model_50_mT,
                                                            popt_50_mT[0],
                                                            90, 0.05), "--k")
ax.plot(field_model_25_mT, anisotropic_B_square_field_shift_45(field_model_25_mT,
                                                            popt_25_mT[0],
                                                            45, 0.025), "-g")
ax.plot(field_model_50_mT, anisotropic_B_square_field_shift_45(field_model_50_mT,
                                                            popt_50_mT[0],
                                                            45, 0.05), "--g")
ax.plot(field_model_25_mT, anisotropic_B_square_field_shift_45(field_model_25_mT,
                                                            popt_25_mT[0],
                                                            135, 0.025), color="darkviolet")
ax.plot(field_model_50_mT, anisotropic_B_square_field_shift_45(field_model_50_mT,
                                                            popt_50_mT[0],
                                                            135, 0.05), linestyle="dashed", color="darkviolet")

#%%

data_folder = Path("./Data")
file_to_open = data_folder / "n_By_mu_-34.900000000000006_L=3000_h=0.001_B_y_in_(0.0-0.24)_Delta=0.08_lambda_R=0.06_lambda_D=0_g_xx=1_g_xy=0_g_yy=1_g_yx=0_theta=1.57_points=48_beta=1000_T=False_chi=0.npz"
Data = np.load(file_to_open)

n_B_y = Data["n_B_y"]
B_values = Data["B_values"]
Delta = Data["Delta"]
# fig, ax = plt.subplots()
# ax.plot(field_model_50_mT, n_B_y[:, 0], "o")
# ax.plot(B_values/Delta, n_B_y[:, 1], "o")

scale_factor_perpendicular = 0.05
scale_factor_parallel = 0.0001
for i, B in enumerate(B_values/Delta*0.05):
    if B<0.05:
        ax.scatter(B, anisotropic_B_square_field_shift_45(B,
                                                 popt_50_mT[0],
                                                 135, 0.05))
    else:
        ax.scatter(B, anisotropic_B_square_field_shift_45(B,
                                                 popt_50_mT[0],
                                                 135, 0.05) + scale_factor_perpendicular*n_B_y[i, 0]/n_B_y[0, 0])


# ax.plot(B_values/Delta*0.05, scale_factor_parallel*(n_B_y[:, 1]-n_B_y[0, 1]), "o")

def anisotropic_B_square_parallel_shift(B, alpha):
    return  alpha*B**2

initial_parameters = [-1]
popt_parallel, pcov_parallel = curve_fit(anisotropic_B_square_parallel_shift, field_45[:10], n_s_45[:10],
                                          p0=initial_parameters)
ax.plot(B_values/Delta*0.05, popt_parallel[0]*(B_values/Delta*0.05)**2, "g-")
# ax.plot(B_values/Delta*0.05, scale_factor*(n_B_y[:, 1]-n_B_y[0, 1]), "o")
ax.plot(B_values/Delta*0.05, scale_factor_parallel*(n_B_y[:, 1]-n_B_y[0, 1])+popt_parallel[0]*(B_values/Delta*0.05)**2, "o")


#%%

file_to_open = "superfluid_density.npz"
data = np.load(file_to_open)
superfluid_density_polinomial = data["superfluid_density_polinomial"]
B_values = data["B_values"]

scale_factor_perpendicular = 0.05
scale_factor_parallel = 0.05

for i, B in enumerate(B_values/Delta*0.05):
    if B<0.05:
        ax.scatter(B, anisotropic_B_square_field_shift_45(B,
                                                 popt_50_mT[0],
                                                 135, 0.05))
    else:
        ax.scatter(B, anisotropic_B_square_field_shift_45(B,
                                                 popt_50_mT[0],
                                                 135, 0.05) + scale_factor_perpendicular*superfluid_density_polinomial[i]/superfluid_density_polinomial[0])


# ax.plot(B_values/Delta*0.05, scale_factor*(n_B_y[:, 1]-n_B_y[0, 1]), "o")




#%% Angle dependence 9.7 GHz

data_folder = Path(r"Files/data gabriel")

file_path = data_folder / 'angle_dep_9_7GHz_90deg.xls'

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

initial_parameters = [1e-2]
popt_25_mT, pcov_25_mT = curve_fit(anisotropic_B_square_angle_shift, angle_25_mT.dropna(), delta_ns_25.dropna(),
                                          p0=initial_parameters)
angle_fit = np.linspace(0, 2*np.pi, 1000)*360/(2*np.pi)
ax.plot(angle_fit, anisotropic_B_square_angle_shift(angle_fit, popt_25_mT[0]), "-")
ax.plot(angle_fit, 4*anisotropic_B_square_angle_shift(angle_fit, popt_25_mT[0]), "-")
ax.plot(angle_fit, 8*anisotropic_B_square_angle_shift(angle_fit, popt_25_mT[0]), "-")
ax.plot(angle_fit, 16*anisotropic_B_square_angle_shift(angle_fit, popt_25_mT[0]), "-")

popt_50_mT, pcov_50_mT = curve_fit(anisotropic_B_square_angle_shift, angle_50_mT.dropna(), delta_ns_50.dropna(),
                                          p0=initial_parameters)

angle_fit = np.linspace(0, 2*np.pi, 1000)*360/(2*np.pi)
ax.plot(angle_fit, 1/4*anisotropic_B_square_angle_shift(angle_fit, popt_50_mT[0]), "--")
ax.plot(angle_fit, anisotropic_B_square_angle_shift(angle_fit, popt_50_mT[0]), "--")
ax.plot(angle_fit, 2*anisotropic_B_square_angle_shift(angle_fit, popt_50_mT[0]), "--")
ax.plot(angle_fit, 4*anisotropic_B_square_angle_shift(angle_fit, popt_50_mT[0]), "--")

ax.set_title("9.7 GHz Resonator, 90°")
ax.set_xlabel(r"Angle")
ax.set_ylabel(r"$n_s$")
ax.legend()
plt.show()

#%% 9.7 GHz resonator

data_folder = Path(r"Files/data gabriel")

file_path = data_folder / 'field_dep_9_7GHz_90deg.xls'

xls = pd.ExcelFile(file_path) # use r before absolute file path 
sheetX = xls.parse(0) #2 is the sheet number+1 thus if the file has only 1 sheet write 0 in paranthesis

n_s_0 = sheetX['n_s 0°'].dropna()
field_0 = sheetX["fields 0°"].dropna()
n_s_0_error = sheetX["n_s 0° err"].dropna()

n_s_45 = sheetX['n_s 45°'].dropna()
field_45 = sheetX["fields 45°"].dropna()[:-1]
n_s_45_error = sheetX["n_s 45° err"].dropna()

n_s_90 = sheetX['n_s 90°'].dropna()
field_90 = sheetX["fields 90°"].dropna()
n_s_90_error = sheetX["n_s 90° err"].dropna()

n_s_135 = sheetX['n_s 135°'].dropna()
field_135 = sheetX["fields 135°"].dropna()
n_s_135_error = sheetX["n_s 135° err"].dropna()

fig, ax = plt.subplots()
ax.errorbar(field_0, n_s_0, yerr=n_s_0_error, label=r"$n_s(0°)$", color="red", fmt="*")
ax.errorbar(field_45, n_s_45, yerr=n_s_45_error, label=r"$n_s(45°)$", color="green", fmt="*")
ax.errorbar(field_90, n_s_90, yerr=n_s_90_error, label=r"$n_s(90°)$", color="black", fmt="*")
ax.errorbar(field_135, n_s_135, yerr=n_s_135_error, label=r"$n_s(135°)$", color="darkviolet", fmt="*")

ax.plot(field_model_25_mT, anisotropic_B_square_field_shift(field_model_25_mT,
                                                            -popt_25_mT[0],
                                                            90, 0.025), "-r")
ax.plot(field_model_50_mT, anisotropic_B_square_field_shift(field_model_50_mT,
                                                            -popt_50_mT[0],
                                                            90, 0.05), "--r")
ax.plot(field_model_25_mT, anisotropic_B_square_field_shift(field_model_25_mT,
                                                            -popt_25_mT[0],
                                                            45, 0.025), "-g")
ax.plot(field_model_50_mT, anisotropic_B_square_field_shift(field_model_50_mT,
                                                            -popt_50_mT[0],
                                                            45, 0.05), "--g")
ax.plot(field_model_25_mT, anisotropic_B_square_field_shift(field_model_25_mT,
                                                            -popt_25_mT[0],
                                                            135, 0.025), color="darkviolet")
ax.plot(field_model_50_mT, anisotropic_B_square_field_shift(field_model_50_mT,
                                                            -popt_50_mT[0],
                                                            135, 0.05), linestyle="dashed", color="darkviolet")

from matplotlib.lines import Line2D
custom_lines = [Line2D([0], [0], color="blue", lw=2),
                Line2D([0], [0], color="blue", lw=2, linestyle="dashed")]

ax.axvline(0.025, linestyle="dashed")
ax.axvline(0.05, linestyle="dashed")
ax.axvline(0.075, linestyle="dashed")
ax.axvline(0.1, linestyle="dashed")

ax.set_xlabel(r"$B$ [$T$]")
ax.set_ylabel(r"$n_s$")
legend = ax.legend()
ax.legend(custom_lines, ["Angular fit until 25mT", "Angular fit until 50mT"], loc="lower left")
ax.add_artist(legend)
ax.set_title("9.7 GHz Resonator, 90°")
plt.tight_layout()
plt.show()