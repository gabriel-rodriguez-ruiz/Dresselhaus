#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jun 17 18:25:45 2025

@author: gabriel
"""

from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

#%%
import os

abspath = os.path.abspath(__file__)
dname = os.path.dirname(abspath)
os.chdir(dname)
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

from scipy.optimize import curve_fit

# data_folder = Path(r"C:\Users\Gabriel\OneDrive - Universidad Nacional de San Martin\Doctorado-DESKTOP-JBOMLCA\Archivos\Data_19_06_25\Data")
# file_to_open = data_folder / "n_By_mu_-349.0_L=3000_h=0.001_B_y_in_(0.0-0.48)_Delta=0.08_lambda_R=0.56_lambda_D=0_g_xx=1_g_xy=0_g_yy=1_g_yx=0_theta=1.57_points=32_beta=1000_T=True.npz"

data_folder = Path("./Data")
file_to_open = data_folder / "n_By_mu_-34.900000000000006_L=2500_h=0.001_B_y_in_(0.0-0.8)_Delta=0.08_lambda_R=0.56_lambda_D=0_g_xx=1_g_xy=0_g_yy=1_g_yx=0_theta=1.57_points=24_beta=1000_T=True_chi=0.npz"


Data = np.load(file_to_open)

n_B_y = Data["n_B_y"]
B_values = Data["B_values"]
# Lambda = Data["Lambda"]
Lambda_R = Data["Lambda_R"]
# Lambda_D = Data["Lambda_D"]
Delta = Data["Delta"]
# Delta = 0.08    # meV
theta = Data["theta"]
w_0 = Data["w_0"]
mu = Data["mu"]
L_x = Data["L_x"]
h = Data["h"]
points = Data["points"]
g_xx = Data["g_xx"]
g_yy = Data["g_yy"]
g_xy = Data["g_xy"]
g_yx = Data["g_yx"]

def interpolation_for_theory(x):
    return [
            np.interp(x, B_values/Delta, n_B_y[:, 0]),        #I have change x to x/2
            np.interp(x, B_values/Delta, n_B_y[:, 1]),
            np.interp(x, B_values/Delta, 1/2*(n_B_y[:, 0]+n_B_y[:, 1]+n_B_y[:,2]+n_B_y[:,3])),
            np.interp(x, B_values/Delta, 1/2*(n_B_y[:, 0]-n_B_y[:, 1]))
            ]  
def model_parallel(x, a, c):
    return a*(interpolation_for_theory(x)[1] - interpolation_for_theory(0)[1]) + c* x**2

def model_perpendicular(x, a, c):
    return a*(interpolation_for_theory(x)[0] - interpolation_for_theory(0)[0])  + c* x**2

def model_diagonal(x, a, c):
    return a*(interpolation_for_theory(x)[2] - interpolation_for_theory(0)[2])  + c* x**2
def model_anti_diagonal(x, a, c):
    return a*(interpolation_for_theory(x)[2] - interpolation_for_theory(0)[2])  + c* x**2

B_c = field_0[13]   #data["field 0°"][14]# 0.07  T critical field
mu_B = 5.79e-2 # meV/T
g = 2*0.08 / (mu_B*B_c)  #Delta/(mu_B*B_c )   #1 / 1.7
g_xx = g
g_yy = g

B_parallel = field_45
B_perpendicular = field_135
B_diagonal = field_90
B_anti_diagonal = field_0

x_model_parallel  = field_45/(0.7*B_c)
x_model_perpendicular  = field_135/(1.2*B_c)
x_model_diagonal  = field_90/(1.2*B_c)
x_model_anti_diagonal  = field_0/(1.2*B_c)


initial_parameters_parallel = [ 2.10435188e+06, -1.26647832e+03]
popt_parallel, pcov_parallel = curve_fit(model_parallel, x_model_parallel, n_s_45,
                                          p0=initial_parameters_parallel)

initial_parameters_perpendicular = [ 3.73583079e+06, -5.40525133e+02]
popt_perpendicular, pcov_perpendicular = curve_fit(
                                                   model_perpendicular, x_model_perpendicular, n_s_135,
                                                   p0=initial_parameters_perpendicular
                                                   )

initial_parameters_diagonal = [ 3.73583079e+06, -5.40525133e+02]
popt_diagonal, pcov_diagonal = curve_fit(
                                        model_diagonal, x_model_diagonal, n_s_90,
                                        p0=initial_parameters_diagonal
                                        )

initial_parameters_anti_diagonal = [ 3.73583079e+06, -5.40525133e+02]
popt_anti_diagonal, pcov_anti_diagonal = curve_fit(
                                        model_anti_diagonal, x_model_anti_diagonal, n_s_0,
                                        p0=initial_parameters_anti_diagonal
                                        )

standard_deviation_parallel = np.sqrt(np.diag(pcov_parallel))
standard_deviation_perpendicular = np.sqrt(np.diag(pcov_perpendicular))
standard_deviation_diagonal = np.sqrt(np.diag(pcov_diagonal))
standard_deviation_anti_diagonal = np.sqrt(np.diag(pcov_anti_diagonal))

ax.plot(B_parallel, model_parallel(x_model_parallel, *popt_parallel), "-b",  label=r"fit of $n_s(\gamma=0, \theta=0°)$", zorder=3)
ax.plot(B_perpendicular, model_perpendicular(x_model_perpendicular, *popt_perpendicular), "--k",  label=r"fit of $n_s(\gamma=0, \theta=90°)$")
ax.plot(B_diagonal, model_diagonal(x_model_diagonal, *popt_diagonal), "--g",  label=r"fit of $n_s(\gamma=\pi/4, \theta=90°)$")
ax.plot(B_anti_diagonal, model_anti_diagonal(x_model_anti_diagonal, *popt_anti_diagonal), "--r",  label=r"fit of $n_s(\gamma=\pi/4, \theta=0°)$")

ax.set_title(r"4.8 GHz Resonator "+ r"$(\gamma=45°)$"+"\n"+r"$\lambda_R=$" + f"{np.round(Lambda_R,2)}"
             +r"; $\Delta=$" + f"{Delta}"
             +r"; $\theta=$" + f"{np.round(theta,2)}"
             + r"; $\mu$"+f"={mu}"
             +r"; $w_0$"+f"={w_0}"
             +r"; $L_x=$"+f"{L_x}"
             +f"; h={h}"+"\n"
             +r"$g_{xx}=$" + f"{np.round(g_xx,2)}"
             +r"; $g_{yy}=$" + f"{np.round(g_yy,2)}"
             + r"; $T=$" + f"{np.round(1/(1000*8.617e-2), 4)}")

# ax.set_title("4.9 GHz Resonator, 45°")
ax.set_xlabel(r"$B$ [$T$]")
ax.set_ylabel(r"$n_s$")
ax.legend()
# plt.show()

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

from scipy.optimize import curve_fit

data_folder = Path(r"C:\Users\Gabriel\OneDrive - Universidad Nacional de San Martin\Doctorado-DESKTOP-JBOMLCA\Archivos\Data_19_06_25\Data")
file_to_open = data_folder / "n_By_mu_-349.0_L=3000_h=0.001_B_y_in_(0.0-0.48)_Delta=0.08_lambda_R=0.56_lambda_D=0_g_xx=1_g_xy=0_g_yy=1_g_yx=0_theta=1.57_points=32_beta=1000_T=True.npz"

# data_folder = Path("./Data")
# file_to_open = data_folder / "n_By_mu_-34.900000000000006_L=2500_h=0.001_B_y_in_(0.0-0.8)_Delta=0.08_lambda_R=0.56_lambda_D=0_g_xx=1_g_xy=0_g_yy=1_g_yx=0_theta=1.57_points=24_beta=1000_T=True_chi=0.npz"


Data = np.load(file_to_open)

n_B_y = Data["n_B_y"]
B_values = Data["B_values"]
# Lambda = Data["Lambda"]
Lambda_R = Data["Lambda_R"]
# Lambda_D = Data["Lambda_D"]
Delta = Data["Delta"]
# Delta = 0.08    # meV
theta = Data["theta"]
w_0 = Data["w_0"]
mu = Data["mu"]
L_x = Data["L_x"]
h = Data["h"]
points = Data["points"]
g_xx = Data["g_xx"]
g_yy = Data["g_yy"]
g_xy = Data["g_xy"]
g_yx = Data["g_yx"]

def interpolation_for_theory(x):
    return [
            np.interp(x, B_values/Delta, n_B_y[:, 0]),        #I have change x to x/2
            np.interp(x, B_values/Delta, n_B_y[:, 1]),
            np.interp(x, B_values/Delta, 1/2*(n_B_y[:, 0]+n_B_y[:, 1]))
            ]
def model_parallel(x, a, c):
    return a*(interpolation_for_theory(x)[1] - interpolation_for_theory(0)[1]) + c* x**2

def model_perpendicular(x, a, c):
    return a*(interpolation_for_theory(x)[0] - interpolation_for_theory(0)[0])  + c* x**2

def model_diagonal(x, a, c):
    return a*(interpolation_for_theory(x)[2] - interpolation_for_theory(0)[2])  + c* x**2

B_c = field_0[13]   #data["field 0°"][14]# 0.07  T critical field
mu_B = 5.79e-2 # meV/T
g = 2*0.08 / (mu_B*B_c)  #Delta/(mu_B*B_c )   #1 / 1.7
g_xx = g
g_yy = g

B_parallel = field_0
B_perpendicular = field_90
B_diagonal = field_45

x_model_parallel  = field_0/B_c
x_model_perpendicular  = field_90/B_c
x_model_diagonal  = field_45/B_c


initial_parameters_parallel = [ 2.10435188e+06, -1.26647832e+03]
popt_parallel, pcov_parallel = curve_fit(model_parallel, x_model_parallel, n_s_0,
                                          p0=initial_parameters_parallel)

initial_parameters_perpendicular = [ 3.73583079e+06, -5.40525133e+02]
popt_perpendicular, pcov_perpendicular = curve_fit(
                                                   model_perpendicular, x_model_perpendicular, n_s_90,
                                                   p0=initial_parameters_perpendicular
                                                   )

initial_parameters_diagonal = [ 3.73583079e+06, -5.40525133e+02]
popt_diagonal, pcov_diagonal = curve_fit(
                                        model_diagonal, x_model_diagonal, n_s_45,
                                        p0=initial_parameters_diagonal
                                        )

standard_deviation_parallel = np.sqrt(np.diag(pcov_parallel))
standard_deviation_perpendicular = np.sqrt(np.diag(pcov_perpendicular))

ax.plot(B_parallel, model_parallel(x_model_parallel, *popt_parallel), "-b",  label=r"fit of $n_s(\gamma=0, \theta=0°)$", zorder=3)
ax.plot(B_perpendicular, model_perpendicular(x_model_perpendicular, *popt_perpendicular), "--k",  label=r"fit of $n_s(\gamma=0, \theta=90°)$")
ax.plot(B_diagonal, model_diagonal(x_model_diagonal, *popt_diagonal), "--g",  label=r"fit of $n_s(\gamma=\pi/4, \theta=90°)$")

ax.set_title(r"5.7 GHz Resonator "+ r"$(\gamma=0)$"+"\n"+r"$\lambda_R=$" + f"{np.round(Lambda_R,2)}"
             +r"; $\Delta=$" + f"{Delta}"
             +r"; $\theta=$" + f"{np.round(theta,2)}"
             + r"; $\mu$"+f"={mu}"
             +r"; $w_0$"+f"={w_0}"
             +r"; $L_x=$"+f"{L_x}"
             +f"; h={h}"+"\n"
             +r"$g_{xx}=$" + f"{np.round(g_xx,
             2)}"
             +r"; $g_{yy}=$" + f"{np.round(g_yy,2)}"
             + r"; $T=$" + f"{np.round(1/(1000*8.617e-2), 4)}")

# ax.set_title("5.7 GHz Resonator, 0°")
ax.set_xlabel(r"$B$ [$T$]")
ax.set_ylabel(r"$n_s$")
ax.legend()
plt.tight_layout()
plt.show()

#%% Angle dependence 4.9 GHz

data_folder = Path(r"Files/data gabriel")

file_path = data_folder / 'angle_dep_4_9_GHz_45deg.xlsx'

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

fig, ax1 = plt.subplots() 
# ax1 = plt.subplot(121)
# ax2 = plt.subplot(122, projection='polar')

ax1.errorbar(angle_25_mT, delta_ns_25, yerr=delta_ns_err_25, label=r"$n_s(25mT)$", fmt="-o")
ax1.errorbar(angle_50_mT, delta_ns_50, yerr=delta_ns_err_50, label=r"$n_s(50mT)$", fmt="-o")
ax1.errorbar(angle_75_mT, delta_ns_75, yerr=delta_ns_err_75, label=r"$n_s(75mT)$", fmt="-o")
ax1.errorbar(angle_100_mT, delta_ns_100, yerr=delta_ns_err_100, label=r"$n_s(100mT)$", fmt="-o")

# ax2.errorbar(angle_25_mT, delta_ns_25, yerr=delta_ns_err_25, label=r"$n_s(25mT)$", fmt="o")
# ax2.errorbar(angle_50_mT, delta_ns_50, yerr=delta_ns_err_50, label=r"$n_s(50mT)$", fmt="o")
# ax2.errorbar(angle_75_mT, delta_ns_75, yerr=delta_ns_err_75, label=r"$n_s(75mT)$", fmt="o")
# ax2.errorbar(angle_100_mT, delta_ns_100, yerr=delta_ns_err_100, label=r"$n_s(100mT)$", fmt="o")



ax1.set_title("4.9 GHz Resonator, 45°")
ax1.set_xlabel(r"Angle")
ax1.set_ylabel(r"$n_s$")
ax.legend()
plt.tight_layout()
plt.show()

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

#%% Angle dependence 9.7 GHz

data_folder = Path(r"Files/data gabriel")

file_path = data_folder / 'angle_dep_9_7_GHz_90deg.xlsx'

xls = pd.ExcelFile(file_path) # use r before absolute file path 
sheetX = xls.parse(0) #2 is the sheet number+1 thus if the file has only 1 sheet write 0 in paranthesis

angle_25_mT = sheetX['angle 25 mT']  
delta_ns_25 = sheetX["delta ns 25"]
delta_ns_err_25 = sheetX["delta ns err 25"]

angle_50_mT = sheetX['angle 50 mT']  
delta_ns_50 = sheetX["delta ns 50"]
delta_ns_err_50 = sheetX["delta ns err 50"]

# angle_75_mT = sheetX['angle 75 mT']  
# delta_ns_75 = sheetX["delta ns 75"]
# delta_ns_err_75 = sheetX["delta ns err 75"]

# angle_100_mT = sheetX['angle 100 mT']  
# delta_ns_100 = sheetX["delta ns 100"]
# delta_ns_err_100 = sheetX["delta ns err 100"]

fig, ax = plt.subplots()
ax.errorbar(angle_25_mT, delta_ns_25, yerr=delta_ns_err_25, label=r"$n_s(25mT)$", fmt="-o")
ax.errorbar(angle_50_mT, delta_ns_50, yerr=delta_ns_err_50, label=r"$n_s(50mT)$", fmt="-o")
# ax.errorbar(angle_75_mT, delta_ns_75, yerr=delta_ns_err_75, label=r"$n_s(75mT)$", fmt="-o")
# ax.errorbar(angle_100_mT, delta_ns_100, yerr=delta_ns_err_100, label=r"$n_s(100mT)$", fmt="-o")



ax.set_title("9.7 GHz Resonator, 90°")
ax.set_xlabel(r"Angle")
ax.set_ylabel(r"$n_s$")
ax.legend()
plt.show()