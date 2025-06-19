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

n_s_0 = sheetX['n_s 0°']
field_0 = sheetX["fields 0°"]
n_s_0_error = sheetX["n_s 0° err"]

n_s_45 = sheetX['n_s 45°']
field_45 = sheetX["fields 45°"]
n_s_45_error = sheetX["n_s 45° err"]

n_s_90 = sheetX['n_s 90°']
field_90 = sheetX["fields 90°"]
n_s_90_error = sheetX["n_s 90° err"]

n_s_135 = sheetX['n_s 135°']
field_135 = sheetX["fields 135°"]
n_s_135_error = sheetX["n_s 135° err"]

fig, ax = plt.subplots()
ax.errorbar(field_0, n_s_0, yerr=n_s_0_error, label=r"$n_s(0°)$", color="red", fmt="o")
ax.errorbar(field_45, n_s_45, yerr=n_s_45_error, label=r"$n_s(45°)$", color="green", fmt="o")
ax.errorbar(field_90, n_s_90, yerr=n_s_90_error, label=r"$n_s(90°)$", color="black", fmt="o")
ax.errorbar(field_135, n_s_135, yerr=n_s_135_error, label=r"$n_s(135°)$", color="darkviolet", fmt="o")


ax.set_title("4.9 GHz Resonator, 45°")
ax.set_xlabel(r"$B$ [$T$]")
ax.set_ylabel(r"$n_s$")
ax.legend()
# plt.show()

#%% 5.7 GHz resonator

data_folder = Path(r"Files/data gabriel")

file_path = data_folder / 'field_dep_5_7_GHz_0deg.xlsx'

xls = pd.ExcelFile(file_path) # use r before absolute file path 
sheetX = xls.parse(0) #2 is the sheet number+1 thus if the file has only 1 sheet write 0 in paranthesis

n_s_0 = sheetX['n_s 0°']
field_0 = sheetX["fields 0°"]
n_s_0_error = sheetX["n_s 0° err"]

n_s_45 = sheetX['n_s 45°']
field_45 = sheetX["fields 45°"]
n_s_45_error = sheetX["n_s 45° err"]

n_s_90 = sheetX['n_s 90°']
field_90 = sheetX["fields 90°"]
n_s_90_error = sheetX["n_s 90° err"]

n_s_135 = sheetX['n_s 135°']
field_135 = sheetX["fields 135°"]
n_s_135_error = sheetX["n_s 135° err"]

# fig, ax = plt.subplots()
ax.errorbar(field_0, n_s_0, yerr=n_s_0_error, label=r"$n_s(0°)$", color="red", fmt="*")
ax.errorbar(field_45, n_s_45, yerr=n_s_45_error, label=r"$n_s(45°)$", color="green", fmt="*")
ax.errorbar(field_90, n_s_90, yerr=n_s_90_error, label=r"$n_s(90°)$", color="black", fmt="*")
ax.errorbar(field_135, n_s_135, yerr=n_s_135_error, label=r"$n_s(135°)$", color="darkviolet", fmt="*")


ax.set_title("5.7 GHz Resonator, 0°")
ax.set_xlabel(r"$B$ [$T$]")
ax.set_ylabel(r"$n_s$")
ax.legend()
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