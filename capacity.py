import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.optimize import curve_fit
import os
from tkinter import filedialog as fd

files = []
names = []
data = []

files = fd.askopenfilenames(initialdir=os.getcwd() + "/Messungen", title="text", filetypes= (("CSV files","*.csv"), ("all files","*.*")))
if (len(files) == 0): exit()

for i in range (0,len(files)):
    names.append (files[i].replace(os.getcwd().replace("\\", "/") , ". "))

curvetype = "rising"
# Angegebener Wert der verwendeten Widerstände
R_ges = 910 #[Ohm]
R_tolerance = 1 #[%]

R_abw = R_ges * R_tolerance / 100

# Muss bei wechsel der Schaltung neu gemessen weren (ohne Kondensator)
offset = 7.3883e-11 # [F]

Tau_ges = []
Tau_ges_err = []

#finde ein Intervall, das nur eine Ladung/Entladung enthält
def findIntervall(i):
    dataLength = len(data[i])

    s = 0
    e = int(0.8 * dataLength)

    if (curvetype == "rising"):
        for j in range (s, e):
            if data[i][j,1] > 1:
                return [j, j + int(dataLength / 1.5)]

    elif (curvetype == "falling"):
        for j in range (s, e):
            if data[i][j,1] > 1:
                return [j, j + int(dataLength / 1.5)]

    else:
        print ("ERROR: unexpected curve type " + curvetype)
        exit()

#exponentialfunktion
def exp(x):
    return np.e**x

#fitfunktion für steigenden Spannungsverlauf
def f_rising(t, U0, tau, t0, C):
    return U0 * (1 - exp(-(t-t0)/tau)) + C

#fitfunktion für fallenden Spannungsverlauf
def f_falling(t, U0, tau):
    return U0 * exp(-1 * t / tau)

#importiere messdaten in data.
for i in range (0, len(files)):
    data.append (pd.read_csv(files[i], sep=",", skiprows=12 ,decimal="."))
    data[i] = np.array(data[i])

# Vollständige diagramme (debugging)
# for i in range (0, len(files)):
#     (s, e) = findIntervall(i)

#     plt.title(names[i])
#     plt.xlabel("t in s")
#     plt.ylabel("U in V")

#     plt.plot(data[i][:,0], data[i][:,1], label="Measurements")
#     plt.axvline(data[i][s,0], linestyle="dotted", color="black", label="Fit Intervall")
#     plt.axvline(data[i][e,0], linestyle="dotted", color="black")
#     plt.legend()

#     plt.show()
#     plt.clf()

#Teildiagramme mit Fitkurve
for i in range (0, len(files)):

    start, end = findIntervall(i)

    rawData = [data[i][start:end,0], data[i][start:end,1]]

    # Startparameter für Fit
    params = [max(rawData[1]), rawData[0][len(rawData[0]) -1] - rawData[0][0], 0, 0]

    covariance_matrix = np.array([])

    if (curvetype == "rising"):
        params, covariance_matrix = curve_fit(f_rising, rawData[0], rawData[1], p0=params, bounds=([0,0,-30,-1e-2], [30,120,30,1e-2]))
        plt.plot(rawData[0], f_rising(rawData[0], params[0], params[1], params[2], params[3]), color="orange", label="Fit Curve", linewidth=2)

    elif (curvetype == "falling"):
        params, covariance_matrix = curve_fit(f_falling, rawData[0], rawData[1], p0=[10, 1e-3])
        plt.plot(rawData[0], f_falling(rawData[0], params[0], params[1]), color="orange", label="Fit Curve", linewidth=2)

    tau = params[1]
    tau_err = np.sqrt(covariance_matrix[1][1])

    Tau_ges.append(tau)
    Tau_ges_err.append(tau_err)

    C = tau / R_ges
    C_err = tau_err / R_ges

    print("\nMessung " + files[i] + ": Tau = " + np.format_float_scientific(tau, 4) + " +- " + np.format_float_scientific(tau_err, 4) + " s")
    print("Messung " + files[i] + ":   C = " + np.format_float_scientific(C, 4) + " +- " + np.format_float_scientific(C_err, 4) + " F\n")

    textstr = ""
    textstr += r"$U_0$ = " + '%.2f' % params[0] + " V\n"
    textstr += r"$\tau$ = " + np.format_float_scientific(tau, 2) + " s\n"
    textstr += r"$\sigma_{\tau} =$ " + np.format_float_scientific(tau_err, 2) + " s\n"
    textstr += r"$C =$ " + np.format_float_scientific(C, 2) + " F\n"
    textstr += r"$\Delta C =$ " + np.format_float_scientific(C_err, 2) + " F"

    props = dict(boxstyle='round', facecolor='wheat', alpha=0.5)

    textpos = (data[i][start] + data[i][end]) / 2
    textX = textpos[0]
    textY = textpos[1]

    plt.text(textX, textY, textstr, fontsize=14,
    verticalalignment='top', bbox=props)

    plt.title(names[i])
    plt.xlabel("t in s")
    plt.ylabel("U in V")

    plt.scatter(rawData[0], rawData[1] , label="Measurements", s=8)
    plt.legend()

    plt.savefig("out/C_" + str(i) + ".png", format="png")
    plt.savefig("out/C_" + str(i) + ".pdf", format="pdf")

    plt.clf()

C = np.array(Tau_ges) / R_ges
C_err = np.array(Tau_ges_err) / R_ges
dR = (np.mean(Tau_ges)/(R_ges)**2)
C_err_R_err = np.linalg.norm([np.std(Tau_ges) / R_ges, dR * R_abw])

print("Gesamt:                              C = " + np.format_float_scientific(np.mean(C), 4) + " +- " + np.format_float_scientific(C_err_R_err, 4) + " F (" + str(np.round(C_err_R_err / np.mean(C) * 100, 2)) + " %)")
print("Korrigiert mit Eigenkapazität:       C = " + np.format_float_scientific(np.mean(C) - offset, 4) + " +- " + np.format_float_scientific(C_err_R_err, 4) + " F (" + str(np.round(C_err_R_err / (np.mean(C) - offset) * 100, 2)) + " %)")
print("Statistischer Fehler (Std. Abw.):        " + np.format_float_scientific(np.std(C), 4) + " F")
print("Größter individueller Fit-Fehler:        " + np.format_float_scientific(np.max(C_err), 4) + " F")
print("Fehler durch Widerstand-Toleranz:        " + np.format_float_scientific(np.max(dR * R_abw), 4) + " F\n")