import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from scipy import optimize
import pint
import math

u = pint.UnitRegistry()

# Constants

n_crew = 4
n_passengers = 50
n_people = n_passengers + n_crew
weight_per_passenger = 100 * u.kg

Swet_Sref = 6
AR = 8
K_ld = 15.5

Rcr = 2000 * u.km
Eltr = 45 * u.minutes
Rdiv = 370 * u.km

# Speed of sound = 589 knots at 35,000 feet
Vcr = 589 * 0.75 * u.knots

# Roskam, page 57 for inefficiencies of c_j

# cj_cr = 0.5 * u.lb / u.lb / u.hr
# cj_ltr = 0.6 * u.lb / u.lb / u.hr
# cj_diversion = 0.9 * u.lb / u.lb / u.hr

g = 9.81 * u.m / (u.s ** 2)
cj_cr = (19.8 * u.milligram / u.newton / u.s) * g
# TODO: Look into better methods for this
cj_ltr = 1.2 * cj_cr
cj_diversion = 1.8 * cj_cr

# Calculate fuel fractions

f_W_to = 0.99 * 0.99 * 0.995
f_W_climb = 0.98
f_W_descent = 0.99
f_W_shutdown = 0.992

# Range equation fractions (x2)

LDmax = 20.67

# Actual calculations now

W_PL = n_people * weight_per_passenger

# Weight fractions...

# Cruise
f_W_cr = 1/math.exp(Rcr / ((Vcr/cj_cr) * (LDmax * 0.867)))

# Loiter
f_W_ltr = 1/math.exp(Eltr / ((1/cj_ltr) * (LDmax)))

# Diversion cruise
f_W_div = 1/math.exp(Rdiv / ((275 * u.knot/cj_diversion) * (LDmax)))

# No penalty for diversion climb/land - only to about 10,000 ft
W = np.array([1, f_W_to, f_W_climb, f_W_cr, f_W_descent, 1, f_W_div, 1, f_W_ltr, f_W_shutdown])

# Cumulative
W_cumulative = [np.product(W[0:x]) for x in range(1, len(W)+1)]

# 1% trapped fuel assumption (Errikos)
M_ff = 1 - np.product(W)
M_ff = (1.01 * (M_ff))

# Weight estimation now, with iteration

# Guess
M_ff = 0.204

def guess_empty_weight(W_0):
	W_PL_lb = 11904.96
	W_0E = W_0 - M_ff * W_0 - W_PL_lb
	W_E = W_0E - 880
	return W_E

def empty_weight_divergence(W_0):
	# Minimum allowable weight equation from Roskam, page 18
	A = 0.0833
	B = 1.0383
	WcompWmetal = 0.98#1#0.80

	W_E_min = 10 ** ((np.log10(W_0) - A) / B) * WcompWmetal

	W_E_guess = guess_empty_weight(W_0)

	return W_E_guess - W_E_min 

print("Optimizing...")
root = optimize.newton(empty_weight_divergence, 130000)
print(f"MTOW {root/2.20462} kg")