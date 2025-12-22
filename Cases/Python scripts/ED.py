import numpy as np
from scipy.integrate import solve_ivp
from scipy.optimize import root_scalar
import matplotlib.pyplot as plt

# Parámetros

tend = 1.9e8 #1.454e8 #1.555e8
U = 0.025 # 0.1 # 0.05

M = 0.06354 
F = 96485
n = 2
rho = 8960

g = 0.01 
D = 5e-10
bc = -0.05133
ba = 0.01711
j0 = 10
c0 = 100

kappas = 5
kappaf = 10

cte = M/n/F/rho       # Constante multiplicativa

def cs_given_j(h, j):
    return  max (1e-13, 1 - (g-h)/n/F/D*j/c0)
    
# Definir la ecuación implícita para j: j = exp(-alpha * h) + beta * j^2
def implicit_j_eq(j, h):
    #cs = max (1e-13, 1 - (g-h)/n/F/D*j/c0)
    cs = cs_given_j(h, j)
    Rsf = h/kappas + (g-h)/kappaf
    deltafi = -(U - j * Rsf)
    return j - j0 * ( cs * np.exp(deltafi / bc)  - np.exp(deltafi / ba) )
    
# Encuentra j dado h resolviendo la ecuación implícita
def solve_for_j(h):
    sol = root_scalar(implicit_j_eq, args=(h,), method='brentq', bracket=[1e-6, 1000])
    if sol.converged:
        return sol.root
    else:
        raise ValueError("No se pudo resolver j implícito para h = {}".format(h))

# Ecuación diferencial dh/dt = cte * j(h) con j implícito
def dh_dt(t, h):
    j_val = solve_for_j(h)
    return cte * j_val

# Condición inicial
h0 = [0.0]

# Tiempo
t_span = (0, tend) 
t_eval = np.linspace(*t_span, 100)

# Resolver ODE
sol = solve_ivp(dh_dt, t_span, h0, t_eval=t_eval)

# Calcular j(t) y cs(t)
j_vals = []
cs_vals = []

for h in sol.y[0]:
    j = solve_for_j(h)
    #cs = 1 - (g - h) / n / F / D * j / c0
    cs = cs_given_j(h, j)
    j_vals.append(j)
    cs_vals.append(cs)

j_vals = np.array(j_vals)
cs_vals = np.array(cs_vals)

# Guardar resultados en archivo de texto
output_data = np.column_stack((sol.t, sol.y[0], j_vals, cs_vals))
np.savetxt('output.txt', output_data, header='t h j cs', comments='')

# Graficar h(t)
plt.figure()
plt.plot(sol.t, sol.y[0]/g, label='H(t)')
plt.xlabel('Tiempo t')
plt.ylabel('h(t)')
plt.title('Evolución de h(t)')
plt.grid(True)
plt.legend()

# Graficar j(t)
plt.figure()
plt.plot(sol.t, j_vals, label='j(t)', color='orange')
plt.xlabel('Tiempo t')
plt.ylabel('j(t)')
plt.title('Evolución de j(t)')
plt.grid(True)
plt.legend()

# Graficar cs(t)
plt.figure()
plt.plot(sol.t, cs_vals, label='cs(t)', color='green')
plt.xlabel('Tiempo t')
plt.ylabel('cs(t)')
plt.title('Evolución de cs(t)')
plt.grid(True)
plt.legend()

plt.show()











