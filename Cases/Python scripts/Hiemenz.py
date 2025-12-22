import numpy as np
from scipy.special import airy
from scipy.integrate import solve_ivp
from scipy.optimize import root_scalar
import matplotlib.pyplot as plt
import warnings

# Evitar warnings de versiones de NumPy/Scipy
warnings.filterwarnings("ignore", category=UserWarning)

# Parámetros
tend =  7.5e6 #5e6 # Tiempo final
U = 0.1 #0.2 #
uin = 0.5e-4 #2e-4 # Rein0 = 8

nu = 1e-6
M = 0.06354
F = 96485
n = 2
rho =  8960 
g = 0.01
D = 5e-10
bc = -0.05133
ba = 0.01711
j0 = 10
c0 = 100

kappas = 5
kappaf = 10

Sc = nu / D
cte = M / n / F / rho  # Constante multiplicativa

def cs_given_j(h, j):
    dh = 2 * (g-h)
    Rein = uin * dh / nu 
    tita2 = 24 + 32/35 * Rein
    Sh = 0.616 * (Rein * Sc * tita2) ** (1.0/3.0)
    km = Sh * D / dh
    csurf = c0 - j / n / F / km
    return max(1e-13, csurf / c0)
    
def Rsf_(h):
    return h / kappas + (g - h) / kappaf

def deltafi_(j, Rsf):
    return -(U - j * Rsf)     
    

def implicit_j_eq(j, h):
    cs = cs_given_j(h, j)
    Rsf = Rsf_(h)
    deltafi = deltafi_(j, Rsf)
    return j - j0 * (cs * np.exp(deltafi / bc) - np.exp(deltafi / ba))

def find_bracket_for_root(func, h, j_min=1e-6, j_max=1e3, steps=100):
    js = np.logspace(np.log10(j_min), np.log10(j_max), steps)
    fvals = [func(j, h) for j in js]

    for i in range(len(js) - 1):
        if fvals[i] * fvals[i + 1] < 0:
            return js[i], js[i + 1]
    return None

def compute_j_and_deltafi(h):
    if h <= 1e-18:
        h = 1e-18
    bracket = find_bracket_for_root(implicit_j_eq, h)
    if bracket is None:
        j = 1e-18
    else:
        sol = root_scalar(implicit_j_eq, args=(h,), method='brentq', bracket=bracket)
        if sol.converged:
            j = sol.root
        else:
            raise ValueError(f"No se pudo resolver j para h = {h}")
    cs = cs_given_j(h, j)
    Rsf = Rsf_(h)
    deltafi = deltafi_(j, Rsf)
    return j, deltafi

def dh_dt(t, h):
    h_scalar = h[0] if isinstance(h, (list, np.ndarray)) else h
    j_val, _ = compute_j_and_deltafi(h_scalar)
    return [cte * j_val]

# Condición inicial
h0 = [1e-18]  # Evitar h=0 exacto

# Tiempo de integración
t_span = (0, tend)
t_eval = np.linspace(*t_span, 100)

# Resolver ODE
sol = solve_ivp(dh_dt, t_span, h0, t_eval=t_eval, method='RK45')

# Post-procesar para j, deltafi y cs
j_vals = []
deltafi_vals = []
cs_vals = []

for h in sol.y[0]:
    j, deltafi = compute_j_and_deltafi(h)
    cs = cs_given_j(h, j)
    j_vals.append(j)
    deltafi_vals.append(deltafi)
    cs_vals.append(cs)

j_vals = np.array(j_vals)
deltafi_vals = np.array(deltafi_vals)
cs_vals = np.array(cs_vals)

# Guardar resultados
output_data = np.column_stack((sol.t, sol.y[0], j_vals, deltafi_vals, cs_vals))
np.savetxt('output.txt', output_data, header='t h j deltafi cs', comments='')

# Graficar h(t)
plt.figure()
plt.plot(sol.t, sol.y[0] / g, label='h(t)/g')
plt.xlabel('Tiempo t')
plt.ylabel('h(t)/g')
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

# Graficar deltafi(t)
plt.figure()
plt.plot(sol.t, deltafi_vals, label='deltafi(t)', color='red')
plt.xlabel('Tiempo t')
plt.ylabel('deltafi(t)')
plt.title('Evolución de deltafi(t)')
plt.grid(True)
plt.legend()

plt.show()

