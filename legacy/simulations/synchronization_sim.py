import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import odeint

"""
NOETIKO - Phase Synchronization Simulation
Model: Kuramoto-Sakaguchi with 1.42 GHz Carrier
Context: Validation of Part 2 (Noetiko Convergence)
"""

def kuramoto_sakaguchi(theta, t, omega, K, alpha, N):
    """
    Berechnet die zeitliche Ableitung der Phasen für N gekoppelte Oszillatoren.
    alpha: Phasen-Shift (Sakaguchi-Parameter), beschreibt non-lokale Verzögerung.
    K: Kopplungsstärke (induziert durch das A-Feld).
    """
    dtheta = np.zeros(N)
    for i in range(N):
        # Grundfrequenz + Summe der Kopplungen zu allen anderen Oszillatoren
        coupling = np.sum(np.sin(theta - theta[i] - alpha))
        dtheta[i] = omega[i] + (K / N) * coupling
    return dtheta

# --- Parameter-Setup ---
N = 50                 # Anzahl der biologischen Oszillatoren (z.B. Zell-Cluster)
T_max = 0.05           # Simulationszeit
dt = 0.0001
t = np.arange(0, T_max, dt)

# Referenzfrequenz: 1.42 GHz (Wasserstoff-Linie)
# Für die Simulation skalieren wir diese auf eine darstellbare Kreisfrequenz omega
f_target = 1.42e9 
omega_mean = 100       # Normierte Trägerfrequenz für die Visualisierung
omega = np.random.normal(omega_mean, 1.0, N)  # Natürliche Varianz der Oszillatoren

# NOETIKO Parameter
K = 15.0               # Kopplungsstärke (aktiviert durch Artifact_01)
alpha = 0.1            # Phasen-Shift (Sakaguchi-Parameter)

# Initialzustand: Volles Chaos (zufällige Phasen zwischen 0 und 2*pi)
theta0 = np.random.uniform(0, 2*np.pi, N)

# --- Simulation ---
print(f"Starte NOETIKO Konvergenz-Simulation (N={N})...")
sol = odeint(kuramoto_sakaguchi, theta0, t, args=(omega, K, alpha, N))

# Berechnung des Ordnungsparameters R (Kohärenz-Grad)
# R = 1.0 bedeutet perfekte Synchronisation
r_t = np.abs(np.mean(np.exp(1j * sol), axis=1))

# --- Visualisierung ---
plt.figure(figsize=(12, 6))

# Subplot 1: Phasen-Evolution
plt.subplot(1, 2, 1)
for i in range(N):
    plt.plot(t, np.sin(sol[:, i]), alpha=0.3)
plt.title("Phasen-Konvergenz (Kuramoto-Sakaguchi)")
plt.xlabel("Zeit [s]")
plt.ylabel("sin(theta) - Oszillation")
plt.grid(True, alpha=0.3)

# Subplot 2: Ordnungsparameter (Kohärenz-Anstieg)
plt.subplot(1, 2, 2)
plt.plot(t, r_t, color='gold', linewidth=2, label="Ordnungsparameter R")
plt.axhline(y=1.0, color='red', linestyle='--', alpha=0.5)
plt.title("NOETIKO Kohärenz-Gradient")
plt.xlabel("Zeit [s]")
plt.ylabel("Grad der Synchronisation (R)")
plt.ylim(0, 1.1)
plt.legend()
plt.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig("noetiko_convergence_plot.png")
print("Simulation abgeschlossen. Plot gespeichert als 'noetiko_convergence_plot.png'.")
plt.show()
