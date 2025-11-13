import numpy as np
import pandas as pd
from math import pi
from scipy.optimize import differential_evolution, minimize
import matplotlib.pyplot as plt

# Load data
df = pd.read_csv("xy_data.csv")
x_obs = df["x"].values
y_obs = df["y"].values

# Generate uniform t values
t = np.linspace(6, 60, len(df))

# Parametric model
def model_xy(t, theta_deg, M, X):
    theta = theta_deg * pi / 180.0
    exp_term = np.exp(M * np.abs(t))
    sin03 = np.sin(0.3 * t)
    x = t * np.cos(theta) - exp_term * sin03 * np.sin(theta) + X
    y = 42 + t * np.sin(theta) + exp_term * sin03 * np.cos(theta)
    return x, y

# L1 loss function
def l1_loss(params):
    theta_deg, M, X = params
    if not (0 < theta_deg < 50 and -0.05 < M < 0.05 and 0 < X < 100):
        return 1e9
    x_pred, y_pred = model_xy(t, theta_deg, M, X)
    return np.sum(np.abs(x_obs - x_pred) + np.abs(y_obs - y_pred))

# Global optimization
bounds = [(0.0, 50.0), (-0.05, 0.05), (0.0, 100.0)]
result_de = differential_evolution(l1_loss, bounds, maxiter=200, popsize=20, tol=1e-6, polish=True, seed=42)

# Local refinement
x0 = result_de.x
result_local = minimize(l1_loss, x0, method="Powell", options={"xtol":1e-8, "ftol":1e-8, "maxiter":10000})

# Final parameters
theta_est, M_est, X_est = result_local.x
loss = result_local.fun

# Output results
print("\nEstimated Parameters:")
print(f"theta (degrees): {theta_est:.6f}")
print(f"M: {M_est:.6f}")
print(f"X: {X_est:.6f}")
print(f"Total L1 loss: {loss:.6f}")

# Generate fitted curve
x_fit, y_fit = model_xy(t, theta_est, M_est, X_est)

# Save curve plot
plt.figure(figsize=(8,6))
plt.scatter(x_obs, y_obs, label="Observed", s=10)
plt.plot(x_fit, y_fit, color="red", label="Fitted Curve")
plt.legend()
plt.xlabel("x")
plt.ylabel("y")
plt.title("Observed vs Fitted Parametric Curve")
plt.savefig("curve_fit.png")
plt.show()

# Residual analysis
res = np.abs(x_obs - x_fit) + np.abs(y_obs - y_fit)
print(f"Median L1 residual: {np.median(res):.6f}")
print(f"95th percentile residual: {np.percentile(res,95):.6f}")
