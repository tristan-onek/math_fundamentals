# Stochastic analysis practice - GBM
# Imports:
import numpy as np
import matplotlib.pyplot as plt
# End imports

# Parameters:
S0 = 100        # Starting stock price
mu = 0.05       # Drift or 'mean return'
sigma = 0.2     # Volatility
T = 1           # Time horizon in years
N = 252         # Number of time steps (e.g. trading days per year)
dt = T / N      # Time step size
simulations = 10  # The number of simulated paths

# Generate GBM paths:
time = np.linspace(0, T, N)
paths = np.zeros((simulations, N))
for i in range(simulations):
    dW = np.random.normal(0, np.sqrt(dt), N - 1) # generate random Brownian increments
    W = np.insert(np.cumsum(dW), 0, 0)  # Wiener process (starts at 0)
    
    # Simulate GBM path
    path = S0 * np.exp((mu - 0.5 * sigma**2) * time + sigma * W)
    paths[i, :] = path

# Plot the simulated paths
plt.figure(figsize=(10, 6))
for i in range(simulations):
    plt.plot(time, paths[i], lw=1)
plt.title("Simulating Geometric Brownian Motion Paths")
plt.xlabel("Time (Years)")
plt.ylabel("Stock Price")
plt.grid(True)
plt.show()
