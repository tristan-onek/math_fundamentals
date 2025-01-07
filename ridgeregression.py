# imports:
import matplotlib.pyplot as plt
from sklearn.linear_model import Ridge
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
import numpy as np
# end imports

# Generate data
np.random.seed(42)
X = 2 * np.random.rand(100, 1)  # 100 samples and 1 feature
true_coeff = 3.5
noise = np.random.randn(100, 1)  # Adding noise
y = true_coeff * X + 5 + noise
# End data generation

# Split data into training/testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Ridge Regression with different alpha values
alphas = [0.01, 0.1, 1, 10, 100]
mse_train = []
mse_test = []

plt.figure(figsize=(12, 6))
for alpha in alphas:
    ridge_reg = Ridge(alpha=alpha)
    ridge_reg.fit(X_train, y_train)

    # Predictions
    y_train_pred = ridge_reg.predict(X_train)
    y_test_pred = ridge_reg.predict(X_test)

    # Calculate MSE
    mse_train.append(mean_squared_error(y_train, y_train_pred))
    mse_test.append(mean_squared_error(y_test, y_test_pred))

    # Plot the predictions
    plt.scatter(X_test, y_test, label=f'Test Data (alpha={alpha})')
    plt.plot(X_test, y_test_pred, label=f'Ridge Fit (alpha={alpha})', linewidth=2)

plt.title("Ridge Regression - Different Alpha Values...")
plt.xlabel("Feature (X)")
plt.ylabel("Target (y)")
plt.legend()
plt.grid()
plt.show()

# Print the MSE values
for i, alpha in enumerate(alphas):
    print(f"Alpha: {alpha}, Train MSE: {mse_train[i]:.4f}, Test MSE: {mse_test[i]:.4f}")
