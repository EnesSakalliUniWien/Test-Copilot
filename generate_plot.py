import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression

# Generate random sample data
np.random.seed(42)  # for reproducibility
X = 2 * np.random.rand(200, 1)
y = 4 + 3 * X + np.random.randn(200, 1)

# Create and fit the linear regression model
model = LinearRegression()
model.fit(X, y)

# Generate points for the regression line
X_test = np.array([[0], [2]])
y_pred = model.predict(X_test)


# Create the plot
plt.figure(figsize=(10, 6))
plt.scatter(X, y, color="blue", alpha=0.5, label="Data points")
plt.plot(X_test, y_pred, color="red", label="Linear regression")
plt.xlabel("X")
plt.ylabel("y")
plt.title("Linear Regression Example")
plt.legend()
plt.grid(True, alpha=0.3)

# Save the plot
plt.savefig("linear_regression_plot.png")
plt.close()
