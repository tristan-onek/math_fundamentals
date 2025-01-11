set.seed(123) # For reproducibility
x <- 1:10
y <- 2 * x + rnorm(10, mean = 0, sd = 1) # Linear relationship with some noise

# Create a data frame
data <- data.frame(x = x, y = y)

# Perform linear regression
model <- lm(y ~ x, data = data)

# Print model summary
summary(model)

# Plot the data and the regression line
plot(data$x, data$y, main = "Linear Regression Example", xlab = "x", ylab = "y", pch = 16, col = "blue")
abline(model, col = "red", lwd = 2)
legend("topleft", legend = "Regression Line", col = "red", lwd = 2)
