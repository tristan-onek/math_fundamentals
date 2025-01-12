library(ggplot2)

# Simulate data
set.seed(123)  # For reproducibility
n <- 200  # Number of observations
age <- round(rnorm(n, mean = 35, sd = 10))  # Generate random ages
income <- round(rnorm(n, mean = 50, sd = 15))  # Generate random incomes
buy <- rbinom(n, 1, prob = plogis(-4 + 0.1 * age + 0.05 * income))  # Purchase outcome

# Combine into a data frame
data <- data.frame(age, income, buy)

# View first few rows
head(data)

#fitting logistic regression model:
model <- glm(buy ~ age + income, family = binomial, data = data)

#Summary of the model:
summary(model)

#Predicted probabilities:
data$predicted_prob <- predict(model, type = "response")

# Example prediction:
new_customer <- data.frame(age = 40, income = 60)
predicted_prob_new <- predict(model, newdata = new_customer, type = "response")

cat("Predicted probability of purchase for the new customer: ", predicted_prob_new, "\n")

# Viz of predicted probabilities vs income - age is fixed
fixed_age <- 35
new_data <- data.frame(
  age = fixed_age,
  income = seq(min(data$income), max(data$income), length.out = 100)
)
new_data$predicted_prob <- predict(model, newdata = new_data, type = "response")

# Plot
ggplot(data, aes(x = income, y = buy)) +
  geom_point(alpha = 0.6, color = "blue") +
  geom_line(data = new_data, aes(x = income, y = predicted_prob), color = "red", size = 1) +
  labs(
    title = "Logistic Regression: Predicted Probability vs Income",
    x = "Income",
    y = "Purchase Probability"
  ) +
  theme_minimal()
