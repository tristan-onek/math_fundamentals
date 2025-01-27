# Overview of basic probability and statistics content.

import numpy as np
import scipy.stats as stats
import matplotlib.pyplot as plt

# 1. Random Sampling:
print("# 1. A Random Sampling")
np.random.seed(42)  # For reproducibility
sample = np.random.choice(range(1, 101), size=10, replace=False)
print("Random sample of 10 numbers from 1 to 100 (without replacement):", sample)

# 2. Mean, Median, and Standard Deviation:
print("\n# 2. Mean, Median, and Standard Deviation")
data = [12, 15, 14, 10, 18, 20, 12, 17]
mean = np.mean(data)
median = np.median(data)
std_dev = np.std(data)
print(f"Data: {data}\nMean: {mean}, Median: {median}, Standard Deviation: {std_dev}")

# 3. Normal Distribution
print("\n# 3. Normal Distribution")
mu, sigma = 0, 1  # Mean and standard deviation
x = np.linspace(-4, 4, 100)
y = stats.norm.pdf(x, mu, sigma)
plt.figure(figsize=(8, 4))
plt.plot(x, y, label="Normal Distribution")
plt.title("Normal Distribution (Mean=0, Std Dev=1)")
plt.xlabel("x")
plt.ylabel("Density")
plt.legend()
plt.grid()
plt.show()

# 4. Probability of Event:
print("\n# 4. Probability of an Event")
# Tossing a fair coin 3 times, probability of exactly 2 heads
n = 3  # number of trials
p = 0.5  # probability of success (head)
k = 2  # number of successes
def prob_event(n, k, p):
    return stats.binom.pmf(k, n, p)

prob = prob_event(n, k, p)
print(f"Probability of getting exactly {k} heads in {n} tosses: {prob:.4f}")

# 5. Confidence Intervals:
print("\n# 5. Confidence Interval")
sample_data = np.random.normal(loc=5, scale=2, size=50)  # Generate sample data
confidence_level = 0.95
mean = np.mean(sample_data)
sem = stats.sem(sample_data)  # Standard error of the mean
margin_of_error = sem * stats.t.ppf((1 + confidence_level) / 2., len(sample_data) - 1)
ci_lower = mean - margin_of_error
ci_upper = mean + margin_of_error
print(f"Sample Mean: {mean:.2f}, Confidence Interval (95%): [{ci_lower:.2f}, {ci_upper:.2f}]")

# 6. Correlation coefficient:
print("\n# 6. Correlation Coefficient")
x = np.random.rand(100)
y = 2 * x + np.random.normal(scale=0.1, size=100)  # Add slight noise
correlation = np.corrcoef(x, y)[0, 1]
print(f"Correlation coefficient between x and y: {correlation:.2f}")

# Visualization of Correlation
plt.figure(figsize=(8, 4))
plt.scatter(x, y, alpha=0.7, label=f"Correlation = {correlation:.2f}")
plt.title("Scatter Plot of x and y")
plt.xlabel("x")
plt.ylabel("y")
plt.legend()
plt.grid()
plt.show()
# 7. Hypothesis Testing:
print("\n# 7. Hypothesis Testing")
# One-sample t-test: Is the mean of the sample_data significantly different from 5?
stat, p_value = stats.ttest_1samp(sample_data, popmean=5)
alpha = 0.05  # Significance level
print(f"T-statistic: {stat:.2f}, p-value: {p_value:.4f}")
if p_value < alpha:
    print("Reject the null hypothesis: The mean is significantly different from 5.")
else:
    print("Fail to reject the null hypothesis: No significant difference from 5.")
