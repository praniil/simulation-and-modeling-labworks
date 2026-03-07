import numpy as np
import matplotlib.pyplot as plt
from scipy import stats

# 1. LCG Random Number Generation
no_of_samples = 1000000
seed          = 42
multiplier    = 1103515245
incrementor   = 12345
modulus       = 2**31

array_random = np.zeros(no_of_samples)
array_random[0] = seed % modulus

for i in range(1, no_of_samples):
    array_random[i] = (multiplier * array_random[i-1] + incrementor) % modulus

array_random = array_random / modulus

# 2. Histogram (Uniformity - Graphical)
num_of_bins = 10

plt.figure(figsize=(8, 4))
plt.hist(array_random, bins=num_of_bins, edgecolor='black')
plt.xlabel('Value')
plt.ylabel('Frequency')
plt.title('Pseudo Random Number Histogram (Uniformity Check)')
plt.tight_layout()
plt.savefig("random_number_gen_histogram.png")
plt.show()

# 3. Chi-Squared Test
expected_frequency  = no_of_samples / num_of_bins
actual_frequency, _ = np.histogram(array_random, bins=num_of_bins)

chi_square          = np.sum((actual_frequency - expected_frequency)**2 / expected_frequency)
confidence          = 0.05
degree_of_freedom   = num_of_bins - 1
chi_square_critical = stats.chi2.ppf(1 - confidence, df=degree_of_freedom)

print("=" * 50)
print("Chi-Squared Test Results:")
print(f"  Computed chi-square value : {chi_square:.4f}")
print(f"  Critical chi-square value : {chi_square_critical:.4f}")
print(f"  Degrees of freedom        : {degree_of_freedom}")
print(f"  Significance level (a)    : {confidence}")
if chi_square < chi_square_critical:
    print("  Result: ACCEPT H0 - Numbers are uniformly distributed.")
else:
    print("  Result: REJECT H0 - Numbers are NOT uniformly distributed.")
print("=" * 50)

# 4. Kolmogorov-Smirnov Test
ks_stat, ks_p_value = stats.kstest(array_random, 'uniform')

print("\nKolmogorov-Smirnov Test Results:")
print(f"  KS Statistic : {ks_stat:.6f}")
print(f"  p-value      : {ks_p_value:.6f}")
if ks_p_value > confidence:
    print("  Result: ACCEPT H0 - Numbers follow uniform distribution.")
else:
    print("  Result: REJECT H0 - Numbers do NOT follow uniform distribution.")
print("=" * 50)

# 5. Autocorrelation (Independence Test)
subset   = array_random[:5000]
max_lag  = 50
n        = len(subset)
mean     = np.mean(subset)
variance = np.var(subset)

autocorr_values = np.zeros(max_lag + 1)
for k in range(max_lag + 1):
    autocorr_values[k] = (
        np.sum((subset[:n-k] - mean) * (subset[k:] - mean))
        / ((n - k) * variance)
    )

plt.figure(figsize=(10, 4))
plt.stem(range(max_lag + 1), autocorr_values, markerfmt='C0o', basefmt='k-')

conf_bound = 1.96 / np.sqrt(n)
plt.axhline( conf_bound, color='red', linestyle='--', label=f'95% Confidence Bound (±{conf_bound:.4f})')
plt.axhline(-conf_bound, color='red', linestyle='--')
plt.xlabel('Lag (k)')
plt.ylabel('Autocorrelation R(k)')
plt.title('Autocorrelation of Generated Random Numbers')
plt.legend()
plt.tight_layout()
plt.savefig("autocorrelation_plot.png")
plt.show()

print("\nAutocorrelation Test Results:")
print(f"  95% Confidence bound           : ±{conf_bound:.6f}")
print(f"  Max autocorrelation at lag > 0 : {np.max(np.abs(autocorr_values[1:])):.6f}")
if np.all(np.abs(autocorr_values[1:]) < conf_bound):
    print("  Result: Numbers are INDEPENDENT (all lags within bounds).")
else:
    print("  Result: Possible correlation detected.")
print("=" * 50)