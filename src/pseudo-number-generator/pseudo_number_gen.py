import numpy as np
import matplotlib.pyplot as plt

no_of_samples = 100000000
seed = 42

multiplier = 23
incrementor = 43
modulus = 2**31 - 1   # larger modulus = better randomness

array_random = np.zeros(no_of_samples)
array_random[0] = seed % modulus

for i in range(1, no_of_samples):
    array_random[i] = (multiplier * array_random[i-1] + incrementor) % modulus

# normalize to [0,1)
array_random = array_random / modulus

plt.hist(array_random, bins=10)
plt.xlabel('Value')
plt.ylabel('Frequency')
plt.title('Pseudo Random Histogram')
plt.savefig("random_number_gen_histogram")

#chi square test
num_of_bins = 10

expected_frequency = no_of_samples / num_of_bins

actual_frequency, _ = np.histogram(array_random, bins=num_of_bins)

chi_square = np.sum((actual_frequency - expected_frequency) ** 2 / expected_frequency)

confidence = 0.05
degree_of_freedom = num_of_bins - 1

print(f"Chi-Squared Test Results:")
print(f"chi_square: ", chi_square)
print(f"degree of freedom", degree_of_freedom)
print(f"confidence", confidence)

#auto correlation


