import numpy as np
import matplotlib.pyplot as plt

# %% Defining amplifier parameters
beta = 100
V_BE = 0.7
V_CC = 12
R1 = 10e4
R2 = 2.2e3
RE = 4.7e2
RC = 2.2e3
RL = 10e3
C_in = 10e-6
C_out = 10e-6
C_e = 100e-6

# %% Performing DC Bias Analysis
Vin_amp = 10e-3
f_in = 1e3

V_TH = V_CC * (R2 / (R1 + R2))
R_TH = (R1 * R2) / (R1 + R2)

Ib = (V_TH - V_BE) / (R_TH + (beta + 1) * RE)
Ic = beta * Ib
Ie = Ic + Ib

Ve = Ie * RE
Vb = Ve + V_BE
Vc = V_CC - Ic * RC
V_CE = Vc - Ve

VT = 25e-3
re = VT / Ie
gm = Ic / VT

# Print DC operating point
print("Ib =", Ib, "A")
print("Ic =", Ic, "A")
print("Ie =", Ie, "A")
print("Vb =", Vb, "V")
print("Ve =", Ve, "V")
print("Vc =", Vc, "V")
print("V_CE =", V_CE, "V")

# %% Frequency Response Analysis
f = np.logspace(0, 6, 1000)  # 1 Hz to 1 MHz
w = 2 * np.pi * f

ZCin = 1 / (1j * w * C_in)
ZCout = 1 / (1j * w * C_out)
ZCe = 1 / (1j * w * C_e)

Zeeff = re + (RE * ZCe) / (RE + ZCe)
Zin = (R_TH * beta * Zeeff) / (R_TH + beta * Zeeff)

H_in = Zin / (Zin + ZCin)
H_em = re / Zeeff
H_out = RL / (RL + ZCout)
Av_load = -gm * (RC * RL) / (RC + RL)

H_total = Av_load * H_in * H_em * H_out

Mag = 20 * np.log10(np.abs(H_total))
Phase = np.angle(H_total, deg=True)

# %% Plot Bode Magnitude and Phase and save
plt.figure(figsize=(8,6))
plt.subplot(2,1,1)
plt.semilogx(f, Mag)
plt.ylabel("Magnitude (dB)")
plt.title("Frequency Response (Bode Plot)")
plt.grid(True, which="both")

plt.subplot(2,1,2)
plt.semilogx(f, Phase)
plt.xlabel("Frequency (Hz)")
plt.ylabel("Phase (degrees)")
plt.grid(True, which="both")

plt.tight_layout()
plt.savefig("images/bode_plot.png")
plt.show()

# %% Time-Domain Response
T = 1 / f_in
t = np.linspace(0, 5 * T, 5000)
vin = Vin_amp * np.sin(2 * np.pi * f_in * t)

# Gain and phase at input frequency
idx = np.argmin(np.abs(f - f_in))
Gain_f = np.abs(H_total[idx])
Phase_f = np.angle(H_total[idx])

vout = Gain_f * Vin_amp * np.sin(2 * np.pi * f_in * t + Phase_f)

# %% Plot Input and Output signals and save
plt.figure(figsize=(8,4))
plt.plot(t, vin, label="Input Voltage")
plt.plot(t, vout, label="Output Voltage")
plt.xlabel("Time (seconds)")
plt.ylabel("Voltage (V)")
plt.title("Time Domain Response")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig("images/time_domain_plot.png")
plt.show()