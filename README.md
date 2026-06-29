# Simulation-Based Trade-off Analysis of Sensing and Communication Performance in OFDM-Based 5G ISAC Systems

**Author:** Ifeanyichukwu Salvation Ude-Natha  
**Affiliation:** Department of Electrical/Electronic Engineering, University of Benin, Nigeria  
**Language:** Python (NumPy, SciPy, Matplotlib)

---

## What This Project Is About

5G and 6G networks are being designed to do two things at once with the same radio signal: **send data** to users and **sense the environment** like a radar. This is called **Integrated Sensing and Communication (ISAC)**.

The problem is that these two functions compete for the same radio resources. The more of the signal you dedicate to sensing, the less is left for communication — and vice versa. This project quantifies exactly where that trade-off lies in a real 5G New Radio (NR) system.

The key insight is that **5G already has a built-in parameter that controls this balance** — called `DMRSAdditionalPosition`. Rather than proposing a theoretical new waveform, this study sweeps that real, standard-defined parameter and measures what happens to both sensing and communication performance at each setting.

---

## Background: Key Concepts Explained

### What is OFDM?
OFDM (Orthogonal Frequency-Division Multiplexing) is the waveform used by 5G. It splits a radio signal across hundreds of subcarriers (narrow frequency channels) simultaneously. It is well-suited for ISAC because its structure makes it easy to extract radar information from the reflected signal using a mathematical operation called a 2D Fast Fourier Transform (2D FFT).

### What is DM-RS?
DM-RS (Demodulation Reference Signal) are known pilot symbols embedded in the 5G signal. Because they are deterministic (known in advance), they can be used as reliable sensing templates for radar processing — unlike random data symbols. The more DM-RS symbols you insert, the better the sensing performance, but the fewer resource elements remain for actual data.

### What is DMRSAdditionalPosition?
This is a standard 3GPP 5G NR parameter (defined in TS 38.211) that controls how many DM-RS pilot symbols are inserted per slot. It takes values from 0 to 3:

| DMRSAdditionalPosition | DM-RS symbols per slot | Pilot overhead |
|---|---|---|
| 0 | 1 | 7.1% |
| 1 | 2 | 14.3% |
| 2 | 3 | 21.4% |
| 3 | 4 | 28.6% |

This is the parameter swept in this study.

### What is a Pareto Trade-off?
A Pareto trade-off means you cannot improve one thing without making something else worse. In this context, no single `DMRSAdditionalPosition` value can simultaneously maximise both communication throughput and sensing accuracy. The results trace the Pareto-optimal boundary between the two.

### What is the CRLB?
The Cramér-Rao Lower Bound (CRLB) is an information-theoretic floor on how accurately any algorithm can estimate a parameter (like target velocity). It does not depend on which detection algorithm is used, making it a fair, algorithm-agnostic benchmark for sensing performance. In this study, it is operationalised through velocity metrics derived from the DM-RS configuration.

### What is a Range-Doppler Map?
A Range-Doppler map is a 2D radar image produced by applying a 2D FFT to the received pilot signals. One axis shows target range (distance), the other shows target velocity (Doppler shift). A bright peak at a specific range-velocity coordinate indicates a detected target.

---

## System Setup

| Parameter | Value |
|---|---|
| Carrier Frequency | 30 GHz |
| Channel Bandwidth | 50 MHz |
| Subcarrier Spacing | 60 kHz |
| Number of Subcarriers | 921 |
| Transmit Power | 46 dBm |
| Tx/Rx Antennas | 8 |
| Target Range | 80 m |
| Target Velocity | 30 m/s |
| Radar Cross Section | 1.0 m² |
| Slots Simulated | 4 |
| OFDM Symbols per Slot | 14 |

The simulation models a single 5G base station (gNodeB) simultaneously serving a downlink communication user and performing bistatic radar sensing of a moving target using the same transmitted waveform.

---

## What the Simulation Does

The simulation sweeps `DMRSAdditionalPosition` from 0 to 3 and at each value measures:

1. **Spectral Efficiency (bits/s/Hz)** — how much data can be delivered per unit of spectrum. This is the communication performance metric.
2. **Maximum Unambiguous Velocity (m/s)** — the fastest target the radar can detect without aliasing. Higher is better for sensing.
3. **Velocity Resolution (m/s)** — the minimum velocity difference between two targets the radar can distinguish. Lower is better for sensing.
4. **Peak Range-Doppler SINR (dB)** — the signal-to-noise ratio at the target peak on the Range-Doppler map. Higher means cleaner target detection.

---

## Results Summary

| DMRSAdditionalPosition | Pilot Overhead | Spectral Efficiency | Max Velocity | Velocity Resolution | Peak SINR |
|---|---|---|---|---|---|
| 0 | 7.1% | 20.85 bits/s/Hz | 10.71 m/s | 7.14 m/s | 42.61 dB |
| 1 | 14.3% | 19.24 bits/s/Hz | 16.66 m/s | 6.38 m/s | 45.62 dB |
| 2 | 21.4% | 17.64 bits/s/Hz | 29.98 m/s | 5.88 m/s | 47.38 dB |
| 3 | 28.6% | 16.04 bits/s/Hz | 29.98 m/s | 5.88 m/s | 48.63 dB |

**Key finding:** `DMRSAdditionalPosition = 2` is the practically optimal configuration for balanced ISAC operation. Velocity-related sensing metrics saturate at position 2, meaning pushing to position 3 offers no additional benefit on those metrics — only further communication degradation. Peak SINR continues to improve beyond position 2, making position 3 the better choice only when raw radar detection quality is the priority.

---

## Plots Generated

The simulation produces seven plots:

1. **Pilot Overhead vs DMRSAdditionalPosition** — establishes the resource cost driving the trade-off
2. **Spectral Efficiency vs DMRSAdditionalPosition** — communication performance degradation
3. **Max Unambiguous Velocity vs DMRSAdditionalPosition** — sensing capability improvement
4. **Velocity Resolution vs DMRSAdditionalPosition** — sensing discrimination improvement
5. **Range-Doppler Maps (4-panel)** — visual radar imagery at each configuration
6. **ISAC Trade-off Summary (dual-axis)** — headline figure showing the Pareto boundary
7. **Peak Range-Doppler SINR vs DMRSAdditionalPosition** — raw detection quality

---

## How to Run

**Requirements:**
```
pip install numpy scipy matplotlib
```

**Run the simulation:**
```
python isac_tradeoff_simulation.py
```

All seven plots will be generated and saved automatically.

---

## References

Key references this work builds on:

- Liu et al., "Integrated Sensing and Communications: Toward Dual-Functional Wireless Networks for 6G and Beyond," *IEEE JSAC*, 2022.
- Zhang et al., "An Overview of Signal Processing Techniques for Joint Communication and Radar Sensing," *IEEE JSTSP*, 2021.
- Sturm and Wiesbeck, "Waveform Design and Signal Processing Aspects for Fusion of Wireless Communications and Radar Sensing," *Proceedings of the IEEE*, 2011.
- Liu et al., "Cramér-Rao Bound Optimization for Joint Radar-Communication Design," *IEEE TSP*, 2022.
- Keskin et al., "Limited Feedforward Waveform Design for OFDM Dual-Functional Radar-Communications," *IEEE TSP*, 2021.

---

## About

This project is an independent research simulation completed as part of graduate application preparation and personal research development in 5G/6G wireless communications and ISAC systems.
