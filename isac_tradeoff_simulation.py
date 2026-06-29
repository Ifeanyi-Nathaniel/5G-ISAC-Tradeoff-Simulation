"""
=============================================================================
SIMULATION-BASED TRADE-OFF ANALYSIS OF SENSING AND COMMUNICATION PERFORMANCE
IN OFDM-BASED 5G ISAC SYSTEMS
=============================================================================
Description:
    This script simulates an OFDM-based 5G Integrated Sensing and
    Communication (ISAC) system. The trade-off between communication
    throughput and sensing performance is investigated by varying the
    DMRSAdditionalPosition parameter from 0 to 3.

    MEASUREMENTS MADE:
      Communication Side:
        1. Pilot Overhead (%)
        2. Spectral Efficiency (bits/s/Hz)

      Sensing Side:
        3. Maximum Unambiguous Doppler Shift (Hz)
        4. Maximum Unambiguous Velocity (m/s)
        5. Velocity Resolution (m/s)
        6. Peak SINR on Range-Doppler Map (dB)

    PLOTS PRODUCED:
      Plot 1  - Pilot Overhead (%) vs DMRSAdditionalPosition
      Plot 2  - Spectral Efficiency vs DMRSAdditionalPosition
      Plot 3  - Max Unambiguous Velocity vs DMRSAdditionalPosition
      Plot 4  - Velocity Resolution vs DMRSAdditionalPosition
      Plot 5  - Range-Doppler Maps (2D heatmaps, 4-panel subplot)
      Plot 6  - Trade-off Summary: Dual-axis (Spectral Efficiency +
                Max Velocity) vs DMRSAdditionalPosition  [HEADLINE FIGURE]

Requirements:
    pip install numpy scipy matplotlib
=============================================================================
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from scipy.constants import speed_of_light

# =============================================================================
# SYSTEM PARAMETERS  (5G NR, consistent with 3GPP NR standard)
# =============================================================================

fc          = 30e9          # Carrier frequency: 30 GHz
BW          = 50e6          # Channel bandwidth: 50 MHz
BW_occ      = 0.90          # Bandwidth occupancy: 90 %
delta_f     = 60e3          # Subcarrier spacing: 60 kHz (numerology mu=2)
N_FFT       = 1024          # FFT size
Fs          = 61.44e6       # Sample rate: 61.44 MHz

P_tx_dBm    = 46            # Transmit power: 46 dBm
P_tx        = 10**((P_tx_dBm - 30) / 10)   # Watts
NF_dB       = 5             # Noise figure: 5 dB
NF          = 10**(NF_dB / 10)
k_B         = 1.38e-23      # Boltzmann constant
T0          = 290           # Reference temperature (K)
N0          = k_B * T0 * NF * BW           # Noise power (W)

# 5G NR frame structure (mu=2, 60 kHz SCS)
N_slots_per_subframe = 4
N_symbols_per_slot   = 14
N_slots              = 4    # Slots to simulate
N_symbols_total      = N_slots * N_symbols_per_slot   # 56 symbols

# Active subcarriers
N_active = int(N_FFT * BW_occ)   # 921 subcarriers

# Antenna configuration (ULA)
N_tx = 8
N_rx = 8
array_gain = N_tx * N_rx   # = 64

# Target parameters
R_target  = 80.0    # Range (m)
v_target  = 9.5     # Radial velocity (m/s) — matches MathWorks example Target 1
RCS       = 1.0     # Radar cross-section (m²)
wavelength = speed_of_light / fc

# SNR for Range-Doppler map simulation
SNR_RD_dB = 15.0    # dB — representative operating SNR

# =============================================================================
# DM-RS CONFIGURATION
# DMRSAdditionalPosition controls extra pilot symbol rows per slot.
#   pos=0 → 1 DM-RS symbol/slot   (minimum pilot overhead)
#   pos=1 → 2 DM-RS symbols/slot
#   pos=2 → 3 DM-RS symbols/slot
#   pos=3 → 4 DM-RS symbols/slot  (maximum pilot overhead)
#
# In 5G NR the DM-RS symbols occupy specific positions within a 14-symbol slot.
# The standard positions used here follow 3GPP TS 38.211 Table 7.4.1.1.2-3
# for mapping type A, single-symbol DM-RS:
#   pos=0 → symbol 2
#   pos=1 → symbols 2, 7
#   pos=2 → symbols 2, 7, 11
#   pos=3 → symbols 2, 5, 8, 11
# =============================================================================

DMRS_positions  = [0, 1, 2, 3]
DMRS_sym_per_slot = {0: 1, 1: 2, 2: 3, 3: 4}

# Symbol indices within a slot (0-based) for each configuration
DMRS_sym_indices = {
    0: [2],
    1: [2, 7],
    2: [2, 7, 11],
    3: [2, 5, 8, 11],
}

# =============================================================================
# MEASUREMENT FUNCTIONS
# =============================================================================

def pilot_overhead(dmrs_pos):
    """
    Pilot overhead fraction = DM-RS REs / Total REs.
    Returned as a percentage.
    """
    n_dmrs  = DMRS_sym_per_slot[dmrs_pos] * N_slots
    n_total = N_symbols_total
    return (n_dmrs / n_total) * 100.0


def spectral_efficiency(dmrs_pos):
    """
    Spectral efficiency (bits/s/Hz).

    SE = (data REs / total REs) * log2(1 + SNR_comm)

    The per-subcarrier communication SNR is computed from a simple free-space
    link budget.
    """
    n_dmrs_sym   = DMRS_sym_per_slot[dmrs_pos]
    n_data_sym   = N_symbols_per_slot - n_dmrs_sym   # data symbols per slot
    data_RE_frac = n_data_sym / N_symbols_per_slot    # fraction of slot for data

    # Free-space path loss at R_target
    path_loss = (4 * np.pi * R_target * fc / speed_of_light) ** 2
    P_rx      = P_tx / path_loss
    SNR_comm  = (P_rx * N_active) / N0               # per-subcarrier SNR

    SE = data_RE_frac * np.log2(1 + SNR_comm)
    return SE, SNR_comm


def max_unambiguous_doppler(dmrs_pos):
    """
    Maximum unambiguous Doppler shift (Hz).

    f_D_max = 1 / (2 * Mt * T_OFDM)

    where Mt is the maximum gap between consecutive DM-RS symbols (in symbols)
    and T_OFDM is the OFDM symbol duration.

    A larger Mt (sparser pilots) means a smaller maximum Doppler.
    """
    sym_indices = DMRS_sym_indices[dmrs_pos]

    # Extend indices across all slots
    all_indices = []
    for s in range(N_slots):
        all_indices.extend([idx + s * N_symbols_per_slot for idx in sym_indices])

    # Maximum separation between consecutive DM-RS symbols
    if len(all_indices) > 1:
        Mt = max(np.diff(all_indices))
    else:
        # Only one pilot across all slots → use full frame length
        Mt = N_symbols_total - 1

    T_OFDM = 1.0 / delta_f     # OFDM symbol duration (s), ignoring CP
    f_D_max = 1.0 / (2.0 * Mt * T_OFDM)
    return f_D_max, Mt, T_OFDM


def max_unambiguous_velocity(f_D_max):
    """
    Convert max Doppler to max unambiguous radial velocity (m/s).

    v_max = f_D_max * lambda / 2
    """
    return f_D_max * wavelength / 2.0


def velocity_resolution(dmrs_pos):
    """
    Velocity resolution (m/s).

    The Doppler resolution is determined by the total observation time
    spanned by the DM-RS symbols:

    delta_f_D = 1 / T_obs

    where T_obs = (last DM-RS symbol index - first) * T_OFDM.
    Velocity resolution = delta_f_D * lambda / 2.
    """
    sym_indices = DMRS_sym_indices[dmrs_pos]
    all_indices = []
    for s in range(N_slots):
        all_indices.extend([idx + s * N_symbols_per_slot for idx in sym_indices])

    T_OFDM = 1.0 / delta_f
    T_obs   = (max(all_indices) - min(all_indices)) * T_OFDM

    if T_obs == 0:
        T_obs = T_OFDM  # single pilot: resolution limited to one symbol

    delta_fD  = 1.0 / T_obs
    delta_v   = delta_fD * wavelength / 2.0
    return delta_v


def peak_sinr_range_doppler(dmrs_pos):
    """
    Simulated peak SINR on the Range-Doppler map (dB).

    The Range-Doppler map is built from N_sense pilot symbols, each carrying
    N_active subcarriers.

    Coherent processing gain:
      - Range dimension : N_active subcarriers
      - Doppler dimension: N_sense DM-RS symbols

    Peak SINR (dB) = SNR_per_RE (dB) + 10*log10(N_active * N_sense)

    This reflects the fact that more DM-RS symbols → more coherent
    integration → better SINR on the map.
    """
    n_sense = DMRS_sym_per_slot[dmrs_pos] * N_slots

    # Radar SNR per resource element (bistatic radar equation + array gain)
    SNR_rad_per_RE = (P_tx * array_gain * RCS * wavelength**2) / \
                     ((4 * np.pi)**3 * R_target**4 * N0)

    # Coherent processing gain
    proc_gain = N_active * n_sense

    peak_sinr_linear = SNR_rad_per_RE * proc_gain
    peak_sinr_dB     = 10 * np.log10(peak_sinr_linear)
    return peak_sinr_dB, n_sense


# =============================================================================
# RANGE-DOPPLER MAP SIMULATION
# =============================================================================

def simulate_range_doppler_map(dmrs_pos, SNR_dB=SNR_RD_dB):
    """
    Simulate a Range-Doppler map for the given DMRSAdditionalPosition.

    The map is built by:
      1. Creating a 2D pilot matrix (N_active subcarriers x N_sense symbols).
      2. Injecting a target echo at (R_target, v_target) with the given SNR.
      3. Adding AWGN noise.
      4. Taking a 2D FFT (IFFT along frequency → range, FFT along time → Doppler).
      5. Returning the log-magnitude map with labelled axes.
    """
    n_sense  = DMRS_sym_per_slot[dmrs_pos] * N_slots
    SNR_lin  = 10 ** (SNR_dB / 10)

    # --- Frequency and time axes ---
    # Subcarrier frequencies (centred)
    f_k = (np.arange(N_active) - N_active // 2) * delta_f   # Hz

    # DM-RS symbol times
    sym_indices = DMRS_sym_indices[dmrs_pos]
    all_sym = []
    for s in range(N_slots):
        all_sym.extend([idx + s * N_symbols_per_slot for idx in sym_indices])
    T_OFDM   = 1.0 / delta_f
    t_n      = np.array(all_sym) * T_OFDM    # seconds

    # --- Build received signal matrix (N_active x n_sense) ---
    # Target delay
    tau_target = 2 * R_target / speed_of_light      # two-way delay (s)
    # Target Doppler
    f_D_target = 2 * v_target * fc / speed_of_light  # Doppler shift (Hz)

    # Signal amplitude from SNR
    signal_amp = np.sqrt(SNR_lin)

    # Phase matrix: phase from delay (along freq) + Doppler (along time)
    Y = np.zeros((N_active, n_sense), dtype=complex)
    for ki, fk in enumerate(f_k):
        for ni, tn in enumerate(t_n):
            # Target contribution
            Y[ki, ni] = signal_amp * np.exp(-1j * 2 * np.pi * fk * tau_target) \
                                   * np.exp( 1j * 2 * np.pi * f_D_target * tn)

    # Add AWGN noise (unit variance)
    noise = (np.random.randn(N_active, n_sense)
             + 1j * np.random.randn(N_active, n_sense)) / np.sqrt(2)
    Y += noise

    # --- 2D processing ---
    # IFFT along frequency axis → range profile
    range_profile = np.fft.ifft(Y, n=N_active, axis=0) * N_active

    # FFT along slow-time axis → Doppler profile
    rd_map = np.fft.fftshift(np.fft.fft(range_profile, n=N_active, axis=1), axes=1)

    # Log-magnitude map
    rd_map_dB = 20 * np.log10(np.abs(rd_map) + 1e-12)

    # --- Range and Doppler axes ---
    range_res = speed_of_light / (2 * N_active * delta_f)
    range_ax  = np.arange(N_active) * range_res           # metres

    f_D_max_sim = 1.0 / (2.0 * T_OFDM)
    doppler_ax  = np.linspace(-f_D_max_sim, f_D_max_sim, N_active)
    velocity_ax = doppler_ax * wavelength / 2.0            # m/s

    return rd_map_dB, range_ax, velocity_ax


# =============================================================================
# MAIN SIMULATION LOOP
# =============================================================================

print("=" * 65)
print("  5G ISAC TRADE-OFF SIMULATION")
print("  Varying DMRSAdditionalPosition: 0 → 3")
print("=" * 65)

results = []

for dp in DMRS_positions:
    oh           = pilot_overhead(dp)
    SE, SNR_comm = spectral_efficiency(dp)
    f_D_max, Mt, T_OFDM = max_unambiguous_doppler(dp)
    v_max        = max_unambiguous_velocity(f_D_max)
    v_res        = velocity_resolution(dp)
    sinr_dB, n_s = peak_sinr_range_doppler(dp)

    results.append({
        'dmrs_pos'   : dp,
        'n_dmrs_sym' : DMRS_sym_per_slot[dp],
        'pilot_oh'   : oh,
        'SE'         : SE,
        'f_D_max'    : f_D_max,
        'v_max'      : v_max,
        'v_res'      : v_res,
        'sinr_dB'    : sinr_dB,
        'n_sense'    : n_s,
        'Mt'         : Mt,
    })

    print(f"\n  DMRSAdditionalPosition = {dp}")
    print(f"  DM-RS symbols/slot     : {DMRS_sym_per_slot[dp]}")
    print(f"  Pilot overhead         : {oh:.1f} %")
    print(f"  Spectral efficiency    : {SE:.4f} bits/s/Hz")
    print(f"  Max pilot gap (Mt)     : {Mt} symbols")
    print(f"  Max unambiguous Doppler: {f_D_max/1e3:.3f} kHz")
    print(f"  Max unambiguous velocity:{v_max:.3f} m/s")
    print(f"  Velocity resolution    : {v_res:.4f} m/s")
    print(f"  Peak R-D map SINR      : {sinr_dB:.2f} dB")

print("\n" + "=" * 65)

# --- Summary table ---
print("\n  RESULTS SUMMARY TABLE")
hdr = f"  {'pos':<5} {'Pilot OH%':<12} {'SE(b/s/Hz)':<13} {'v_max(m/s)':<13} {'v_res(m/s)':<13} {'SINR(dB)':<10}"
print(hdr)
print("  " + "-" * 66)
for r in results:
    print(f"  {r['dmrs_pos']:<5} {r['pilot_oh']:<12.1f} {r['SE']:<13.4f} "
          f"{r['v_max']:<13.3f} {r['v_res']:<13.4f} {r['sinr_dB']:<10.2f}")

# =============================================================================
# EXTRACT ARRAYS FOR PLOTTING
# =============================================================================

dp_vals   = [r['dmrs_pos']  for r in results]
oh_vals   = [r['pilot_oh']  for r in results]
SE_vals   = [r['SE']        for r in results]
vmax_vals = [r['v_max']     for r in results]
vres_vals = [r['v_res']     for r in results]
sinr_vals = [r['sinr_dB']   for r in results]

# Colour palette — consistent across all plots
BLUE   = '#1f77b4'
RED    = '#d62728'
GREEN  = '#2ca02c'
PURPLE = '#9467bd'
ORANGE = '#ff7f0e'

MARKER_STYLE = dict(linewidth=2.5, markersize=9)

# =============================================================================
# PLOT 1 — Pilot Overhead (%) vs DMRSAdditionalPosition
# =============================================================================

fig1, ax = plt.subplots(figsize=(7, 4.5))
bars = ax.bar(dp_vals, oh_vals, color=[BLUE, ORANGE, GREEN, RED],
              edgecolor='black', linewidth=0.8, width=0.55)
for bar, val in zip(bars, oh_vals):
    ax.text(bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.3, f'{val:.1f}%',
            ha='center', va='bottom', fontsize=10, fontweight='bold')
ax.set_xlabel('DMRSAdditionalPosition', fontsize=12)
ax.set_ylabel('Pilot Overhead (%)', fontsize=12)
ax.set_title('Plot 1 — Pilot Overhead vs DMRSAdditionalPosition\n'
             '(Resource cost driving the ISAC trade-off)', fontsize=11)
ax.set_xticks(dp_vals)
ax.set_ylim(0, max(oh_vals) * 1.25)
ax.grid(axis='y', linestyle='--', alpha=0.5)
fig1.tight_layout()
fig1.savefig('plot1_pilot_overhead.pdf', bbox_inches='tight')

# =============================================================================
# PLOT 2 — Spectral Efficiency vs DMRSAdditionalPosition
# =============================================================================

fig2, ax = plt.subplots(figsize=(7, 4.5))
ax.plot(dp_vals, SE_vals, 'o-', color=BLUE, **MARKER_STYLE, label='Spectral Efficiency')
for x, y in zip(dp_vals, SE_vals):
    ax.annotate(f'{y:.3f}', (x, y),
                textcoords='offset points', xytext=(0, 10),
                ha='center', fontsize=9)
ax.set_xlabel('DMRSAdditionalPosition', fontsize=12)
ax.set_ylabel('Spectral Efficiency (bits/s/Hz)', fontsize=12)
ax.set_title('Plot 2 — Spectral Efficiency vs DMRSAdditionalPosition\n'
             '(Communication performance degrades as pilots increase)', fontsize=11)
ax.set_xticks(dp_vals)
ax.grid(linestyle='--', alpha=0.5)
ax.legend(fontsize=10)
fig2.tight_layout()
fig2.savefig('plot2_spectral_efficiency.pdf', bbox_inches='tight')

# =============================================================================
# PLOT 3 — Max Unambiguous Velocity vs DMRSAdditionalPosition
# =============================================================================

fig3, ax = plt.subplots(figsize=(7, 4.5))
ax.plot(dp_vals, vmax_vals, 's-', color=GREEN, **MARKER_STYLE,
        label='Max Unambiguous Velocity')
for x, y in zip(dp_vals, vmax_vals):
    ax.annotate(f'{y:.2f} m/s', (x, y),
                textcoords='offset points', xytext=(0, 10),
                ha='center', fontsize=9)
ax.set_xlabel('DMRSAdditionalPosition', fontsize=12)
ax.set_ylabel('Max Unambiguous Velocity (m/s)', fontsize=12)
ax.set_title('Plot 3 — Max Unambiguous Velocity vs DMRSAdditionalPosition\n'
             '(Sensing capability ceiling improves with denser pilots)', fontsize=11)
ax.set_xticks(dp_vals)
ax.grid(linestyle='--', alpha=0.5)
ax.legend(fontsize=10)
fig3.tight_layout()
fig3.savefig('plot3_max_velocity.pdf', bbox_inches='tight')

# =============================================================================
# PLOT 4 — Velocity Resolution vs DMRSAdditionalPosition
# =============================================================================

fig4, ax = plt.subplots(figsize=(7, 4.5))
ax.plot(dp_vals, vres_vals, '^-', color=PURPLE, **MARKER_STYLE,
        label='Velocity Resolution')
for x, y in zip(dp_vals, vres_vals):
    ax.annotate(f'{y:.4f} m/s', (x, y),
                textcoords='offset points', xytext=(0, 10),
                ha='center', fontsize=9)
ax.set_xlabel('DMRSAdditionalPosition', fontsize=12)
ax.set_ylabel('Velocity Resolution (m/s)', fontsize=12)
ax.set_title('Plot 4 — Velocity Resolution vs DMRSAdditionalPosition\n'
             '(Finer resolution as observation time grows with more pilots)', fontsize=11)
ax.set_xticks(dp_vals)
ax.grid(linestyle='--', alpha=0.5)
ax.legend(fontsize=10)
fig4.tight_layout()
fig4.savefig('plot4_velocity_resolution.pdf', bbox_inches='tight')

# =============================================================================
# PLOT 5 — Range-Doppler Maps (4-panel subplot, one per configuration)
# =============================================================================

np.random.seed(42)   # reproducibility

fig5, axes5 = plt.subplots(2, 2, figsize=(14, 10))
fig5.suptitle('Plot 5 — Simulated Range-Doppler Maps\n'
              f'(Target at R={R_target} m, v={v_target} m/s, SNR={SNR_RD_dB} dB)',
              fontsize=13, fontweight='bold')

for idx, dp in enumerate(DMRS_positions):
    ax = axes5[idx // 2][idx % 2]
    rd_map_dB, range_ax, vel_ax = simulate_range_doppler_map(dp)

    # Focus on region of interest around target
    r_lo, r_hi = 0, 200       # metres
    v_lo, v_hi = -50, 50      # m/s
    r_mask = (range_ax >= r_lo) & (range_ax <= r_hi)
    v_mask = (vel_ax  >= v_lo) & (vel_ax  <= v_hi)

    Z    = rd_map_dB[np.ix_(r_mask, v_mask)]
    R_ax = range_ax[r_mask]
    V_ax = vel_ax[v_mask]

    im = ax.pcolormesh(V_ax, R_ax, Z, cmap='jet',
                       vmin=np.percentile(Z, 10), vmax=np.percentile(Z, 99.5),
                       shading='auto')
    plt.colorbar(im, ax=ax, label='Magnitude (dB)')

    # Mark true target position
    ax.axvline(v_target, color='white', linewidth=1.5, linestyle='--', alpha=0.8)
    ax.axhline(R_target, color='white', linewidth=1.5, linestyle='--', alpha=0.8)
    ax.plot(v_target, R_target, 'w*', markersize=14,
            label=f'Target ({v_target} m/s, {R_target} m)')

    n_sense = DMRS_sym_per_slot[dp] * N_slots
    ax.set_title(f'DMRSAdditionalPosition = {dp}  '
                 f'({DMRS_sym_per_slot[dp]} pilot sym/slot, {n_sense} total)',
                 fontsize=10)
    ax.set_xlabel('Velocity (m/s)', fontsize=10)
    ax.set_ylabel('Range (m)', fontsize=10)
    ax.legend(loc='upper right', fontsize=8, framealpha=0.7)

fig5.tight_layout()
fig5.savefig('plot5_range_doppler_maps.pdf', bbox_inches='tight')

# =============================================================================
# PLOT 6 — HEADLINE TRADE-OFF SUMMARY (dual y-axis)
# =============================================================================

fig6, ax6a = plt.subplots(figsize=(8, 5))

color_comm   = BLUE
color_sense  = RED

# Left axis: Spectral Efficiency (communication)
line1, = ax6a.plot(dp_vals, SE_vals, 'o-', color=color_comm,
                   linewidth=2.5, markersize=9, label='Spectral Efficiency (comm.)')
ax6a.set_xlabel('DMRSAdditionalPosition', fontsize=12)
ax6a.set_ylabel('Spectral Efficiency (bits/s/Hz)', fontsize=12, color=color_comm)
ax6a.tick_params(axis='y', labelcolor=color_comm)
ax6a.set_xticks(dp_vals)

# Right axis: Max Unambiguous Velocity (sensing)
ax6b = ax6a.twinx()
line2, = ax6b.plot(dp_vals, vmax_vals, 's--', color=color_sense,
                   linewidth=2.5, markersize=9, label='Max Unambiguous Velocity (sens.)')
ax6b.set_ylabel('Max Unambiguous Velocity (m/s)', fontsize=12, color=color_sense)
ax6b.tick_params(axis='y', labelcolor=color_sense)

# Annotate communication values
for x, y in zip(dp_vals, SE_vals):
    ax6a.annotate(f'{y:.3f}', (x, y),
                  textcoords='offset points', xytext=(-18, 8),
                  color=color_comm, fontsize=8.5)

# Annotate sensing values
for x, y in zip(dp_vals, vmax_vals):
    ax6b.annotate(f'{y:.2f}', (x, y),
                  textcoords='offset points', xytext=(6, 8),
                  color=color_sense, fontsize=8.5)

# Shaded regions to highlight trade-off zones
ax6a.axvspan(-0.4, 0.5, alpha=0.07, color=BLUE,  label='Comm-favoured zone')
ax6a.axvspan( 2.5, 3.4, alpha=0.07, color=RED,   label='Sensing-favoured zone')

ax6a.set_title('Plot 6 — HEADLINE: ISAC Trade-off Summary\n'
               'Communication (↓) vs Sensing Capability (↑) '
               'as Pilot Density Increases',
               fontsize=11, fontweight='bold')

# Unified legend
lines  = [line1, line2]
labels = [l.get_label() for l in lines]
ax6a.legend(lines, labels, loc='center right', fontsize=10, framealpha=0.9)

ax6a.grid(linestyle='--', alpha=0.4)
ax6a.set_xlim(-0.4, 3.4)
fig6.tight_layout()
fig6.savefig('plot6_tradeoff_summary.pdf', bbox_inches='tight')

# =============================================================================
# DONE
# =============================================================================
# =============================================================================
# PLOT 7 — Peak SINR on Range-Doppler Map vs DMRSAdditionalPosition
# =============================================================================

fig7, ax = plt.subplots(figsize=(7, 4.5))
ax.plot(dp_vals, sinr_vals, 'D-', color=ORANGE, **MARKER_STYLE,
        label='Peak R-D Map SINR')
for x, y in zip(dp_vals, sinr_vals):
    ax.annotate(f'{y:.2f} dB', (x, y),
                textcoords='offset points', xytext=(0, 10),
                ha='center', fontsize=9)
ax.set_xlabel('DMRSAdditionalPosition', fontsize=12)
ax.set_ylabel('Peak SINR on Range-Doppler Map (dB)', fontsize=12)
ax.set_title('Plot 7 — Peak Range-Doppler SINR vs DMRSAdditionalPosition\n'
             '(More pilot symbols → more coherent integration → higher SINR)',
             fontsize=11)
ax.set_xticks(dp_vals)
ax.grid(linestyle='--', alpha=0.5)
ax.legend(fontsize=10)
fig7.tight_layout()
fig7.savefig('plot7_peak_sinr.pdf', bbox_inches='tight')


plt.show()

print("\n  All plots saved:")
print("    plot1_pilot_overhead.pdf")
print("    plot2_spectral_efficiency.pdf")
print("    plot3_max_velocity.pdf")
print("    plot4_velocity_resolution.pdf")
print("    plot5_range_doppler_maps.pdf")
print("    plot6_tradeoff_summary.pdf  ← headline figure")
print("    plot7_peak_sinr.pdf")
print("\n  Simulation complete.")
