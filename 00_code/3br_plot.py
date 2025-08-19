from matplotlib import pyplot as plt
import argparse
import os
import re
import numpy as np
import rebound as rb

def get_orbital_periods(sim):
    planets = sim.particles[1:sim.N_active]
    star = sim.particles[0]
    orbits = [planet.orbit(primary = star) for planet in planets]
    periods = np.array([orbit.P for orbit in orbits])
    return periods

def get_mean_longitudes(sim):
    planets = sim.particles[1:sim.N_active]
    star = sim.particles[0]
    orbits = [planet.orbit(primary = star) for planet in planets]
    mean_longitudes = np.array([orbit.l for orbit in orbits])
    return mean_longitudes

from scipy.signal import butter, filtfilt
def low_pass_filter(data, cutoff, fs, order=5):
    """
    Apply a low-pass Butterworth filter to a 1D NumPy array.

    Parameters
    ----------
    data : np.ndarray
        The input signal.
    cutoff : float
        The cutoff frequency of the filter (in Hz).
    fs : float
        The sampling frequency of the data (in Hz).
    order : int, optional
        The order of the filter (default is 5).

    Returns
    -------
    filtered_data : np.ndarray
        The filtered signal.
    """
    nyquist = 0.5 * fs
    normal_cutoff = cutoff / nyquist  # Normalize
    b, a = butter(order, normal_cutoff, btype='low', analog=False)
    filtered_data = filtfilt(b, a, data)
    return filtered_data

def sample_periods_and_mean_longitudes_at_Tsyn(sim,Tfin,eta=0.49):
    """
    Sample orbital periods and mean longitudes of a REBOUND simulation 
    at regular intervals based on a fraction of the synodic period 
    between the first two planets.

    Parameters
    ----------
    sim : rebound.Simulation
        A REBOUND Simulation object containing at least three particles:
        the central star (index 0) and at least two planets (indices 1 and 2).
    Tfin : float
        Total integration time in simulation units.
    eta : float, optional
        Fraction of the synodic period between planets 1 and 2 used as the 
        sampling interval, where `n_sample ≈ eta * Tsyn / dt`. 
        Default is 0.49.

    Returns
    -------
    periods : numpy.ndarray, shape (N_pl, N_out)
        Array of sampled orbital periods for each planet (rows) at each output time (columns).
    mean_longitudes : numpy.ndarray, shape (N_pl, N_out)
        Array of sampled mean longitudes for each planet (rows) at each output time (columns).
    """

    Tsyn = 2*np.pi / ( sim.particles[1].n - sim.particles[2].n )
    n_sample = int(np.floor(eta * Tsyn / sim.dt))
    N_out = int(np.ceil(Tfin / (n_sample * sim.dt)))
    periods = []
    mean_longitudes = []
    times = np.zeros(N_out)
    for i in range(N_out):
        times[i] = sim.t
        periods.append(get_orbital_periods(sim))
        mean_longitudes.append(get_mean_longitudes(sim))
        sim.steps(n_sample)
    periods = np.transpose(periods)
    mean_longitudes = np.transpose(mean_longitudes)
    return times, periods, mean_longitudes

from three_body_mmr_widths import get_n0_Q3br_dndL_Minv, eval_at_zero
def three_body_resonance_plot_points(masses,j_in,j_out,Delta_in,Delta_out,N_Delta):
    Deltas = np.linspace(Delta_in,Delta_out,N_Delta)
    n_0 = np.zeros((Deltas.size,3))
    n_plus = np.zeros((Deltas.size,3))
    n_minus = np.zeros((Deltas.size,3))
    for l,Delta in enumerate(Deltas):
        n0,Q3br,dn_dL,Minv = get_n0_Q3br_dndL_Minv(
            masses,
            Delta,
            (j_in,1),
            (j_out,1)
        )    
        dI = 2 * np.sqrt(np.abs(eval_at_zero(Q3br) / Minv))
        kvec= np.array([j_in-1,1-j_in-j_out,j_out])
        n_0[l]  = n0 
        n_plus[l]  = n0 + dn_dL * kvec * dI
        n_minus[l] = n0 - dn_dL * kvec * dI
    return (n_0,n_plus,n_minus)



import numpy as np
import matplotlib.pyplot as plt
from matplotlib import gridspec
import rebound as rb

def main():
    resonance_js = [3, 3, 3, 4, 4]
    sim0 = rb.Simulation("/Users/hadden/Papers/11_breaking_chains/02_initial_conditions/hd110067/rebound_x_ics/hd110067_K_500")
    data_file_string = "/Users/hadden/Papers/11_breaking_chains/03_data/run_mtot_{}_mtp_1_K_{}_continued.sa"
    masses = np.array([p.m for p in sim0.particles[1:]])
    T_integrate = 2 * np.pi * 5e3

    fig = plt.figure(figsize=(1.4 * 9, 1.4 * 9))
    outer_gs = gridspec.GridSpec(2, 2, figure=fig, wspace=0.3, hspace=0.3)

    top_axes = []
    bottom_axes = []

    for i in range(4):
        inner_gs = gridspec.GridSpecFromSubplotSpec(
            2, 1, subplot_spec=outer_gs[i], height_ratios=[1, 2], hspace=0.05
        )
        ax_top = fig.add_subplot(inner_gs[0])
        ax_bottom = fig.add_subplot(inner_gs[1])

        plt.setp(ax_top.get_xticklabels(), visible=False)

        top_axes.append(ax_top)
        bottom_axes.append(ax_bottom)

    # === Theory-based curves in the bottom panels ===
    for i in range(1, 5):
        j_in, j_out = resonance_js[i - 1:i + 1]
        mass_trio = masses[i - 1:i + 2]
        n0, n_plus, n_minus = three_body_resonance_plot_points(
            mass_trio, j_in, j_out, 0.002, 0.0125, 20
        )
        bottom_axes[i - 1].axhline((j_out - 1) / j_out, color='k')
        bottom_axes[i - 1].axvline(j_in / (j_in - 1), color='k')
        for nn in (n0, n_plus, n_minus):
            bottom_axes[i - 1].plot(nn.T[0] / nn.T[1], nn.T[2] / nn.T[1], color='gray')

    # === Loop over parameter combos and plot filtered data ===
    par_combos = [(f, K) for f in (0.03, 0.1) for K in (10, 100)]
    for par_combo in par_combos:
        data_file_name = data_file_string.format(*par_combo)
        sim = rb.Simulation(data_file_name)
        times, periods, mean_longitudes = sample_periods_and_mean_longitudes_at_Tsyn(sim, T_integrate)

        for i, (ax_top, ax_bottom) in enumerate(zip(top_axes, bottom_axes)):
            # bottom: period ratio space
            x_data = low_pass_filter(periods[i + 1] / periods[i], 0.01, 1)
            y_data = low_pass_filter(periods[i + 1] / periods[i + 2], 0.01, 1)
            ax_bottom.plot(x_data, y_data, label="{:.2f},{}".format(*par_combo))

            # top: three-body angle
            kvec = np.zeros(6)
            j_in = resonance_js[i]
            j_out = resonance_js[i + 1]
            kvec[i] = j_in - 1
            kvec[i + 1] = 1 - j_out - j_in
            kvec[i + 2] = j_out
            phi = np.mod(kvec @ mean_longitudes, 2 * np.pi) * 180 / np.pi
            physical_time = 0.1**1.5 * (times - times[0]) / (2 * np.pi)
            ax_top.plot(physical_time, phi, label="{:.2f},{}".format(*par_combo))

        bottom_axes[0].legend(title = "$(f,K)$")

    # === Axis labels and tick settings ===
    for ax in top_axes:
        ax.set_ylabel(r"$\phi$ [deg]")
        ax.set_ylim(0, 360)
        ax.set_yticks([0,  180, 360])

    for ax in bottom_axes:
        ax.set_xlabel(r"$P_{i+1} / P_i$")
        ax.set_ylabel(r"$P_{i+1} / P_{i+2}$")

    # Only show left y-axis labels for leftmost plots
    for ax in top_axes[1::2]:
        ax.set_yticklabels([])
    for ax in bottom_axes[1::2]:
        ax.set_yticklabels([])

    plt.show()

if __name__ == "__main__":
    main()

