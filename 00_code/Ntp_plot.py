import glob, re
import numpy as np
import matplotlib.pyplot as plt

fig, ax = plt.subplots()

# Store handles for legends
mtp_handles = {}
K_handles = {}

for fi in glob.glob("/Users/hadden/Papers/11_breaking_chains/03_data/*plot_data*"):
    data = np.load(fi)
    f = re.search(r"mtot_(\d+\.\d+)", fi).groups()[0]
    mtp = re.search(r"mtp_(\d+)", fi).groups()[0]
    K = re.search(r"K_(\d+)", fi).groups()[0]

    color = 'r' if mtp == '3' else 'k'
    ls = '-' if K == '100' else '--'

    # plot actual data
    ax.plot(
        0.1**1.5 * data['time'] / (2e6*np.pi),
        data['Ntp'],
        color=color,
        ls=ls
    )

    # store representative handles
    if mtp not in mtp_handles:
        mtp_handles[mtp] = ax.plot([], [], color=color, ls='-', label=r"${}~M_\mathrm{{Pluto}}$".format(mtp))[0]
    if K not in K_handles:
        K_handles[K] = ax.plot([], [], color='k', ls=ls, label=f"{K}")[0]

ax.set_xscale('log')
ax.set_yscale('log')
ax.set_xlabel('Time [Myr]',fontsize = 16)
ax.set_ylabel('$N_s$',fontsize = 16)
ax.set_ylim(ymin=0.9)
plt.tick_params(direction='in',size = 8,labelsize=12)
plt.tick_params(direction='in',size = 6,which='minor')
# Create legends
leg1 = ax.legend(handles=mtp_handles.values(), title="$m_s$", loc='upper right')
ax.add_artist(leg1)  # add first legend manually
ax.legend(handles=K_handles.values(), title="$K$", loc='lower left')
plt.tight_layout()
plt.savefig("/Users/hadden/Papers/11_breaking_chains/04_figures/Ntp_vs_time_plot.pdf")
