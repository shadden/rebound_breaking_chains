import rebound as rb
import numpy as np
from process_sim import process_sa
from matplotlib import pyplot as plt
from celmech.disturbing_function import get_fg_coefficients
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec

def plot_resonances(data,title = r"$m_{{s}}=M_\mathrm{{Pluto}}~;~f=10\%~;~ K=100$"):
    # assumes: get_fg_coefficients is available in scope
    a = data['a'] * 0.1
    e = data['e']
    q = a * (1 - e)
    Q = a * (1 + e)

    physical_time = 0.1**1.5 * data['time'] / (2*np.pi) / 1e6  # Myr

    pomega = data['omega'] + data['Omega']
    z = e * np.exp(1j * pomega)
    l = data['l']
    P = a**1.5

    Npl = a.shape[1]
    n3 = max(Npl - 2, 0)  # number of independent 3-body angles

    # build figure layout: 3 big rows (a/q/Q, Δ, e) + n3 short rows (each phi_3br)
    fig = plt.figure(figsize=(8,10))
    gs = GridSpec(3 + n3, 1, height_ratios=[2, 2, 2] + [1] * n3, hspace=0.15)
    axes = [fig.add_subplot(gs[i]) for i in range(3 + n3)]
    
    # styling
    for ax in axes:
        ax.tick_params(direction='in', size=8, labelsize=10)
        ax.tick_params(direction='in', which='minor', size=6)

    # ---- Panel 1: a, q, Q (log y) ----
    for y, y1, y2 in zip(a.T, q.T, Q.T):
        msk = y > 0
        ln, = axes[0].plot(physical_time[msk], y[msk], lw=2)
        axes[0].fill_between(physical_time[msk], y1[msk], y2[msk],
                             color=ln.get_color(), alpha=0.4)
    axes[0].set_ylabel(r"$a,q,Q$ [AU]", fontsize=15)
    axes[0].set_yscale('log')
    # axes[0].set_ylim(np.min(q[0]) * 0.9, np.max(Q[0]) * 1.1)
    axes[0].set_ylim(np.min(q[0]) * 0.7, 0.5)
    axes[0].set_title(title,fontsize=15)

    # ---- Panel 2: eccentricities ----
    for msk, ecc in zip((a > 0).T, e.T):
        axes[1].plot(physical_time[msk], ecc[msk])
    axes[1].set_ylabel(r"$e$", fontsize=15)
    axes[1].set_ylim(1e-3,0.35)
    axes[1].set_yscale("log")  

    # ---- Panel 3: Δ for consecutive pairs ----
    res_kvecs = np.zeros((Npl - 1, Npl))
    for n, Pin, Pout, zin, zout, lin, lout in zip(
            range(Npl - 1), P.T, P.T[1:], z.T, z.T[1:], l.T, l.T[1:]):
        msk = ~np.logical_or(np.isnan(Pin), np.isnan(Pout))
        # estimate j for near-(j:j-1)
        j = int(np.round(1 + 1/(Pout[0]/Pin[0] - 1)))
        Delta = ((j - 1) * Pout / (j * Pin) - 1)
        axes[2].plot(physical_time[msk], Delta[msk])

        # store coefficients for 3-body construction; no 2-body angle plotted
        res_kvecs[n, n+1] = j
        res_kvecs[n, n] = 1 - j
    axes[2].set_ylabel(r"$\Delta$", fontsize=15)


    # ---- Panels 4..: each 3-body angle in its own small panel ----
    if n3 > 0:
        k_3br = res_kvecs[1:] - res_kvecs[:-1]          # shape: (Npl-2, Npl)
        phi_3brs = np.mod(k_3br @ l.T, 2*np.pi) - np.pi # (Npl-2, Nt)

        for i, phi_3br in enumerate(phi_3brs):
            ax = axes[3 + i]
            ax.plot(physical_time, (phi_3br / np.pi), 'k.', ms=1)
            ax.set_ylim(-1, 1)
            ax.set_ylabel(rf"$\phi_{{{i+1}}}/\pi$", fontsize=12)

    # ---- shared x on bottom only ----
    axes[-1].set_xlabel('Time [Myr]', fontsize=15)
    for ax in axes:
        ax.set_xscale('log')
        ax.set_xlim(0.9e-3,30)
     # hide x tick labels for all but the bottom axis
    for ax in axes[:-1]:
        ax.tick_params(labelbottom=False)

    return fig, axes

def main():
    import argparse
    import os
    parser = argparse.ArgumentParser()
    parser.add_argument("--reload",action="store_true")
    parser.add_argument("--save",required=False,help="File to save plot.")

    args = parser.parse_args()
    save_file_name = "/Users/hadden/Papers/11_breaking_chains/03_data/figure_1_data.npz"
    if args.reload or not os.path.exists(save_file_name):
        archive_file = "/Users/hadden/Papers/11_breaking_chains/03_data/migration_hd110067/run_mtot_0.1_mtp_1_K_100_continued.sa"
        sa = rb.Simulationarchive(archive_file)
        data = process_sa(sa)
        np.savez_compressed(save_file_name,**data)
    else:
        data = np.load(save_file_name)
    print(list(data.keys()))
    fig,ax = plot_resonances(data)
    if args.save:
        fig.savefig(args.save)
    else:
        fig.show()
    # fig,ax = plt.subplots(5,sharex = True,figsize =(12,6))
    # ax = ax.T.reshape(-1)
    # for a in ax:
    #     plt.sca(a)
    #     plt.tick_params(direction='in',size=8,labelsize=12)
    #     plt.tick_params(direction='in',which='minor',size=6)
    # a=data['a'] * 0.1
    # e=data['e']
    # q=a*(1-e)
    # Q=a*(1+e)
    # Periods = 2*np.pi * a**(1.5)
    # physical_time = 0.1**(1.5) * data['time']/(2*np.pi)/1e6
    # ### a-q-Q plot
    # for y,y1,y2 in zip(a.T,q.T,Q.T):
    #     msk = y>0
    #     l,=ax[0].plot(physical_time[msk],y[msk],lw=2)
    #     ax[0].fill_between(physical_time[msk],y1[msk],y2[msk],color=l.get_color(),alpha=0.4)
    # ax[0].set_ylabel(r"$a,q,Q$ [AU]",fontsize=15)
    # ax[0].set_yscale('log')
    # ax[0].set_ylim(np.min(q[0])*0.9,np.max(q[0])*1.)

    # ### Delta plot and 2-body resonant angle plot
    # pomega = data['omega']+data['Omega']
    # z = e*np.exp(1j*pomega)
    # l = data['l']
    # P = a**(1.5)
    # Npl = a.shape[1]
    # res_kvecs = np.zeros((Npl-1,Npl))
    # for n,Pin,Pout,zin,zout,lin,lout in zip(range(Npl),P.T,P.T[1:],z.T,z.T[1:],l.T,l.T[1:]):
    #     msk = ~np.logical_or(np.isnan(Pin),np.isnan(Pout))
    #     j = int(np.round(1 + 1/(Pout[0]/Pin[0] - 1)))
    #     Delta = ((j-1)*Pout/Pin/j - 1)
    #     ax[1].plot(physical_time[msk],Delta[msk])
    #     f,g = get_fg_coefficients(j,1)
    #     pmg = np.angle(f*zin+g*zout)
    #     phi_res = np.mod(j*lout+(1-j)*lin - pmg,2*np.pi)-np.pi
    #     res_kvecs[n,n+1] = j
    #     res_kvecs[n,n] = 1-j
    #     ax[3].plot(physical_time[msk],phi_res[msk]/np.pi,'.')
    # ax[3].set_ylim(-1,1)

    
    # ax[1].set_ylabel(r"$\Delta$",fontsize=15)
    # ax[3].set_ylabel(r"$\phi_\mathrm{2br}/\pi$",fontsize=15)

    # # eccentricity plot
    # for msk,ecc in zip((a>0).T,e.T):
    #     ax[2].plot(physical_time[msk],ecc[msk])
    # ax[2].set_ylabel(r"$e$",fontsize=15)

    # # three-body resonant angle plot
    # k_3br = res_kvecs[1:] - res_kvecs[:-1]
    # phi_3brs = np.mod(k_3br @ data['l'].T,2*np.pi)-np.pi
    # for phi_3br in phi_3brs[::-1]:
    #     ax[4].plot(physical_time,phi_3br/np.pi,'.',ms=1)
    # ax[4].set_ylabel(r"$\phi_\mathrm{3br}/\pi$",fontsize=15)
    # ax[4].set_ylim(-1,1)
    # ax[4].set_xlabel('Time [Myr]',fontsize=15)
    # ax[4].set_xscale('log')
    # plt.show()
if __name__=="__main__":
    main()