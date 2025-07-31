import numpy as np
from matplotlib import pyplot as plt
from celmech.disturbing_function import get_fg_coefficients
import argparse

def plot_data(data):
    fig,ax = plt.subplots(4,2,sharex = True,figsize =(12,6))
    ax = ax.T.reshape(-1)
    for a in ax:
        plt.sca(a)
        plt.tick_params(direction='in',size=8,labelsize=12)
        plt.tick_params(direction='in',which='minor',size=6)
    a=data['a'] * 0.1
    e=data['e']
    q=a*(1-e)
    Q=a*(1+e)
    Periods = 2*np.pi * a**(1.5)
    physical_time = 0.1**(1.5) * data['time']/(2*np.pi)/1e6

    for y,y1,y2 in zip(a.T,q.T,Q.T):
        msk = y>0
        l,=ax[0].plot(physical_time[msk],y[msk],lw=2)
        ax[0].fill_between(physical_time[msk],y1[msk],y2[msk],color=l.get_color(),alpha=0.4)
    ax[0].set_ylabel(r"$a,q,Q$ [AU]",fontsize=15)
    ax[0].set_yscale('log')

    pomega = data['omega']+data['Omega']
    z = e*np.exp(1j*pomega)
    l = data['l']
    P = a**(1.5)
    Npl = a.shape[1]
    res_kvecs = np.zeros((Npl-1,Npl))
    for n,Pin,Pout,zin,zout,lin,lout in zip(range(Npl),P.T,P.T[1:],z.T,z.T[1:],l.T,l.T[1:]):
        msk = ~np.logical_or(np.isnan(Pin),np.isnan(Pout))
        j = int(np.round(1 + 1/(Pout[0]/Pin[0] - 1)))
        Delta = ((j-1)*Pout/Pin/j - 1)
        ax[1].plot(physical_time[msk],Delta[msk])
        f,g = get_fg_coefficients(j,1)
        pmg = np.angle(f*zin+g*zout)
        phi_res = np.mod(j*lout+(1-j)*lin - pmg,2*np.pi)-np.pi
        res_kvecs[n,n+1] = j
        res_kvecs[n,n] = 1-j
        ax[4].plot(physical_time[msk],phi_res[msk]/np.pi,'.')
    ax[4].set_ylim(-1,1)

    
    ax[1].set_ylabel(r"$\Delta$",fontsize=15)
    ax[4].set_ylabel(r"$\phi_\mathrm{2br}/\pi$",fontsize=15)

    for msk,ecc in zip((a>0).T,e.T):
        ax[2].plot(physical_time[msk],ecc[msk])
    ax[2].set_ylabel(r"$e$",fontsize=15)

    abs_u = np.abs(data['Te'].T @ data['x'].T)
    rt_L1 = np.sqrt(data['m'][-1,0] * np.sqrt(data['a'][-1,0]))
    for u in abs_u:
        ax[3].plot(physical_time,u/rt_L1)
    ax[3].set_ylabel(r"$|u|/\sqrt{\Lambda_1}$",fontsize=15)


    k_3br = res_kvecs[1:] - res_kvecs[:-1]
    phi_3brs = np.mod(k_3br @ data['l'].T,2*np.pi)-np.pi
    for phi_3br in phi_3brs:
        ax[5].plot(physical_time,phi_3br/np.pi,'.')
    ax[5].set_ylabel(r"$\phi_\mathrm{3br}/\pi$",fontsize=15)
    ax[5].set_ylim(-1,1)

    for m_p in data['m'].T:
        ax[6].plot(physical_time,(m_p-m_p[0])/3e-6)
    ax[6].set_ylabel(r"$\Delta m_p$ [$M_\oplus$]",fontsize=15)

    ax[7].plot(physical_time,data['Ntp'],'k-')
    ax[7].set_ylabel(r"$N_\mathrm{tp}$",fontsize=15)

    ax[7].set_yscale('log')
    ax[3].set_xlabel('Time [Myr]',fontsize=15)
    ax[7].set_xlabel('Time [Myr]',fontsize=15)
    return fig,ax

def plot_data_simple(data):
    fig,ax = plt.subplots(5,sharex = True,figsize =(12,6))
    ax = ax.T.reshape(-1)
    for a in ax:
        plt.sca(a)
        plt.tick_params(direction='in',size=8,labelsize=12)
        plt.tick_params(direction='in',which='minor',size=6)
    a=data['a'] * 0.1
    e=data['e']
    q=a*(1-e)
    Q=a*(1+e)
    Periods = 2*np.pi * a**(1.5)
    physical_time = 0.1**(1.5) * data['time']/(2*np.pi)/1e6
    ### a-q-Q plot
    for y,y1,y2 in zip(a.T,q.T,Q.T):
        msk = y>0
        l,=ax[0].plot(physical_time[msk],y[msk],lw=2)
        ax[0].fill_between(physical_time[msk],y1[msk],y2[msk],color=l.get_color(),alpha=0.4)
    ax[0].set_ylabel(r"$a,q,Q$ [AU]",fontsize=15)
    ax[0].set_yscale('log')
    ax[0].set_ylim(np.min(q[0])*0.9,np.max(q[0])*1.)

    ### Delta plot and 2-body resonant angle plot
    pomega = data['omega']+data['Omega']
    z = e*np.exp(1j*pomega)
    l = data['l']
    P = a**(1.5)
    Npl = a.shape[1]
    res_kvecs = np.zeros((Npl-1,Npl))
    for n,Pin,Pout,zin,zout,lin,lout in zip(range(Npl),P.T,P.T[1:],z.T,z.T[1:],l.T,l.T[1:]):
        msk = ~np.logical_or(np.isnan(Pin),np.isnan(Pout))
        j = int(np.round(1 + 1/(Pout[0]/Pin[0] - 1)))
        Delta = ((j-1)*Pout/Pin/j - 1)
        ax[1].plot(physical_time[msk],Delta[msk])
        f,g = get_fg_coefficients(j,1)
        pmg = np.angle(f*zin+g*zout)
        phi_res = np.mod(j*lout+(1-j)*lin - pmg,2*np.pi)-np.pi
        res_kvecs[n,n+1] = j
        res_kvecs[n,n] = 1-j
        ax[3].plot(physical_time[msk],phi_res[msk]/np.pi,'.')
    ax[3].set_ylim(-1,1)

    
    ax[1].set_ylabel(r"$\Delta$",fontsize=15)
    ax[3].set_ylabel(r"$\phi_\mathrm{2br}/\pi$",fontsize=15)

    # eccentricity plot
    for msk,ecc in zip((a>0).T,e.T):
        ax[2].plot(physical_time[msk],ecc[msk])
    ax[2].set_ylabel(r"$e$",fontsize=15)

    # three-body resonant angle plot
    k_3br = res_kvecs[1:] - res_kvecs[:-1]
    phi_3brs = np.mod(k_3br @ data['l'].T,2*np.pi)-np.pi
    for phi_3br in phi_3brs:
        ax[4].plot(physical_time,phi_3br/np.pi,'.')
    ax[4].set_ylabel(r"$\phi_\mathrm{3br}/\pi$",fontsize=15)
    ax[4].set_ylim(-1,1)
    ax[4].set_xlabel('Time [Myr]',fontsize=15)
    
    return fig,ax

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True, help="Path to input .npz file")
    parser.add_argument("--output", required=False, help="Path to output plot file.")
    args = parser.parse_args()
    input_path = args.input
    
    if not input_path.endswith(".npz"):
        raise ValueError("Expected a .npz numpy data file.")
    data = np.load(input_path)
    if 'Te' in data.keys():
        fig,ax = plot_data(data)
        if args.output is not None:
            output_path = args.output
            fig.savefig(output_path)
        else:
            plt.show()
    else:
        print("simulation '{}' was unstable".format(input_path))
        fig,ax = plot_data_simple(data)
        if args.output is not None:
            output_path = args.output
            fig.savefig(output_path)
        else:
            plt.show()



if __name__=="__main__":
    main()