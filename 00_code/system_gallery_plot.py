import numpy as np
from matplotlib import pyplot as plt
import re
import os
import argparse
import rebound as rb

def extract_dk2_values(directory):
    pattern = re.compile(r'dK2_([-+]?[0-9]*\.?[0-9]+(?:[eE][-+]?[0-9]+)?)')
    dk2_values = set()

    for filename in os.listdir(directory):
        match = pattern.search(filename)
        if match:
            dk2_values.add(float(match.group(1)))

    
    return sorted(dk2_values)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--system_name", required=True, help="Path to input directory with '.npz' files")
    parser.add_argument("--outfile", required=False, help="Path to save file for plot")
    args = parser.parse_args()
    system_name = args.system_name
    data_dir = f"/Users/hadden/Papers/11_breaking_chains/03_data/{system_name}/outfiles/"
    dk2_values = extract_dk2_values(data_dir)
    assert len(dk2_values)>0, "no files found in directory {}".format(data_dir)
    fig,axes = plt.subplots(4,4,sharex=True,sharey = True,figsize = (15,5))
    par_combos = [(mtp,mfrac) for mtp in [1,3] for mfrac in [0.03,0.1]]
    for row,dk2_val in zip(axes,dk2_values):
        for ax,(m_small,m_frac) in zip(row,par_combos):
            data = np.load(data_dir + f"{system_name}_dK2_{dk2_val:.5f}_mtot_frac_{m_frac:.2f}_mtp_{m_small:.2f}mpluto.npz")
            phys_time = 0.1**(1.5) * data['time']/(2e6*np.pi)
            for aa,ee in zip(0.1*data['a'].T,data['e'].T):
                msk = aa>0
                q = aa*(1-ee)
                Q = aa*(1+ee)
                l,=ax.plot(phys_time[msk],aa[msk])
                ax.fill_between(phys_time[msk],q[msk],Q[msk],color=l.get_color(),alpha=0.5)
    axes[-1,-1].set_xscale('log')
    #axes[-1,-1].set_yscale('log')
    
    axes[-1,-1].set_ylim(0.1*0.9*np.min(data['a'][0]),0.1*np.max(data['a'][0])*1.1)
    axes[-1,-1].set_xlim(0.01,41)
    for ax,(m_small,m_frac) in zip(axes[0],par_combos):
        ax.set_title(f"$m_s = {m_small:.2f}; f = {m_frac:.2f}$")
    for ax,(m_small,m_frac) in zip(axes[-1],par_combos):
        ax.set_xlabel("Time [Myr]")
    for ax in axes.T[0]:
        ax.set_ylabel("$a$ [AU]")

    
    dk2_values = extract_dk2_values(data_dir)
    outfile = args.outfile
    if args.outfile:
        fig.savefig(outfile)
    else:
        plt.show()
        

if __name__=="__main__":
    main()




