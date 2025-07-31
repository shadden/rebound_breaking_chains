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
    input_dir = "/Users/hadden/Papers/11_breaking_chains/02_initial_conditions/" + system_name + "/mtot_frac_0.03_mtp_1.00mpluto/"
    data_dir = "/Users/hadden/Papers/11_breaking_chains/03_data/" + system_name + "/outfiles/"

    dk2_values = extract_dk2_values(input_dir)
    dk2_values_done = extract_dk2_values(data_dir)
    Delta_12 = np.zeros(len(dk2_values))
    eccs = []
    ### Plot full track 
    for i,dk2 in enumerate(dk2_values):
        finame = input_dir+f"{system_name}_dK2_{dk2:.5f}_mtot_frac_0.03_mtp_1.00mpluto.bin"
        sim = rb.Simulation(finame)
        ps = sim.particles[:sim.N_active]
        P1,P2 = ps[1].P,ps[2].P
        j = int(np.round(1 + 1 / (P2/P1-1)))
        Delta_12[i] = (j-1)*P2 / P1 / j - 1
        eccs.append([p.e for p in ps[1:]])
    eccs = np.transpose(eccs)
    for l,e_i in enumerate(eccs):
        plt.plot(Delta_12,e_i,label=r"$e_{}$".format(l+1))

    ### Plot actually simulated systems 
    colors = plt.rcParams['axes.prop_cycle'].by_key()['color']
    color_iter = iter(colors)  # Reset to start

    eccs = []
    Delta_12 = np.zeros(len(dk2_values_done))
    for i,dk2 in enumerate(dk2_values_done):
        finame = input_dir+f"{system_name}_dK2_{dk2:.5f}_mtot_frac_0.03_mtp_1.00mpluto.bin"
        sim = rb.Simulation(finame)
        ps = sim.particles[:sim.N_active]
        P1,P2 = ps[1].P,ps[2].P
        j = int(np.round(1 + 1 / (P2/P1-1)))
        Delta_12[i] = (j-1)*P2 / P1 / j - 1
        eccs.append([p.e for p in ps[1:]])
    eccs = np.transpose(eccs)
    colors = plt.rcParams['axes.prop_cycle'].by_key()['color']
    for e_i, color in zip(eccs, colors):
        plt.plot(Delta_12, e_i, 's', ms=8, color=color)

    plt.legend()
    plt.xscale('log')
    plt.xlabel(r"$\frac{{{}}}{{{}}} \frac{{P_2}}{{P_1}} - 1$".format(j-1,j),fontsize=16)
    plt.ylabel(r"Eccentricity",fontsize=16)
    plt.tick_params(direction='in',size=8,labelsize=12)
    plt.tick_params(direction='in',size=6,which='minor')
    plt.tight_layout()
    outfile = args.outfile
    if args.outfile:
        plt.savefig(outfile)
    else:
        plt.show()
        

if __name__=="__main__":
    main()




