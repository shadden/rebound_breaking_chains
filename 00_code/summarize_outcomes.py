from matplotlib import pyplot as plt
from celmech.disturbing_function import get_fg_coefficients
import argparse
import os
import re
import numpy as np

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
    args = parser.parse_args()
    system_name = args.system_name
    input_dir = "/Users/hadden/Papers/11_breaking_chains/03_data/" + system_name + "/outfiles/"

    dk2_values = extract_dk2_values(input_dir)
    mtps = [1.0,3.0]
    mtots = [0.03,0.1] 
    
    for mtot in mtots:
        for mtp in mtps:
            for dk2 in dk2_values:
                file_name = input_dir + f"{system_name}_dK2_{dk2:.5f}_mtot_frac_{mtot:.2f}_mtp_{mtp:.2f}mpluto.npz"
                data = np.load(file_name)
                times = data['time']
                a = data['a']
                periods = np.transpose(a**(1.5))

                





if __name__=="__main__":
    main()