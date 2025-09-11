import numpy as np
import rebound as rb
import argparse
import os
import celmech as cm
from celmech.secular import LaplaceLagrangeSystem
def auto_add_resonances_to_LaplaceLagrangeSystem(llsys,threshold = 0.05):
    resonances_to_add = dict()
    for i in range(1,llsys.N):
        p_in = llsys.particles[i]
        for j in range(i+1,llsys.N):
            p_out = llsys.particles[j]
            Pratio = p_out.P/p_in.P
            jres = int(np.round(1 + 1/(Pratio - 1)))
            Delta = (jres - 1) * Pratio / jres - 1 
            if np.abs(Delta) < threshold:
                resonances_to_add[(i,j)] = jres
    llsys.add_first_order_resonance_terms(resonances_to_add)         
def process_sa(sa):
    """
    Fill numpy arrays of orbital parameters of active particles in a simulation archive.
    """
    sim0 = sa[0]
    sim_final = sa[-1]
    fixed_planet_number_Q = sim_final.N_active == sim0.N_active
    Nsim = len(sa)
    Npl = sim0.N_active - 1 if sim0.N_active > 0 else sim0.N-1
    time = np.zeros(Nsim)
    m = np.zeros((Nsim,Npl))
    a = np.zeros((Nsim,Npl))
    e = np.zeros((Nsim,Npl))
    inc = np.zeros((Nsim,Npl))
    l = np.zeros((Nsim,Npl))
    omega = np.zeros((Nsim,Npl))
    Omega = np.zeros((Nsim,Npl))
    Ntp  = np.zeros(Nsim)
    x = np.zeros((Nsim,Npl),dtype = np.complex128)
    for i,sim in enumerate(sa):
        time[i] = sim.t
        star = sim.particles[0]
        # number of surviving bound test particles
        Ntp[i] = np.sum([p.orbit(primary=sim.particles[0]).a>0 for p in sim.particles[Npl+1:]])
        if fixed_planet_number_Q:
            sim_c = rb.Simulation()
            sim_c.add(star.copy())
        for j,p in enumerate(sim.particles[1:Npl+1]):
            orbit = p.orbit(primary=star)
            m[i,j] = p.m
            a[i,j] = orbit.a
            e[i,j] = orbit.e
            inc[i,j] = orbit.inc
            l[i,j] = orbit.l
            omega[i,j] = orbit.omega
            Omega[i,j] = orbit.Omega
            if fixed_planet_number_Q:
                sim_c.add(p.copy())
        if False and fixed_planet_number_Q:
            sim_c.move_to_com()
            pvars = cm.Poincare.from_Simulation(sim_c)
            x[i] = [p.x for p in pvars.particles[1:]]
    results = {
        "m":m,
        "a":a,
        "e":e, 
        "inc":inc,
        "l":l, 
        "omega":omega, 
        "Omega":Omega, 
        "Ntp":Ntp,
        "time":time
    }
    # if the number of planets is preserved, compute secular info
    if False and fixed_planet_number_Q:
        llsys = LaplaceLagrangeSystem.from_Poincare(pvars)
        auto_add_resonances_to_LaplaceLagrangeSystem(llsys)
        Te, De = llsys.diagonalize_eccentricity()
        results['x'] = x
        results['Te'] = Te
        results['De'] = De
    return results
        

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True, help="Path to input .bin file")
    parser.add_argument("--output", required=True, help="Path to output .npz file")
    args = parser.parse_args()
    sa_path = args.input
    if os.path.exists(sa_path):
        print(f"Reading from archive: {sa_path}")
    else:
        print(f"File '{sa_path}' does not exist.")
    sa = rb.Simulationarchive(sa_path)
    results = process_sa(sa)
    np.savez_compressed(args.output,**results)

if __name__=="__main__":
    main()