import numpy as np
import rebound as rb
import argparse

def add_test_particles(sim, m_tot, m_tp):
    # Get star and planets (assuming star is particle 0)
    particles = sim.particles
    star = particles[0]
    planets = particles[1:]
    
    a_vals = np.array([p.a for p in planets])
    m_vals = np.array([p.m for p in planets])
    
    # Hill radius for each planet
    r_hill = a_vals * (m_vals / (3 * star.m))**(1/3)
    
    a_inner = a_vals[0] - 10 * r_hill[0]
    a_outer = a_vals[-1] + 10 * r_hill[-1]
    
    if a_inner <= 0:
        raise ValueError("Inner edge of test particle region is <= 0. Check units and planet spacing.")
    
    # Total number of particles to add
    n_particles = int(m_tot / m_tp)
    
    added = 0
    rng = np.random.default_rng()

    while added < n_particles:
        a = rng.uniform(a_inner, a_outer)
        
        # Check separation from each planet (in terms of Hill radii)
        too_close = False
        for a_p, r_H in zip(a_vals, r_hill):
            if abs(a - a_p) < 3 * r_H:
                too_close = True
                break
        
        if too_close:
            continue
        
        # Circular orbit in the plane of the planets (assume coplanar for now)
        inc = np.random.rayleigh(0.01)
        sim.add(m=m_tp, a=a,f='uniform',e=0.0, inc=inc,Omega='uniform', primary=star, hash=f"tp_{added}")
        added += 1

def add_test_particles_and_assign_radii(sim, m_tot, m_tp, a1_in_AU = 0.1, rho_in_gm_cc = 1):
        
        sim.N_active = sim.N
        add_test_particles(sim,m_tot,m_tp)
        
        # set radii
        rho_Earth_in_gm_cc = 5.5
        REarth_in_AU = 0.0000425876
        mEarth = 3e-6
        
        a1 = sim.particles[1].a
        for p in sim.particles:
            R_in_R_Earth = ((p.m / mEarth) * (rho_Earth_in_gm_cc / rho_in_gm_cc))**(1/3)
            R_in_AU = R_in_R_Earth * REarth_in_AU
            R_in_code_units = R_in_AU * a1  / a1_in_AU
            p.r = R_in_code_units
        sim.testparticle_type = 1
        sim.integrator='trace'
        sim.dt = sim.particles[1].P /25.
        sim.collision = "direct"
        sim.collision_resolve = "merge" # Built in function

MPLUTO_IN_SOLAR = 6.58e-9
def main():
    parser = argparse.ArgumentParser(description="Add test particles to a REBOUND simulation.")
    parser.add_argument("--input", required=True, help="Path to input REBOUND simulation")
    parser.add_argument("--output", required=True, help="Path to output REBOUND simulation")
    parser.add_argument("--mtp", type=float, required=True, help="Mass of test particle in Pluto masses")
    parser.add_argument("--mtot_frac", type=float, required=True, help="Total test particle mass as fraction of system mass")
    args = parser.parse_args()

    sim = rb.Simulation(args.input)
    sim.t = 0
    total_mass = sum(p.m for p in sim.particles[1:])  # skip star
    m_tot = args.mtot_frac * total_mass
    m_tp = args.mtp * MPLUTO_IN_SOLAR

    add_test_particles_and_assign_radii(sim, m_tot, m_tp)
    sim.save_to_file(args.output)

if __name__ == "__main__":
    main()