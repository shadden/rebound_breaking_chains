import numpy as np
import rebound as rb
import reboundx as rbx
from add_test_particles_to_simulation import add_test_particles,assign_radii,MPLUTO_IN_SOLAR
from generate_ics_with_reboundx import tau_alphas_to_tau_as

def set_up_sim(input_file,m_tot_frac = 0.1 ,m_tp_pluto = 1,tau_alpha = 2*np.pi*1e6,K=30,K_tp=1000,dt_frac = 1/30):

    sim = rb.Simulation(input_file)
    sim.t = 0
    m_tot_planets = np.sum([p.m for p in sim.particles[1:]])
    m_test_particles_total = m_tot_frac * m_tot_planets

    # Add test particles to simulation
    sim.N_active = sim.N
    add_test_particles(
        sim,
        m_test_particles_total,
        MPLUTO_IN_SOLAR * m_tp_pluto
    )
    
    # Assign radii to all particles
    assign_radii(sim)
    sim.testparticle_type = 1

    # Extract resonance info and set dissipation timescales
    active_particles = sim.particles[1:sim.N_active]
    test_particles = sim.particles[sim.N_active:]
    periods = np.array([p.P for p in active_particles])
    masses = np.array([p.m for p in active_particles])
    jvals = np.round(1 + 1 / (periods[1:] / periods[:-1] - 1)).astype(int)

    tau_alphas = np.ones(sim.N_active - 2) * np.inf
    tau_alpha_out = tau_alpha
    tau_e = tau_alpha_out / K
    tau_alphas[-1] = tau_alpha_out
    tau_as = tau_alphas_to_tau_as(tau_alphas, masses, [(j, 1) for j in jvals])

    # Add dissipative forces
    extras = rbx.Extras(sim)
    mod = extras.load_operator("modify_orbits_direct")
    extras.add_operator(mod)

    for tau_a, p in zip(tau_as, active_particles):
        p.params["tau_a"] = tau_a
        p.params["tau_e"] = -1 * tau_e
    for p in test_particles:
        p.params["tau_e"] = -tau_alpha / K_tp

    sim.collision = "direct"
    sim.collision_resolve="merge"
    sim.integrator='trace'
    sim.dt = sim.particles[1].P * dt_frac

    return sim
import argparse
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--infile", 
        required=True,
        help="Path to input file for initial conditions"
    )
    parser.add_argument(
        "--outfile", 
        required=True,
        help="Path to output file to save integration results"
    )
    parser.add_argument("--m_tot_frac", type=float, default=0.1,
                        help="Total test particle mass as a fraction of planetary mass (default: 0.1)")
    parser.add_argument("--m_tp_pluto", type=float, default=1.0,
                        help="Test particle mass in Pluto masses (default: 1)")
    parser.add_argument("--tau_alpha", type=float, default=2*np.pi*1e6,
                        help="Tau_alpha for outermost planet (default: 2pi*1e6)")
    parser.add_argument("--K", type=float, default=30,
                        help="Ratio tau_alpha / tau_e for outermost planet (default: 30)")
    parser.add_argument("--K_tp", type=float, default=1000,
                        help="Ratio tau_alpha / tau_e for test particles (default: 100)")
    parser.add_argument("--dt_frac", type=float, default=1/30,
                        help="Timestep as a fraction of the innermost planet's period (default: 1/30)")
    parser.add_argument("--write_interval",type=float,default=5.,
                        help="Time, in seconds, between saved snapshots in the output file (default: 5)")
    parser.add_argument("--orbits_to_integrate",type=float,default = 1e5,
                             help="Number of orbital periods to integrate for (default: 1e5)")
    args = parser.parse_args()
    sim = set_up_sim(
        args.infile,
        m_tot_frac=args.m_tot_frac,
        m_tp_pluto=args.m_tp_pluto,
        tau_alpha=args.tau_alpha,
        K=args.K,
        K_tp=args.K_tp,
        dt_frac=args.dt_frac
    )
    Tfin = args.orbits_to_integrate * sim.particles[1].P
    sim.save_to_file(args.outfile,walltime=args.write_interval)
    sim.integrate(Tfin)

if __name__=="__main__":
    main()