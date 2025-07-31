import argparse
import numpy as np
import rebound as rb
import reboundx as rbx

def tau_alphas_to_tau_as(tau_alpha, masses, resonances):
    Npl = len(masses)
    sma = np.ones(Npl)
    mtrx = -1 * np.eye(Npl)
    mtrx[:, 0] = 1
    for i, jk in enumerate(resonances):
        j, k = jk
        sma[i + 1] = sma[i] * (j / (j - k)) ** (2 / 3)
    mtrx[0] = masses / sma
    gamma_alphas = np.concatenate(([0], 1 / np.array(tau_alpha)))
    gamma_a = np.linalg.inv(mtrx) @ gamma_alphas
    return 1 / gamma_a

def main():
    parser = argparse.ArgumentParser(description="Relax a resonant chain with dissipation")
    parser.add_argument("--input", required=True, help="Path to input REBOUND simulation file")
    parser.add_argument("--output", required=True, help="Path to save the output simulation archive")
    parser.add_argument("--K", type=float, default=10, help="K parameter: tau_alpha_out / tau_e")

    args = parser.parse_args()

    # Load and clone active particles
    sim = rb.Simulation(args.input)
    sim_c = rb.Simulation()
    N_active = sim.N_active if sim.N_active > 0 else sim.N
    for p in sim.particles[:N_active]:
        sim_c.add(p.copy())

    # Extract resonance info and set dissipation timescales
    periods = np.array([p.P for p in sim_c.particles[1:]])
    masses = np.array([p.m for p in sim_c.particles[1:]])
    jvals = np.round(1 + 1 / (periods[1:] / periods[:-1] - 1)).astype(int)

    tau_alphas = np.ones(sim_c.N - 2) * np.inf
    tau_alpha_out = 2 * np.pi * 1e6
    tau_e = tau_alpha_out / args.K
    tau_alphas[-1] = tau_alpha_out
    tau_as = tau_alphas_to_tau_as(tau_alphas, masses, [(j, 1) for j in jvals])

    # Add dissipative forces
    extras = rbx.Extras(sim_c)
    mod = extras.load_operator("modify_orbits_direct")
    extras.add_operator(mod)

    for tau_a, p in zip(tau_as, sim_c.particles[1:]):
        p.params["tau_a"] = tau_a
        p.params["tau_e"] = -1 * tau_e

    # Integrate with output
    Tfin = tau_alpha_out
    sim_c.integrator = "whfast"
    sim_c.dt = sim_c.particles[1].P / 30.
    sim_c.integrate(Tfin)
    sim_c.save_to_file(args.output)

if __name__ == "__main__":
    main()
