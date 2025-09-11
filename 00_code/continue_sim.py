import rebound
import argparse
import os
import numpy as np

def scrub_simulation(sim):
    star = sim.particles[0]
    sim_new = rebound.Simulation()
    sim_new.add(star.copy())
    for p in sim.particles[1:]:
        orbit = p.orbit(primary = star)
        if orbit.a > 0:
            sim_new.add(p.copy())
    sim_new.move_to_com()
    
    sim_new.N_active = sim.N_active
    sim_new.t = sim.t
    sim_new.integrator = sim.integrator
    sim_new.dt = sim.dt
    
    return sim_new

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True, help="Path to input rebound save file.")
    parser.add_argument("--output", required=False, help="Path to output rebound save file.")
    parser.add_argument("--scrub", action='store_true', help="Remove unbound particles and re-center simulation.")


    args = parser.parse_args()
    input_path = args.input
    output_path = args.output if args.output else args.input
    print(f"Starting new simulation from: {input_path}")
    sim = rebound.Simulation(input_path)
    if args.scrub:
        sim = scrub_simulation(sim)
    Tfinal = 1e9 * 2 * np.pi
    print(sim.testparticle_type)
    print(sim.integrator) #='trace'
    print(f"Number of particles: {sim.N}")
    print(f"Number of active particles: {sim.N_active}")
    print(f"Integrator: {sim.integrator}")
    print(f"t={sim.t:.2f}")
    print(f"dt={sim.dt:.2f}")
    sim.collision = "direct"
    sim.collision_resolve = "merge" # Built in function
    print(f"Save file: {output_path}")
    sim.save_to_file(output_path,walltime = 30.)
    print("Integrating...")
    sim.integrate(Tfinal)
if __name__ == "__main__":
    main()
