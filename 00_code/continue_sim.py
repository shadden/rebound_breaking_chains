import rebound
import argparse
import os
import numpy as np

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True, help="Path to input rebound save file.")
    parser.add_argument("--output", required=False, help="Path to output rebound save file.")

    args = parser.parse_args()
    input_path = args.input
    output_path = args.output if args.output else args.input
    print(f"Starting new simulation from: {input_path}")
    sim = rebound.Simulation(input_path)
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
