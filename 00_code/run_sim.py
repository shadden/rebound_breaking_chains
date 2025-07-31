import rebound
import argparse
import os
import numpy as np

def get_innermost_period(sim):
    sim.move_to_com()
    innermost = min(sim.particles[1:], key=lambda p: p.a)
    return innermost.P

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True, help="Path to input .bin file")
    args = parser.parse_args()

    input_path = args.input
    if not input_path.endswith(".bin"):
        raise ValueError("Expected a .bin REBOUND simulation file.")
    sa_path = input_path.replace(".bin", ".sa")
    if os.path.exists(sa_path):
        print(f"Resuming from archive: {sa_path}")
        sim = rebound.Simulation(sa_path)
    else:
        print(f"Starting new simulation from: {input_path}")
        sim = rebound.Simulation(input_path)
    Tfinal = 1e9 * 2 * np.pi
    print(sim.testparticle_type)
    print(sim.integrator) #='trace'
    print(f"Number of particles: {sim.N}")
    print(f"Number of active particles: {sim.N_active}")
    print(f"t={sim.t:.2f}") 
    print(f"dt={sim.dt:.2f}")
    sim.collision = "direct"
    sim.collision_resolve = "merge" # Built in function
    print(f"Save file: {sa_path}")
    sim.save_to_file(sa_path,walltime = 3.)
    print("Integrating...")
    sim.integrate(Tfinal)
if __name__ == "__main__":
    main()
