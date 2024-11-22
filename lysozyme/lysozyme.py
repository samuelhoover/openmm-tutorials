# pyright: basic


import numpy as np
import matplotlib.pyplot as plt
from openmm.app import *
from openmm import *
from openmm.unit import *
from sys import stdout


def load_pdb(pdb):
    return PDBFile(pdb)


def get_forcefield(protein_ff, water_ff):
    return ForceField(protein_ff, water_ff)


def clean_pdb(pdbfile, forcefield):
    modeller = Modeller(pdbfile.topology, pdbfile.positions)
    modeller.deleteWater()
    residues = modeller.addHydrogens(forcefield)

    return modeller


def solvate(modeller, forcefield):
    modeller.addSolvent(forcefield, padding=1.0 * nanometer)

    return modeller


def simulation_setup(modeller, forcefield):
    system = forcefield.createSystem(
        modeller.topology,
        nonbondedMethod=PME,
        nonbondedCutoff=1.0 * nanometer,
        constraints=HBonds,
    )
    integrator = LangevinMiddleIntegrator(
        300 * kelvin, 1 / picosecond, 0.004 * picoseconds
    )
    simulation = Simulation(modeller.topology, system, integrator)
    simulation.context.setPositions(modeller.positions)

    return simulation, system


def create_system(pdb, protein_ff, water_ff):
    pdbfile = load_pdb(pdb)
    forcefield = get_forcefield(protein_ff, water_ff)
    simulation, system = simulation_setup(
        solvate(clean_pdb(pdbfile, forcefield), forcefield), forcefield
    )

    return simulation, system


def minimization(simulation):
    print("\nMinimizing energy")
    simulation.minimizeEnergy()

    return simulation


def setup_reporting(simulation, pdb_out, data_out):
    simulation.reporters.append(PDBReporter(pdb_out, 1000))
    simulation.reporters.append(
        StateDataReporter(
            stdout, 1000, step=True, potentialEnergy=True, temperature=True, volume=True
        )
    )
    simulation.reporters.append(
        StateDataReporter(
            data_out,
            100,
            step=True,
            potentialEnergy=True,
            temperature=True,
            volume=True,
        )
    )


def NVT(simulation):
    print("\nRunning NVT")
    simulation.step(10000)

    return simulation


def NPT(simulation, system):
    system.addForce(MonteCarloBarostat(1 * bar, 300 * kelvin))
    simulation.context.reinitialize(preserveState=True)

    print("\nRunning NPT")
    simulation.step(10000)


def run(pdb, protein_ff, water_ff, pdb_out, data_out):
    # create system
    simulation, system = create_system(pdb, protein_ff, water_ff)

    # energy minimization
    em_result = minimization(simulation)

    # setup reporting
    setup_reporting(em_result, pdb_out, data_out)

    # NVT equilibration
    nvt_result = NVT(em_result)

    # NPT production
    NPT(nvt_result, system)


def analyze_results(data_out):
    data = np.loadtxt(data_out, delimiter=",")

    step = data[:, 0]
    potential_energy = data[:, 1]
    temperature = data[:, 2]
    volume = data[:, 3]

    fig, ax = plt.subplots(figsize=(5.5 * 3, 4), ncols=3)

    for i, (y, ylabel) in enumerate(
        zip(
            [potential_energy / 1000, temperature, volume],
            ["Pot. energy [1000 kJ/mol]", "Temperature [K]", "Volume [nm$^{3}$]"],
        )
    ):
        ax[i].plot(step, y)
        ax[i].set_xlabel("Step")
        ax[i].set_ylabel(ylabel)

    plt.tight_layout()
    plt.show()
    fig.savefig("analysis.pdf")


def main(pdb, protein_ff, water_ff, pdb_out, data_out, run_sim=True, analysis=True):
    if run_sim:
        run(pdb, protein_ff, water_ff, pdb_out, data_out)

    if analysis:
        analyze_results(data_out)


if __name__ == "__main__":
    # inputs
    pdb = "1AKI.pdb"
    protein_ff = "amber14-all.xml"
    water_ff = "amber14/tip3pfb.xml"

    # outputs
    pdb_out = "traj.pdb"
    data_out = "md_log.txt"

    # options
    run_sim = False
    analyze = True

    main(pdb, protein_ff, water_ff, pdb_out, data_out, run_sim, analyze)
