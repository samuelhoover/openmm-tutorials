# pyright: basic


from sys import stdout

import matplotlib.pyplot as plt
import numpy as np
import openmm as mm
import openmm.app as app
import openmm.unit as unit


def load_pdb(pdb):
    return app.PDBFile(pdb)


def get_forcefield(protein_ff):
    return app.ForceField(protein_ff)


def create_system(pdb, forcefield):
    system = forcefield.createSystem(
        pdb.topology,
        nonbondedMethod=app.NoCutoff,
        constraints=app.HBonds,
        hydrogenMass=1.5 * unit.amu,
    )
    integrator = mm.LangevinMiddleIntegrator(
        300 * unit.kelvin, 1 / unit.picosecond, 0.004 * unit.picoseconds
    )
    simulation = app.Simulation(pdb.topology, system, integrator)
    simulation.context.setPositions(pdb.positions)
    simulation.reporters.append(app.DCDReporter("smd_traj.dcd", 10000))
    simulation.reporters.append(
        app.StateDataReporter(
            stdout,
            10000,
            step=True,
            time=True,
            potentialEnergy=True,
            temperature=True,
            speed=True,
        )
    )

    return simulation, system


def equilibrate(simulation):
    simulation.context.setVelocitiesToTemperature(300 * unit.kelvin)
    simulation.step(1000)

    return simulation


def define_CV():
    index1 = 8
    index2 = 98
    cv = mm.CustomBondForce("r")
    cv.addBond(index1, index2)

    return cv


def steered_md(system, simulation, cv, windows):
    r0 = 1.3 * unit.nanometers
    fc_pull = 1000.0 * unit.kilojoules_per_mole / unit.nanometers**2
    v_pulling = 0.02 * unit.nanometers / unit.picosecond  # [nm/ps]
    dt = simulation.integrator.getStepSize()
    total_steps = 30000  # 120 ps
    increment_steps = 10

    pullingForce = mm.CustomCVForce("0.5 * fc_pull * (cv - r0) ^ 2")
    pullingForce.addGlobalParameter("fc_pull", fc_pull)
    pullingForce.addGlobalParameter("r0", r0)
    pullingForce.addCollectiveVariable("cv", cv)
    system.addForce(pullingForce)
    simulation.context.reinitialize(preserveState=True)

    window_coords = []
    window_index = 0

    for i in range(total_steps // increment_steps):
        simulation.step(increment_steps)
        current_cv_value = pullingForce.getCollectiveVariableValues(simulation.context)

        if (i * increment_steps) % 5000 == 0:
            print(f"r0 = {r0}, r = {current_cv_value[0]}")

        r0 += v_pulling * dt * increment_steps
        simulation.context.setParameter("r0", r0)

        if window_index < len(windows) and current_cv_value >= windows[window_index]:
            window_coords.append(
                simulation.context.getState(
                    getPositions=True, enforcePeriodicBox=False
                ).getPositions()
            )
            window_index += 1

    for i, coords in enumerate(window_coords):
        with open(f"window_{i}.pdb", "w") as f:
            app.PDBFile.writeFile(simulation.topology, coords, f)

    return pullingForce


def run_window(simulation, windows, window_index, pullingForce):
    print(f"Running winow {window_index}")

    pdb = app.PDBFile(f"window_{window_index}.pdb")

    simulation.context.setPositions(pdb.positions)

    r0 = windows[window_index]
    simulation.context.setParameter("r0", r0)

    simulation.context.setVelocitiesToTemperature(300 * unit.kelvin)
    simulation.step(1000)

    total_steps = 100000

    record_steps = 1000

    cv_values = []
    for i in range(total_steps // record_steps):
        simulation.step(record_steps)

        current_cv_value = pullingForce.getCollectiveVariableValues(simulation.context)
        cv_values.append([i, current_cv_value[0]])

    np.savetxt(f"cv_values_window_{window_index}.txt", np.array(cv_values))

    print(f"Completed window {window_index}")


def create_windows(pdb, forcefield):
    pdbFile = load_pdb(pdb)
    forcefield = get_forcefield(forcefield)
    simulation, system = create_system(pdbFile, forcefield)
    equil = equilibrate(simulation)
    cv = define_CV()
    pullingForce = steered_md(system, equil, cv, num_windows)

    return simulation, pullingForce


def pull(simulation, windows, num_windows, pullingForce):
    for n in range(num_windows):
        run_window(simulation, windows, n, pullingForce)


def analyze_windows(windows):
    metafilelines = []

    fig, ax = plt.subplots()

    for i, _ in enumerate(windows):
        data = np.loadtxt(f"cv_values_window_{i}.txt")
        ax.hist(data[:, 1])
        metafilelines.append(f"cv_values_window_{i}.txt {windows[i]} 1000\n")

    ax.set_xlabel("r [nm]")
    ax.set_ylabel("Count")

    fig.tight_layout()
    plt.show()
    fig.savefig("cv-histogram.pdf")

    with open("metafile.txt", "w") as f:
        f.writelines(metafilelines)


def analyze_PMF(file):
    pmf = np.loadtxt(file)

    fig, ax = plt.subplots()

    ax.plot(pmf[:, 0], pmf[:, 1])

    ax.set_xlabel("r [nm]")
    ax.set_ylabel("PMF [kJ/mol]")

    fig.tight_layout()
    plt.show()
    fig.savefig("pmf.pdf")


def main(pdb, forcefield, num_windows, run=True, analysis=True):
    windows = np.linspace(1.3, 3.3, num_windows)

    if run:
        simulation, pullingForce = create_windows(pdb, forcefield)
        pull(simulation, windows, num_windows, pullingForce)

    if analysis:
        analyze_windows(windows)
        analyze_PMF("pmf.txt")


if __name__ == "__main__":
    pdb = "deca-ala.pdb"
    ff = "amber14-all.xml"

    num_windows = 24

    run = False
    analysis = True

    main(pdb, ff, num_windows, run, analysis)
