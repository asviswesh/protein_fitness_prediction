from openmm.app import *
from openmm import *
from openmm.unit import *
import argparse
from sys import stdout
import time

start_time = time.time()

parser = argparse.ArgumentParser()
parser.add_argument("pdb_filename")
args = parser.parse_args()

pdb_file = args.pdb_filename

pdb = PDBFile(pdb_file)
forcefield = ForceField("amber14-all.xml", "amber14/tip3pfb.xml")
modeller = Modeller(pdb.topology, pdb.positions)
modeller.addSolvent(forcefield, padding=1.0*nanometers)
system = forcefield.createSystem(
    modeller.topology, nonbondedMethod=PME, nonbondedCutoff=1 * nanometer, constraints=HBonds
)
integrator = LangevinMiddleIntegrator(300 * kelvin, 1 / picosecond, 0.002 * picoseconds)

simulation = Simulation(modeller.topology, system, integrator)
simulation.context.setPositions(modeller.positions)

PDBFile.writeFile(simulation.topology, simulation.context.getState(getPositions=True).getPositions(), open("start.pdb", "w"))

print("Minimizing structure...")
simulation.minimizeEnergy()

print("Starting simulation...")

print("Starting NVT equilibration...")

print("Running for {} steps, ({} ps)".format(10000, 10000 * 0.002))
simulation.reporters.append(DCDReporter("nvt_equilibration.dcd", 1000))
simulation.reporters.append(StateDataReporter(stdout, 1000, step=True, potentialEnergy=True, temperature=True))
simulation.step(10000)

print("Starting NPT equilibration...")
simulation.reporters.clear()
# Apply pressure barometrization to create constant pressure within the solvent environment.
barostat = MonteCarloBarostat(1.0*bar, 300*kelvin)
system.addForce(barostat)
simulation.context.reintialize(preserveState=True)
simulation.reporters.append(DCDReporter("npt_equilibration.dcd", 1000))
simulation.reporters.append(StateDataReporter(stdout, 1000, step=True, potentialEnergy=True, temperature=True))
print("Running for {} steps, ({} ps)".format(10000, 10000 * 0.002))
simulation.step(10000)

production_steps = 500000
print("Starting production run...")
print("Running for {} steps, ({} ps)".format(production_steps, production_steps*0.002))
simulation.reporters.clear()
simulation.reporters.append(DCDReporter("prod.dcd", 100))
simulation.reporters.append(StateDataReporter("prod.log", 1000, step=True, potentialEnergy=True, temperature=True))
simulation.step(production_steps)

print("Done!")
PDBFile.writeFile(simulation.topology, simulation.context.getState(getPositions=True).getPositions(), open("end.pdb", "w"))

end_time = time.time()

print("Elapsed time: %.2f seconds" % (end_time - start_time))
