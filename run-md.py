from openmm.app import *
from openmm import *
from openmm.unit import *
from sys import stdout
import time

# Record the start time
start_time = time.time()

pdb = PDBFile("parent_0001.pdb")
forcefield = ForceField("amber14-all.xml", "amber14/tip3pfb.xml")
modeller = Modeller(pdb.topology, pdb.positions)
modeller.addSolvent(forcefield, padding=1.0*nanometers)
system = forcefield.createSystem(
    modeller.topology, nonbondedMethod=PME, nonbondedCutoff=1 * nanometer, constraints=HBonds
)
integrator = LangevinMiddleIntegrator(300 * kelvin, 1 / picosecond, 0.002 * picoseconds)

simulation = Simulation(modeller.topology, system, integrator)
simulation.context.setPositions(modeller.positions)

# Write the starting structure to a pdb file
PDBFile.writeFile(simulation.topology, simulation.context.getState(getPositions=True).getPositions(), open("start.pdb", "w"))

print("Minimizing structure...")
simulation.minimizeEnergy()

print("Starting simulation...")

print("Starting NVT equilibration...")

# simulation.context.setVelocitiesToTemperature(300*kelvin)
print("Running for {} steps, ({} ps)".format(10000, 10000 * 0.002))
simulation.reporters.append(DCDReporter("nvt_equilibration.dcd", 1000))
simulation.reporters.append(StateDataReporter(stdout, 1000, step=True, potentialEnergy=True, temperature=True))
simulation.step(10000)

# After stabilizing the temperature and creating the proper solvent-solute orientation in the file,
# we need to stabilize the pressure of the system. We do this with NPT equilibriation, in which the number of particles,
# pressure, and temperature are kept constant.
print("Starting NPT equilibration...")
simulation.reporters.clear()
# Apply pressure barometrization to create constant pressure within the solvent environment.
barostat = MonteCarloBarostat(1.0*bar, 300*kelvin)
system.addForce(barostat)
simulation.context.reintialize(preserveState=True)
simulation.reporters.append(DCDReporter("npt_equilibration.dcd", 1000))
simulation.reporters.append(StateDataReporter(stdout, 1000, step=True, potentialEnergy=True, temperature=True))
# Run for 10000 steps, which is 20 ps.
print("Running for {} steps, ({} ps)".format(10000, 10000 * 0.002))
simulation.step(10000)

# After running equillibriation and pressure barometrization, our system is now properly equillibrated at our desired temperature
# and pressure. We can now release the position restraints and run the production MD for the final data collection.
# which will be placed in the .log file.
production_steps = 500000
print("Starting production run...")
# Run for 500000 steps, which is 1000 ps.
print("Running for {} steps, ({} ps)".format(production_steps, production_steps*0.002))
simulation.reporters.clear()
simulation.reporters.append(DCDReporter("prod.dcd", 100))
simulation.reporters.append(StateDataReporter("prod.log", 1000, step=True, potentialEnergy=True, temperature=True))
simulation.step(production_steps)

print("Done!")
PDBFile.writeFile(simulation.topology, simulation.context.getState(getPositions=True).getPositions(), open("end.pdb", "w"))

# Record the end time
end_time = time.time()

print("Elapsed time: %.2f seconds" % (end_time - start_time))
