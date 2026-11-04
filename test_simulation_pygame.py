import numpy as np
import pytest

import simulation_pygame as simulation


def make_atoms(positions, velocities=None):
    positions = np.asarray(positions, dtype=np.float64)
    if positions.ndim == 1:
        positions = positions.reshape(1, 3)

    atoms = np.zeros((len(positions), 9), dtype=np.float64)
    atoms[:, :3] = positions
    if velocities is not None:
        atoms[:, 3:6] = np.asarray(velocities, dtype=np.float64).reshape(-1, 3)
    return atoms


def test_sample_spherical_returns_unit_vectors():
    vectors = simulation.sample_spherical(25)

    assert vectors.shape == (3, 25)
    np.testing.assert_allclose(np.linalg.norm(vectors, axis=0), 1.0)


def test_make_atoms_initializes_position_velocity_and_cell_columns():
    np.random.seed(123)

    atoms = simulation.make_atoms(12)

    assert atoms.shape == (12, 9)
    assert np.issubdtype(atoms.dtype, np.floating)
    np.testing.assert_array_equal(
        atoms[:, 6:9],
        np.floor(atoms[:, :3] / simulation.INITIAL_RADIUS),
    )
    assert np.all(np.linalg.norm(atoms[:, 3:6], axis=1) > 0)


def test_update_cells_uses_floor_for_negative_coordinates():
    atoms = make_atoms([[-101, 99, -1], [100, 200, -200]])

    simulation.update_cells(atoms, 100)

    np.testing.assert_array_equal(atoms[:, 6:9], [[-2, 0, -1], [1, 2, -2]])


def test_get_neighbours_filters_cutoff_and_coincident_particles():
    atoms = make_atoms([[0, 0, 0], [50, 0, 0], [100, 0, 0], [0, 0, 0]])
    simulation.update_cells(atoms, 100)
    cells = {}
    for index, cell in enumerate(atoms[:, 6:9].astype(int)):
        cells.setdefault(tuple(cell), []).append(index)

    neighbours = simulation.get_neighbours(0, atoms, cells, 100)

    np.testing.assert_array_equal(neighbours[:, :3], [[50, 0, 0]])


def test_calculate_forces_is_equal_and_opposite_for_an_interacting_pair():
    atoms = make_atoms([[0, 0, 0], [50, 0, 0]])

    forces = simulation.calculate_forces(atoms, stiffness=5, interaction_radius=100)

    np.testing.assert_allclose(forces, [[-250, 0, 0], [250, 0, 0]])
    np.testing.assert_allclose(np.sum(forces, axis=0), [0, 0, 0])


@pytest.mark.parametrize(
    "positions",
    [
        [[0, 0, 0], [100, 0, 0]],
        [[0, 0, 0], [150, 0, 0]],
        [[0, 0, 0], [0, 0, 0]],
    ],
)
def test_calculate_forces_is_finite_and_zero_outside_active_pair_range(positions):
    atoms = make_atoms(positions)

    forces = simulation.calculate_forces(atoms, stiffness=5, interaction_radius=100)

    assert np.isfinite(forces).all()
    np.testing.assert_array_equal(forces, np.zeros((2, 3)))


def test_potential_energy_counts_each_active_pair_once():
    atoms = make_atoms([[0, 0, 0], [40, 0, 0], [200, 0, 0]])

    energy = simulation.potential_energy(atoms, stiffness=5, interaction_radius=100)

    assert energy == pytest.approx(9000)


def test_potential_energy_is_zero_at_and_beyond_cutoff():
    atoms = make_atoms([[0, 0, 0], [100, 0, 0], [250, 0, 0]])

    assert simulation.potential_energy(atoms, 5, 100) == 0


def test_pair_force_is_negative_gradient_of_potential_energy():
    atoms = make_atoms([[0, 0, 0], [50, 0, 0]])
    force = simulation.calculate_forces(atoms, stiffness=5, interaction_radius=100)[0, 0]
    epsilon = 1e-4

    atoms[0, 0] += epsilon
    energy_plus = simulation.potential_energy(atoms, 5, 100)
    atoms[0, 0] -= 2 * epsilon
    energy_minus = simulation.potential_energy(atoms, 5, 100)

    numerical_force = -(energy_plus - energy_minus) / (2 * epsilon)
    assert force == pytest.approx(numerical_force, rel=1e-7)


def test_energy_components_sum_kinetic_and_potential_energy():
    atoms = make_atoms([[0, 0, 0], [40, 0, 0]], [[3, 0, 0], [0, 4, 0]])

    kinetic, potential, total = simulation.energy_components(atoms, 5, 100)

    assert kinetic == pytest.approx(12.5)
    assert potential == pytest.approx(9000)
    assert total == pytest.approx(kinetic + potential)


def test_step_advances_a_free_particle_by_velocity_and_returns_atoms():
    atoms = make_atoms([[10, -20, 30]], [[2, -4, 6]])

    result = simulation.step(atoms, stiffness=5, interaction_radius=100, dt=0.1)

    assert result is atoms
    np.testing.assert_allclose(atoms[0, :3], [10.2, -20.4, 30.6])
    np.testing.assert_allclose(atoms[0, 3:6], [2, -4, 6])


def test_step_reflects_at_wall_without_changing_kinetic_energy():
    atoms = make_atoms([[9.8, 0, 0]], [[2, 0, 0]])
    initial_energy = simulation.kinetic_energy(atoms)

    simulation.step(atoms, stiffness=5, interaction_radius=2, bounds=10, dt=0.2)

    assert atoms[0, 0] == pytest.approx(9.8)
    assert atoms[0, 3] == pytest.approx(-2)
    assert simulation.kinetic_energy(atoms) == pytest.approx(initial_energy)


def test_velocity_verlet_keeps_total_energy_close_to_constant():
    atoms = make_atoms([[-30, 0, 0], [30, 0, 0]], [[-1, 0, 0], [1, 0, 0]])
    stiffness, radius = 5, 100
    initial_energy = simulation.energy_components(atoms, stiffness, radius)[2]
    energies = []

    for _ in range(500):
        simulation.step(atoms, stiffness, radius)
        energies.append(simulation.energy_components(atoms, stiffness, radius)[2])

    maximum_relative_drift = max(
        abs(energy - initial_energy) / initial_energy for energy in energies
    )
    assert maximum_relative_drift < 3e-4


def test_reset_initializes_energy_history_and_reference():
    state = simulation.ParticleSimulation()

    kinetic, potential, total = simulation.energy_components(
        state.atoms, state.stiffness, state.interaction_radius
    )

    assert state.simulated_time == 0
    assert state.times == [0]
    assert state.kinetic_energies == [kinetic]
    assert state.potential_energies == [potential]
    assert state.total_energies == [total]
    assert state.reference_energy == total


def test_advance_does_not_change_state_when_paused():
    state = simulation.ParticleSimulation()
    state.paused = True
    original_atoms = state.atoms.copy()
    original_times = state.times.copy()

    state.advance()

    np.testing.assert_array_equal(state.atoms, original_atoms)
    assert state.times == original_times
    assert state.simulated_time == 0


def test_advance_integrates_substeps_and_appends_energy_history(monkeypatch):
    state = simulation.ParticleSimulation()
    calls = []

    def record_step(atoms, stiffness, radius, dt):
        calls.append((stiffness, radius, dt))
        return atoms

    monkeypatch.setattr(simulation, "step", record_step)
    state.advance()

    assert len(calls) == simulation.INTEGRATION_STEPS
    assert all(call == (state.stiffness, state.interaction_radius, simulation.INTEGRATION_DT) for call in calls)
    assert state.simulated_time == pytest.approx(simulation.DT)
    assert state.times[-1] == pytest.approx(simulation.DT)
    assert len(state.kinetic_energies) == 2
    assert len(state.potential_energies) == 2
    assert len(state.total_energies) == 2


def test_particle_count_is_clamped_and_reset_recreates_particles():
    state = simulation.ParticleSimulation()

    state.set_particle_count(simulation.MAX_PARTICLES + 10)
    assert state.particle_count == simulation.MAX_PARTICLES
    assert state.atoms.shape == (simulation.MAX_PARTICLES, 9)

    state.set_particle_count(0)
    assert state.particle_count == simulation.MIN_PARTICLES
    assert state.atoms.shape == (simulation.MIN_PARTICLES, 9)
    assert state.simulated_time == 0
    assert state.times == [0]


def test_reset_energy_history_rebaselines_after_parameter_change():
    state = simulation.ParticleSimulation()
    state.advance()
    state.stiffness *= 2

    state.reset_energy_history()

    assert state.simulated_time == 0
    assert state.times == [0]
    assert state.reference_energy == pytest.approx(state.total_energies[0])


def test_energy_history_is_bounded_to_recent_simulated_time():
    state = simulation.ParticleSimulation()

    for _ in range(20):
        state.advance()

    assert len(state.times) < 20
    assert state.times[-1] == pytest.approx(state.simulated_time)
    assert state.times[0] >= state.simulated_time - simulation.HISTORY_SECONDS - simulation.DT
    assert len(state.times) == len(state.kinetic_energies)
    assert len(state.times) == len(state.potential_energies)
    assert len(state.times) == len(state.total_energies)
