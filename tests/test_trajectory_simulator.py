import numpy as np

from sim.trajectory_simulator import TrajectorySimulator


def test_simulator_uses_vehicle_speed_and_steering_in_correct_order():
    simulator = TrajectorySimulator(model_name="KinematicBicycleModel", dt=0.1)

    initial_state = np.array([0.0, 0.0, 5.0, 0.0])  # x, y, v, psi
    controls = np.tile(np.array([0.0, 0.2]), (5, 1))  # [a, delta]

    states = simulator.simulate(initial_state=initial_state, controls=controls)

    # The vehicle should move forward in the x direction and turn left, so the x position and heading angle should increase, while the speed remains constant.
    assert states.shape == (6, 4)
    assert states[1, 0] > states[0, 0]
    assert states[-1, 0] > states[0, 0]
    assert states[-1, 3] > states[0, 3]
    assert states[-1, 2] == states[0, 2]
