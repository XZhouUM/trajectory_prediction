import numpy as np

from sim.motion_model import MotionModel


class KinematicBicycleModel(MotionModel):
    """Kinematic bicycle model discretized by forward Euler.

    State vector: [x, y, v, psi], where x, y are the position coordinates [m], v is the speed [m/s], and psi is the heading angle [rad].
    Control vector: [a, delta], where a is the acceleration [m/s^2], and delta is the steering angle [rad].

    The parameters of the model are:
    L: wheelbase of the vehicle [m], which is the distance between the front and rear axles. This parameter affects the turning radius of the vehicle and is used in the calculation of the heading angle change.
    """

    def __init__(
        self,
        dt: float = 0.1,
        L: float = 2.5,
    ):
        super().__init__(expected_state_dim=4, expected_control_dim=2, dt=dt)
        self._L = float(L)

    def transition(self, state: np.ndarray, control: np.ndarray) -> np.ndarray:
        """State equation of the kinematic bicycle model discretized by forward Euler.

        The kinematic bicycle model equations are:
            x_dot = v * cos(psi)
            y_dot = v * sin(psi)
            v_dot = a
            psi_dot = (v / L) * tan(delta)

        After the forward Euler discretization, the discrete state equations are:
        x(k+1) = x(k) + v(k) * cos(psi(k)) * dt
        y(k+1) = y(k) + v(k) * sin(psi(k)) * dt
        v(k+1) = v(k) + a(k) * dt
        psi(k+1) = psi(k) + (v(k) / L) * tan(delta(k)) * dt
        """


        x, y, v, psi = state
        a, delta = control

        next_x = x + v * np.cos(psi) * self._dt
        next_y = y + v * np.sin(psi) * self._dt
        next_v = v + a * self._dt
        next_psi = psi + (v / self._L) * np.tan(delta) * self._dt

        return np.array([next_x, next_y, next_v, next_psi], dtype=float)
