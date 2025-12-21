import numpy as np
from drone import Drone
from typing import Tuple


class FlightController():

    @classmethod
    def get_max_simulation_steps(self):
        return 500
    @classmethod
    def get_time_interval(self):
        return 0.01

    @classmethod
    def get_thrusts(self, drone: Drone) -> Tuple[float, float]:
        """Takes a given drone object, containing information about its current state
        and calculates a pair of thrust values for the left and right propellers.

        Args:
            drone (Drone): The drone object containing the information about the drones state.

        Returns:
            Tuple[float, float]: A pair of floating point values which respectively represent the thrust of the left and right propellers, must be between 0 and 1 inclusive.
        """


        # The default controller sets each propeller to a value of 0.5 0.5 to stay stationary.
        return (0.5, 0.5)

    @classmethod
    def train(self):
        pass

    @classmethod
    def init_drone(self, mode='deterministic', num_targets=100) -> Drone:
        """Creates a Drone object initialised with a deterministic set of target coordinates.

        Args:
            mode (str, optional): Either 'deterministic' or 'random'. Defaults to 'deterministic'.
            num_targets (int, optional): The number of targets to generate if mode is 'random'. Defaults to 100.

        Returns:
            Drone: An initial drone object with some programmed target coordinates.
        """
        drone = Drone()
        if mode == 'deterministic':  # Fixed set of targets, as given in the problem statement
            drone.add_target_coordinate((0.35, 0.3))
            drone.add_target_coordinate((-0.35, 0.4))
            drone.add_target_coordinate((0.5, -0.4))
            drone.add_target_coordinate((-0.35, 0))
        elif mode == 'random':
            for i in range(num_targets):  # Add 100 random targets for experimentation
                x = np.random.uniform(-0.5, 0.5)
                y = np.random.uniform(-0.5, 0.5)
                drone.add_target_coordinate((x, y))
        return drone

    @classmethod
    def load(self):
        """Load the parameters of this flight controller from disk.
        """
        pass

    @classmethod
    def save(self):
        """Save the parameters of this flight controller to disk.
        """
        pass