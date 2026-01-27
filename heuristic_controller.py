import numpy as np
from flight_controller import FlightController
from drone import Drone
from typing import List, Tuple

Point = Tuple[float, float]

class HeuristicController(FlightController):


    def __init__(self):
        """Creates a heuristic flight controller with some specified parameters

        """

        self.ky = 1.0
        self.kx = 0.5
        self.abs_pitch_delta = 0.1
        self.abs_thrust_delta = 0.3
        
        # Target configuration
        self.target_mode = "fixed"  # "fixed", "random"
        self.n_targets_random = 5
        
        # World bounds
        self.full_bounds = (-0.75, 0.75, -0.5, 0.5)  # xmin, xmax, ymin, ymax
        
        # Target sampling constraints
        self.min_separation = 0.20 # min distance between targets
        self.min_from_origin = 0.15 # min distance from origin (0,0)
        self.min_from_bounds = 0.1  # min distance from bounds
        
    def _fixed_targets(self) -> List[Point]:
        return [
            (0.35, 0.3),
            (-0.35, 0.4),
            (0.5, -0.4),
            (-0.35, 0.0),
        ]
    
    def _sample_random_targets(self, n: int, bounds: Tuple[float, float, float, float],
                               min_sep: float, min_from_origin: float, min_from_bounds: float,
                               max_tries: int = 5000) -> List[Point]:
        """
        Sample n random targets with constraints:
        - min_sep: minimum distance between targets
        - min_from_origin: minimum distance from (0,0)
        - min_from_bounds: minimum distance from boundaries
        """
        xmin, xmax, ymin, ymax = bounds
        targets: List[Point] = []
        tries = 0

        while len(targets) < n and tries < max_tries:
            tries += 1
            x = float(np.random.uniform(xmin, xmax))
            y = float(np.random.uniform(ymin, ymax))

            # Check: not too close to origin (0,0)
            if np.hypot(x, y) < min_from_origin:
                continue

            # Check: not too close to boundaries
            dist_to_left = x - xmin
            dist_to_right = xmax - x
            dist_to_bottom = y - ymin
            dist_to_top = ymax - y
            if min(dist_to_left, dist_to_right, dist_to_bottom, dist_to_top) < min_from_bounds:
                continue

            # Check: minimum separation from ALL existing targets
            valid = True
            for tx, ty in targets:
                if np.hypot(x - tx, y - ty) < min_sep:
                    valid = False
                    break
            if not valid:
                continue

            # All checks passed - add target
            targets.append((x, y))
        
        # fallback if constraints too strict
        if len(targets) < n:
            while len(targets) < n:
                x = float(np.random.uniform(xmin + min_from_bounds, xmax - min_from_bounds))
                y = float(np.random.uniform(ymin + min_from_bounds, ymax - min_from_bounds))
            
                targets.append((x, y))

        return targets
    
    def init_drone(self) -> Drone:
        """
        Create and initialize drone with targets based on mode
        """
        drone = Drone()

        if self.target_mode == "fixed":
            targets = self._fixed_targets()

        elif self.target_mode == "random":
            targets = self._sample_random_targets(
                n=self.n_targets_random,
                bounds=self.full_bounds,
                min_sep=self.min_separation,
                min_from_origin=self.min_from_origin,
                min_from_bounds = self.min_from_bounds
            )

        else:
            raise ValueError(f"Unknown target_mode: {self.target_mode}")
        
        for points in targets:
            drone.add_target_coordinate(points)

        return drone

    def get_max_simulation_steps(self):
            return 3000 # You can alter the amount of steps you want your program to run for here


    def get_thrusts(self, drone: Drone) -> Tuple[float, float]:
        """Takes a given drone object, containing information about its current state
        and calculates a pair of thrust values for the left and right propellers.

        Args:
            drone (Drone): The drone object containing the information about the drones state.

        Returns:
            Tuple[float, float]: A pair of floating point values which respectively represent the thrust of the left and right propellers, must be between 0 and 1 inclusive.
        """

        target_point = drone.get_next_target()
        dx = target_point[0] - drone.x
        dy = target_point[1] - drone.y

        thrust_adj = np.clip(dy * self.ky, -self.abs_thrust_delta, self.abs_thrust_delta)
        target_pitch = np.clip(dx * self.kx, -self.abs_pitch_delta, self.abs_pitch_delta)
        delta_pitch = target_pitch-drone.pitch

        thrust_left = np.clip(0.5 + thrust_adj + delta_pitch, 0.0, 1.0)
        thrust_right = np.clip(0.5 + thrust_adj - delta_pitch, 0.0, 1.0)

        # The default controller sets each propeller to a value of 0.5 0.5 to stay stationary.
        return (thrust_left, thrust_right)

    def train(self):
        """A self contained method designed to train parameters created in the initialiser.
        """
        # --- Code snipped provided for guidance only --- #
        # for n in range(epochs):
        #     # 1) modify parameters
            
        #     # 2) create a new drone simulation
        #     drone = self.init_drone()
        #     # 3) run simulation
        #     for t in range(self.get_max_simulation_steps()):
        #         drone.set_thrust(self.get_thrusts(drone))
        #         drone.step_simulation(self.get_time_interval())
        #     # 4) measure change in quality

        #     # 5) update parameters according to algorithm

        pass

    def load(self):
        """Load the parameters of this flight controller from disk.
        """
        try:
            parameter_array = np.load('heuristic_controller_parameters.npy')
            self.ky = parameter_array[0]
            self.kx = parameter_array[1]
            self.abs_pitch_delta = parameter_array[2]
            self.abs_thrust_delta = parameter_array[3]
        except:
            print("Could not load parameters, sticking with default parameters.")

    def save(self):
        """Save the parameters of this flight controller to disk.
        """
        parameter_array = np.array([self.ky, self.kx, self.abs_pitch_delta, self.abs_thrust_delta])
        np.save('heuristic_controller_parameters.npy', parameter_array)
        