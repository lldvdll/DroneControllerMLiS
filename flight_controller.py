import numpy as np
from drone import Drone
from typing import Tuple, List

Point = Tuple[float, float]


class FlightController():

    @classmethod
    def __init__(self):
        
        # Target configuration
        self.target_mode = "fixed"  # "fixed", "random"
        self.n_targets_random = 5
        
        # World bounds
        self.full_bounds = (-0.75, 0.75, -0.5, 0.5)  # xmin, xmax, ymin, ymax
        
        # Target sampling constraints
        self.min_separation = 0.20 # min distance between targets
        self.min_from_origin = 0.15 # min distance from origin (0,0)
        self.min_from_bounds = 0.1  # min distance from bounds

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
    def _fixed_targets(self) -> List[Point]:
        return [
            (0.35, 0.3),
            (-0.35, 0.4),
            (0.5, -0.4),
            (-0.35, 0.0),
        ]
        
    @classmethod
    def _simple_random_targets(self, n, limits):
        targets = []
        for _ in range(n):
            x = np.random.uniform(limits[0], limits[1])
            y = np.random.uniform(limits[0], limits[1])
            targets.append((x, y))
        return targets
    
    @classmethod
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
    
    def init_drone(self, mode='random', 
                   num_targets=100, limits=(-0.5, 0.5),
                   curriculum_max_episodes=2000, episode=0) -> Drone:
        """Creates a Drone object initialised with a deterministic set of target coordinates.

        Args:
            mode (str, optional): Either 'deterministic' or 'random'. Defaults to 'deterministic'.
            num_targets (int, optional): The number of targets to generate if mode is 'random'. Defaults to 100.

        Returns:
            Drone: An initial drone object with some programmed target coordinates.
        """
        drone = Drone()
        
        effective_mode = mode if mode is not None else self.target_mode
        
        # Provided fixed targets
        if effective_mode == "fixed":
            targets = self._fixed_targets()
            
        # All targets at (0,0) 
        # Adds enough to always be more than typical max_steps - note this is hacky, but good enough
        elif effective_mode == "hover":
            targets = [(0.0, 0.0)] * 10000

        # "random" but with a bunch of constrains 
        elif effective_mode == "random":
            targets = self._sample_random_targets(
                n=self.n_targets_random,
                bounds=self.full_bounds,
                min_sep=self.min_separation,
                min_from_origin=self.min_from_origin,
                min_from_bounds = self.min_from_bounds
            )
            
        # Add 100 uniform random targets within specified range
        elif effective_mode == 'simple_random':
            targets = self._simple_random_targets(n=num_targets, limits=limits)
                
        # Random targets within increasing radius around centre (0.2 < radius < 0.5)
        elif effective_mode == "increasing":
            progress = episode / curriculum_max_episodes
            max_range = min(1.0, progress) / 2.0
            limit = max(0.2, max_range)
            limits = (-limit, limit)
            targets = self._simple_random_targets(n=num_targets, limits=limits)

        else:
            targets = self._fixed_targets()
            print(f"Unknown target_mode: {effective_mode}. Defaulting to fixed targets.")
        
        for points in targets:
            drone.add_target_coordinate(points)

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