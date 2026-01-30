"""
SARSA (on-policy) Drone Flight Controller
    - 3 target modes: fixed, random, curriculum
    - World bounds:
        terminate if out-of-bounds (large penalty)
        near-boundary shaping penalty (smooth)
        danger flag in state if near boundary
    - Discrete state:
        dx: 7 bins
        dy: 7 bins
        vx, vy, theta, omega: 3 bins each
        danger: 0/1 flag
    - Actions: 6 macro-actions
        0 HOVER_STABILISE
        1 TILT_LEFT
        2 TILT_RIGHT
        3 BOOST_UP
        4 BOOST_DOWN
        5 ARREST_MOTION
    - Reward:
        +R_hit when a target is reached
        progress shaping based on normalised distance improvement
        -near_boundary_penalty when close to walls
        -boundary_penalty on out-of-bounds termination
    - Training:
        performance-based curriculum (upgrade stage based on rolling avg hits/returns)
        action_repeat (hold macro-action for several simulator steps)
        stagnation early-stop (terminate when not improving)
    """
import numpy as np
import os
from typing import List, Tuple, Dict, Any
from drone import Drone
from flight_controller import FlightController
import json
import csv
from collections import Counter

Point = Tuple[float, float]

class CustomController(FlightController):
    
    def __init__(self):

        # Episode tracking
        self.episode_count = 0

        # Target configuration
        self.target_mode = "fixed"  # "fixed", "random", or "curriculum"
        self.n_targets_random = 5

        # World bounds
        self.full_bounds = (-0.75, 0.75, -0.5, 0.5)  # xmin, xmax, ymin, ymax
        self.target_bounds = (-0.6, 0.6, -0.4, 0.4)  # early training central target region
        
        # Target sampling constraints
        self.min_separation = 0.20 # min distance between targets
        self.min_from_origin = 0.15 # min distance from origin (0,0)
        self.min_from_bounds = 0.1  # min distance from bounds

        # Simulation settings
        self.max_steps = 5000
        self.dt = 0.01

        # Discretisation thresholds
        self.dx1, self.dx2, self.dx3 = 0.105, 0.3, 0.45
        self.dy1, self.dy2, self.dy3 = 0.05, 0.2, 0.3
        self.vx0 = 0.05
        self.vy0 = 0.03
        self.theta0 = 0.1 # for pitch
        self.omega0 = 0.1 # for pitch_velocity

        # danger threshold: min distance to any wall
        self.b0 = 0.1

        # State tracking
        self.last_state_tuple = None
        self.last_state_id = None

        # Reward parameters
        self.R_hit = 500.0
        # self.R_progress = 0.5
        self.k_progress = 150.0
        self.k_backwards = 0.0
        self.progress_clip = 0.05
        self.boundary_penalty = 300.0 # terminate penalty if out-of-bounds
        self.near_boundary_penalty_scale = 0.30  # shaping weight near boundary
        self.c_step = 0.01

        # default parameters
        self.n_states = 7938
        self.n_actions = 6
        self.gamma = 0.999 # gamma discount
        self.alpha = 0.10 # learning rate
        self.epsilon = 1.0 # exploration
        self.eps_decay = 0.9995
        self.eps_min = 0.2
        self.epsilon_eval = 0.0 # no exploration during evaluation
        self.action_repeat_eval = 1

        # Q-table initialisation
        self.Q = np.zeros((self.n_states, self.n_actions), dtype=np.float32)
        # Initialise with small random values
        self.Q += np.random.randn(self.n_states, self.n_actions) * 0.01

        # Curriculum stages
        self.curriculum_stages = [
            {
                "name": "Stage0-Central-dxdyvelocity",
                "n": 2,
                "target_bounds": self.target_bounds, 
                "max_steps": 2500,
                "alpha": 0.1,
                "eps_start": 1.0, # Start with full exploration
                "eps_decay": 0.9995,
                "eps_min": 0.15,
                "action_repeat": 1,
                "boundary_penalty": 80,
                "near_boundary_penalty_scale": 0.3,
            },
            {
                "name": "Stage1-Full-dxdyvelocity",
                "n": 3,
                "target_bounds": self.full_bounds, 
                "max_steps": 3000,
                "alpha": 0.1,
                "eps_start": 0.6,  # Start with moderate exploration
                "eps_decay": 0.9995,
                "eps_min": 0.1,
                "action_repeat": 3,
                "k_progress": self.k_progress,
                "k_backwards": 0,
                "boundary_penalty": 80,
                "near_boundary_penalty_scale": 0.3,
            },
            {
                "name": "Stage2-Full-all",
                "n": 5,
                "target_bounds": self.full_bounds, 
                "max_steps": 3500,
                "alpha": 0.1,
                "eps_start": 0.6,
                "eps_decay": 0.999,
                "eps_min": 0.05,
                "action_repeat": 3,
                "boundary_penalty": self.boundary_penalty,
                "near_boundary_penalty_scale": self.near_boundary_penalty_scale,
            },
        ]
        self.stage_idx = 0
        self.K = 50 # Rolling window size
        self.recent_returns: List[float] = [] # Total reward per episode
        self.recent_hits: List[int] = [] # Number of targets hit per episode
        self.hit_thresholds = [0.8, 1.5] # Hit thresholds for stage advancement

        # log training history
        self.q_path = None
        self.log_path = None
        self.csv_log_path = None
        self.flush_log_every = 10

    # ============================================================
    # Target generation
    # ============================================================
    def _all_targets_done(self, drone: Drone) -> bool:
        """Check if all targets have been reached"""
        try:
            return len(drone.target_coordinates) == 0
        except (AttributeError, TypeError):
            return False
    
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
    
    # ============================================================
    # Drone initialisation
    # ============================================================
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

        elif self.target_mode == "curriculum":
            stage = self.curriculum_stages[self.stage_idx]
            targets = self._sample_random_targets(
                n=int(stage["n"]),
                bounds=stage["target_bounds"],
                min_sep=self.min_separation,
                min_from_origin=self.min_from_origin,
                min_from_bounds = self.min_from_bounds
            )
        else:
            raise ValueError(f"Unknown target_mode: {self.target_mode}")
        
        for points in targets:
            drone.add_target_coordinate(points)

        return drone
    
    # ============================================================
    # State 
    # ============================================================
    def _distance_to_target(self, drone: Drone) -> float:
        """Compute Euclidean distance to current target"""
        tx, ty = drone.get_next_target()
        return float(np.hypot(drone.x - tx, drone.y - ty))
    
    def _distance_to_boundary(self, drone: Drone) -> float:
        """Compute minimum distance to any boundary"""
        xmin, xmax, ymin, ymax = self.full_bounds
        dx_left = abs(drone.x - xmin)
        dx_right = abs(xmax - drone.x)
        dy_bottom = abs(drone.y - ymin)
        dy_top = abs(ymax - drone.y)
        return float(min(dx_left, dx_right, dy_bottom, dy_top))
    
    def _out_of_bounds(self, drone: Drone) -> bool:
        """Check if drone is outside bounds"""
        xmin, xmax, ymin, ymax = self.full_bounds
        return (drone.x < xmin or drone.x > xmax or 
                drone.y < ymin or drone.y > ymax)
        
    def _bin_3(self, value: float,t0: float) -> int:
        if value < -t0:
            return 0
        if value > t0:
            return 2
        return 1

    def _bin_7(self, value: float, t1: float, t2: float, t3: float) -> int:
        if value < -t3:
            return 0
        if value < -t2:
            return 1
        if value < -t1:
            return 2
        if value <= t1:
            return 3
        if value <= t2:
            return 4
        if value <= t3:
            return 5
        return 6
    
    def get_state_tuple(self, drone: Drone) -> Tuple[int, int, int, int, int, int, int]:
        """
        Convert drone state to discrete tuple:
        (dx_bin, dy_bin, vx_bin, vy_bin, theta_bin, omega_bin, danger)
        """
        tx, ty = drone.get_next_target()

        # Relative position to target
        dx = tx - drone.x
        dy = ty - drone.y

        # Dynamics
        vx = float(drone.velocity_x)
        vy = float(drone.velocity_y)
        theta = float(drone.pitch)
        omega = float(drone.pitch_velocity)

        dx_bin = self._bin_7(dx, self.dx1, self.dx2, self.dx3)
        dy_bin = self._bin_7(dy, self.dy1, self.dy2, self.dy3)
        vx_bin = self._bin_3(vx, self.vx0)
        vy_bin = self._bin_3(vy, self.vy0)
        theta_bin = self._bin_3(theta, self.theta0)
        omega_bin = self._bin_3(omega, self.omega0)

        margin = self._distance_to_boundary(drone)
        danger = 1 if margin < self.b0 else 0

        state = (dx_bin, dy_bin, vx_bin, vy_bin, theta_bin, omega_bin, danger)

        sid = dx_bin
        sid = sid * 7 + dy_bin
        sid = sid * 3 + vx_bin
        sid = sid * 3 + vy_bin
        sid = sid * 3 + theta_bin
        sid = sid * 3 + omega_bin
        sid = sid * 2 + danger

        self.last_state_tuple = state
        self.last_state_id = sid
        
        return state
    
    
    # ============================================================
    # Action
    # ============================================================
    HOVER_STABILISE = 0
    TILT_LEFT = 1
    TILT_RIGHT = 2
    BOOST_UP = 3
    BOOST_DOWN = 4
    ARREST_MOTION = 5

    def _thrust_from_u_tau(self, u: float, tau: float) -> Tuple[float, float]:
        """
        Convert total thrust u and torque tau into individual thrusts (t1, t2).
        Using:
            t1 = u + tau
            t2 = u - tau
        with tau clipped such that both t1,t2 remain in [0,1].
        """
        u = float(np.clip(u, 0.0, 1.0)) # total thrust from both motors combined

        # Compute valid tau range
        tau_min = max(-u, u - 1.0)
        tau_max = min(1.0 - u, u)
        tau = float(np.clip(tau, tau_min, tau_max))

        t1 = u + tau
        t2 = u - tau
        return (t1, t2)

    def _h_hover_stabilise(self, drone: Drone) -> Tuple[float, float]:
        # Hover at current altitude with pitch stabilisation
        u0 = 0.5
        k_theta = 0.6
        k_omega = 0.45 # was 0.25
        tau = -k_theta * float(drone.pitch) - k_omega * float(drone.pitch_velocity)
        return self._thrust_from_u_tau(u0, tau)
    
    def _h_tilt_left(self, drone: Drone) -> Tuple[float, float]:
        curr_dist = self._distance_to_target(drone)
        curr_dist = max(0.0, min(curr_dist, 0.3))
        scale = curr_dist / 0.3   # 0 near, 1 far
        tx, ty = drone.get_next_target()
        dy = ty - drone.y
        
        # Target a bank angle of ~16 degrees (0.28 radians) - This is steep enough to move fast, but safe from flipping.
        target_pitch = -0.35
        
        # 1. Target Above: Climb (0.58 -> 0.64)
        if dy > 0.05 :
            u0 = 0.58 + (0.06 * scale)
        # 2. Target Below: Dive (0.48 -> 0.42)
        elif dy < -0.05:
            u0 = 0.48 - (0.06 * scale)
        else:
            u0 = 0.53
        
        # high k_theta, k_omega when far away; low k_theta, k_omega when close
        k_theta = 0.6 + 0.2 * scale   # [0.4 → 0.8] to [0.6 → 0.8]
        k_omega = 0.45 + 0.15 * scale    # [0.4  → 0.65] to [0.45  → 0.6] 
        
        # Error = Current - Target
        pitch_error = float(drone.pitch) - target_pitch
        
        tau = -k_theta * pitch_error - k_omega * float(drone.pitch_velocity)
        
        return self._thrust_from_u_tau(u0, tau)

    def _h_tilt_right(self, drone: Drone) -> Tuple[float, float]:
        curr_dist = self._distance_to_target(drone)
        curr_dist = max(0.0, min(curr_dist, 0.3))
        scale = curr_dist / 0.3   # 0 near, 1 far
        tx, ty = drone.get_next_target()
        dy = ty - drone.y
        
        target_pitch = 0.35
        
        # 1. Target Above: Climb (0.58 -> 0.64)
        if dy > 0.05 :
            u0 = 0.58 + (0.06 * scale)
        # 2. Target Below: Dive (0.48 -> 0.42)
        elif dy < -0.05:
            u0 = 0.48 - (0.06 * scale)
        else:
            u0 = 0.53
            
        # high k_theta, low k_omega when far away; low k_theta, high k_omega when close
        k_theta = 0.6 + 0.2 * scale   # [0.4 → 0.8] to [0.6 → 0.8]
        k_omega = 0.45 + 0.15 * scale    # [0.4  → 0.65] to [0.45  → 0.6] 

        pitch_error = float(drone.pitch) - target_pitch
        tau = -k_theta * pitch_error - k_omega * float(drone.pitch_velocity)
        
        return self._thrust_from_u_tau(u0, tau)

    def _h_boost_up(self, drone: Drone) -> Tuple[float, float]:
        # increase total thrust, keep level
        u0 = 0.62
        k_theta = 0.6
        k_omega = 0.45  # was 0.25
        tau = -k_theta * float(drone.pitch) - k_omega * float(drone.pitch_velocity)
        return self._thrust_from_u_tau(u0, tau)

    def _h_boost_down(self, drone: Drone) -> Tuple[float, float]:
        # decrease total thrust, keep level
        u0 = 0.38
        k_theta = 0.6
        k_omega = 0.45  # was 0.25
        tau = -k_theta * float(drone.pitch) - k_omega * float(drone.pitch_velocity)
        return self._thrust_from_u_tau(u0, tau)

    def _h_arrest_motion(self, drone: Drone) -> Tuple[float, float]:
        # Actively damp both linear and angular motion.
        # horizontal and vertical velocity
        vx = float(drone.velocity_x)
        vy = float(drone.velocity_y)
        
        # Calculate distance to target (for Position Hold)
        tx, ty = drone.get_next_target()
        dx = tx - drone.x

        # Horizontal damping via pitch control
        # --- LOW SPEED: POSITION HOLD (The Magnet) ---
        # If speed < 0.15, use dx to center the drone.
        if abs(vx) < 0.15:
            # Gentle nudge towards target
            # Max tilt 0.10 rad (5 degrees) so it doesn't fight too hard
            target_pitch = float(np.clip(0.6 * dx, -0.15, 0.15))
            u0 = 0.50  # Hover Thrust
        # --- HIGH SPEED: ANCHOR BRAKE (The Safety) ---
        # If speed > 0.15, ignore position, just stop the car.
        else:
            k_theta_vx = 0.8
            target_pitch = float(np.clip(-k_theta_vx * vx, -0.35, 0.35))  # +/- 20 degree
            u0 = 0.53

        # vertical damping
        k_vy = 0.1
        u = u0 - k_vy * vy

        # PD controller
        k_theta = 0.7 # was 0.9
        k_omega = 0.5 # was 0.35
        pitch_error = float(drone.pitch) - target_pitch
        tau = -k_theta * pitch_error - k_omega * float(drone.pitch_velocity)

        return self._thrust_from_u_tau(u, tau)


    def apply_action(self, action_id: int, drone: Drone) -> Tuple[float, float]:
        if action_id == self.HOVER_STABILISE:
            return self._h_hover_stabilise(drone)
        if action_id == self.TILT_LEFT:
            return self._h_tilt_left(drone)
        if action_id == self.TILT_RIGHT:
            return self._h_tilt_right(drone)
        if action_id == self.BOOST_UP:
            return self._h_boost_up(drone)
        if action_id == self.BOOST_DOWN:
            return self._h_boost_down(drone)
        if action_id == self.ARREST_MOTION:
            return self._h_arrest_motion(drone)
        raise ValueError(f"Invalid action_id: {action_id}")
    
    # ============================================================
    # Reward
    # ============================================================
    def compute_reward(self, drone: Drone, prev_dist: float) -> Tuple[float, float, int, Dict[str, float]]:
        r_parts = {
        "hit": 0.0,
        "progress": 0.0,
        "step": 0.0,
        "near_boundary": 0.0,
        "oob":0.0,
        }

        # Initialisation
        hit = 0
        curr_dist = self._distance_to_target(drone)
        reward = 0.0

        # ---------- Hit reward ----------
        if drone.has_reached_target_last_update:
            hit = 1
            r_parts["hit"] = self.R_hit
            reward += r_parts["hit"]
            return reward, curr_dist, hit, r_parts

        else:
            # ---------- Progress shaping ----------
            # sigma = 0.25      # distance scale of the boost (tune)
            # lam = 3.0         # how much stronger near target (tune)
            # w  = 1.0 + lam * np.exp(-prev_dist / sigma)   # larger near target
            
            dd = prev_dist - curr_dist # progress shaping
            
            progress = 0.0
            backwards = 0.0
            
            # if abs(dd) > 2e-4:
            if dd > 0:
                # progress = float(min(dd, self.progress_clip))  #clipped progress shaping for stability
                progress = dd
            else:
                backwards = -dd
            
            r_parts["progress"] = progress * self.k_progress - backwards * self.k_backwards
            reward += r_parts["progress"]
            
            # ---------- Timestep penalty ----------
            r_parts["step"] = -self.c_step
            reward += r_parts["step"]

            # ---------- Boundary proximity penalty ----------
            dist_to_boundary = self._distance_to_boundary(drone)
            if dist_to_boundary < self.b0:
                penalty_factor = (self.b0 - dist_to_boundary) / self.b0 
                r_parts["near_boundary"] = -self.near_boundary_penalty_scale * penalty_factor
                reward += r_parts["near_boundary"]

            return reward, curr_dist, hit, r_parts
    
    # ============================================================
    # Policy helpers
    # ============================================================
    def _argmax_tiebreak(self, q_row: np.ndarray) -> int: 
        """
        Argmax with random tie-breaking.
        When multiple actions have the same Q-value, choose randomly among them
        """ 
        m = np.max(q_row) 
        idx = np.flatnonzero(q_row == m) 
        return int(np.random.choice(idx))
    
    def _select_action(self, state_id: int, epsilon: float) -> int:
        if np.random.rand() < epsilon:
            # Random exploration
            return int(np.random.randint(self.n_actions))
        else:
            # Exploit: Pick best action
            return self._argmax_tiebreak(self.Q[state_id])
    
    def get_thrusts(self, drone: Drone) -> Tuple[float, float]:
        """
        Policy interface called by simulator (epsilon_eval=0.0 by default)
        """
        state_tuple = self.get_state_tuple(drone)
        state_id = self.last_state_id
        action = self._select_action(state_id, self.epsilon_eval)
        return self.apply_action(action, drone)
    
    # ============================================================
    # Curriculum Learning
    # ============================================================
    def _update_curriculum(self, ep_return: float, ep_hits: int) -> None:
        """Update curriculum stage based on recent performance"""
        self.recent_returns.append(ep_return)
        self.recent_hits.append(ep_hits)

        # Keep only last K episodes
        if len(self.recent_returns) > self.K:
            self.recent_returns = self.recent_returns[-self.K:]
            self.recent_hits = self.recent_hits[-self.K:]
        
        # not enough data yet
        if len(self.recent_returns) < self.K:
            return
        
        # already at last stage
        if self.stage_idx >= len(self.curriculum_stages) - 1:
            return
    
        # Check thresholds
        avg_hit = float(np.mean(self.recent_hits))
        avg_return = float(np.mean(self.recent_returns))

        # thresholds for moving from current stage to the next
        gate_hit = self.hit_thresholds[self.stage_idx]

        # Advance if hit threshold met
        if avg_hit >= gate_hit:
            self._advance_stage(avg_hit, avg_return)
    
    def _advance_stage(self, avg_hit=None, avg_return=None):
        """Advance to next curriculum stage"""
        self.stage_idx += 1
    
        # Reset history for new stage
        self.recent_returns = []
        self.recent_hits = []

        stage = self.curriculum_stages[self.stage_idx]
        self.epsilon = float(stage["eps_start"])
    
        print(f"\n{'='*60}")
        print(f"[Curriculum] Advanced to {stage['name']} (stage {self.stage_idx})")
        if avg_hit is not None and avg_return is not None:
            print(f"  Previous performance: {avg_hit:.2f} hits/ep, {avg_return:.1f} return/ep")
        print(f"  New epsilon: {self.epsilon:.3f}")
        print(f"{'='*60}\n")

    def _apply_stage_hyperparams(self) -> dict:
        """
        Apply hyperparameters for current curriculum stage
        """
        stage = self.curriculum_stages[self.stage_idx]
        
        # Update training params
        self.max_steps = int(stage["max_steps"])
        self.alpha = float(stage["alpha"])
        self.eps_start = float(stage["eps_start"])
        self.eps_decay = float(stage["eps_decay"])
        self.eps_min = float(stage["eps_min"])
        self.action_repeat = int(stage["action_repeat"])
        self.target_bounds = stage["target_bounds"]
        self.total_targets = int(stage["n"])
        self.boundary_penalty = int(stage["boundary_penalty"])
        return stage
   
    # ============================================================
    # Training
    # ============================================================
    def _append_episode_log(self, row: dict) -> None:
        with open(self.log_path, "a") as f:
            f.write(json.dumps(row) + "\n")
    
    def _append_episode_csv(self, log_dict):
        file_exists = os.path.isfile(self.csv_log_path)
        with open(self.csv_log_path, "a", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=log_dict.keys())
            if not file_exists:
                writer.writeheader()
            writer.writerow(log_dict)

    def train(self, num_episodes: int = 5000, save_every: int = 200, print_every: int = 50,) -> None:
        for ep in range(num_episodes):
            r_sums = {
                "hit": 0.0,
                "progress": 0.0,
                "step": 0.0,
                "near_boundary": 0.0,
                "oob": 0.0,
            }
            done_reason = ""

            self.episode_count += 1

            # Apply curriculum stage settings
            if self.target_mode == "curriculum":
                stage = self._apply_stage_hyperparams()
            
            # Initialise episode
            drone = self.init_drone()
            total_return = 0.0
            ep_hits = 0
            crashed = 0
            steps_taken = 0
            done = False
            hit_step_counts_ep = []
            steps_since_last_hit = 0
            progress_step_count = 0
            

            # Initialise distance trackers
            prev_dist = self._distance_to_target(drone)

            # Initialise state and action
            state_tuple = self.get_state_tuple(drone)
            state_id = self.last_state_id
            action = self._select_action(state_id, self.epsilon)
            
            # Log raw state and state bin
            vx_vals, vy_vals, th_vals, om_vals = [], [], [], []
            dx_bins = Counter();dy_bins = Counter();vx_bins = Counter(); vy_bins = Counter(); th_bins = Counter(); om_bins = Counter()       
            # optional: focus near target only
            near_vx_vals, near_vy_vals, near_th_vals, near_om_vals = [], [], [], []
            near_dist = 0.15   # define “near target” window
            sample_every = 10  # log every N simulator steps to keep it light

            # =========================================================
            # Main episode loop
            # =========================================================
            while not done and steps_taken < self.max_steps:
                # =====================================================
                # Execute current action for action_repeat steps
                # =====================================================
                accumulated_reward = 0.0 # Accumulator for rewards during action repetition
                macro_len = 0 # save actually repeated lenth
                dist_sum = 0.0


                for repeat_step in range(self.action_repeat):
                    # Execute action in simulator
                    thrusts = self.apply_action(action, drone)
                    drone.set_thrust(thrusts)
                    drone.step_simulation(self.dt)
                    steps_taken += 1
                    macro_len += 1
                    steps_since_last_hit += 1

                    # update distance & compute immediate reward
                    reward, curr_dist, hit, r_parts = self.compute_reward(drone, prev_dist)
                    if r_parts["progress"] > 0:
                        progress_step_count += 1
                    for k in r_sums:
                        r_sums[k] += r_parts.get(k, 0.0)
                    accumulated_reward += reward
                    
                    # log raw states and state bins
                    if (steps_taken % sample_every) == 0:
                        tx, ty = drone.get_next_target()
                        dx = tx - drone.x
                        dy = ty - drone.y
                        vx = float(drone.velocity_x)
                        vy = float(drone.velocity_y)
                        th = float(drone.pitch)
                        om = float(drone.pitch_velocity)
                        
                        vx_vals.append(vx); vy_vals.append(vy); th_vals.append(th); om_vals.append(om)
                        
                        dx_bins[self._bin_7(dx, self.dx1, self.dx2, self.dx3)] += 1
                        dy_bins[self._bin_7(dy, self.dy1, self.dy2, self.dy3)] += 1
                        vx_bins[self._bin_3(vx, self.vx0)] += 1
                        vy_bins[self._bin_3(vy, self.vy0)] += 1
                        th_bins[self._bin_3(th, self.theta0)] += 1
                        om_bins[self._bin_3(om, self.omega0)] += 1
                        
                        if curr_dist < near_dist:
                            near_vx_vals.append(vx); near_vy_vals.append(vy); near_th_vals.append(th); near_om_vals.append(om)

                    # Track target hits
                    if hit == 1:
                        ep_hits += 1
                        hit_step_counts_ep.append(steps_since_last_hit)
                        steps_since_last_hit = 0 # reset steps count to hit target
                        prev_dist = self._distance_to_target(drone)
                    else:
                        prev_dist = curr_dist
                    # =============================================
                    # Check terminal conditions
                    # =============================================
                    
                    # Out of bounds - crash
                    if self._out_of_bounds(drone):
                        accumulated_reward -= self.boundary_penalty
                        r_sums["oob"] -= self.boundary_penalty
                        crashed = 1
                        done_reason = "crash"
                        done = True
                        break
                    
                    # All targets reached - success!
                    if self._all_targets_done(drone):
                        done_reason = "success"
                        done = True
                        break
                    
                    # Episode length limit
                    if steps_taken >= self.max_steps:
                        done = True
                        done_reason = "max_steps"
                        break
                
                # Update total return
                total_return += accumulated_reward

                # =====================================================
                # SARSA Q-TABLE UPDATE (once per repeated macro-action)
                # =====================================================
                if done:
                    # Terminal state：Q(st​,at​)←Q(st​,at​)+α[Rt​−Q(st​,at​)]
                    Rt = accumulated_reward
                    self.Q[state_id, action] += self.alpha * (Rt - self.Q[state_id, action])
                    break
                else:
                    # Non-terminal: get next state and select next action
                    next_state_tuple = self.get_state_tuple(drone)
                    next_state_id = self.last_state_id
                    next_action = self._select_action(next_state_id, self.epsilon)
                    
                    
                    effective_gamma = self.gamma ** macro_len
                    # SARSA update: Q(s,a) ← Q(s,a) + α[r + γQ(s',a') - Q(s,a)]
                    Rt = accumulated_reward + effective_gamma * self.Q[next_state_id, next_action]
                    
                    # Q learning update: Q(s,a) ← Q(s,a) + α[r + γmax​Q(s',a') - Q(s,a)]
                    # best_next = np.max(self.Q[next_state_id])
                    # Rt = accumulated_reward + effective_gamma * best_next
                    
                    self.Q[state_id, action] += self.alpha * (Rt - self.Q[state_id, action])

                    # Move to next state-action pair
                    state_id = next_state_id
                    action = next_action

            # =========================================================
            # End of episode: bookkeeping
            # =========================================================
            
            # Update curriculum based on performance
            if self.target_mode == "curriculum":
                self._update_curriculum(total_return, ep_hits)
            
            # Decay epsilon
            self.epsilon = max(self.eps_min, self.epsilon * self.eps_decay)
            
            # average steps taken to hit targets
            avg_steps_per_hit = (float(np.mean(hit_step_counts_ep)) if hit_step_counts_ep else None)
            med_steps_per_hit = (float(np.median(hit_step_counts_ep)) if hit_step_counts_ep else None)
            
            # Save per-episode log (JSONL)
            if (ep + 1) % self.flush_log_every == 0:
                q_mean = float(np.mean(self.Q))
                q_std  = float(np.std(self.Q))
                q_max  = float(np.max(self.Q))
                q_min  = float(np.min(self.Q))
                self._append_episode_log({
                    "ep": int(ep + 1),
                    "stage": int(self.stage_idx),
                    "done_reason": done_reason,
                    "action_repeat": int(self.action_repeat),
                    "hits": int(ep_hits),
                    "crash": int(crashed),
                    "return": float(total_return),
                    "steps": int(steps_taken),
                    "avg_steps_per_hit": avg_steps_per_hit,
                    "med_steps_per_hit": med_steps_per_hit,
                    "epsilon": float(self.epsilon),
                    "alpha": float(self.alpha),
                    "q_mean": q_mean,
                    "q_std": q_std,
                    "q_min": q_min,
                    "q_max": q_max,
                    "r_hit": r_sums["hit"],
                    "r_progress": r_sums["progress"],
                    "progress_steps": progress_step_count,
                    "r_step": r_sums["step"],
                    "r_near_boundary": r_sums["near_boundary"],
                    "r_oob": r_sums["oob"],
                    "mean_return_per_step": total_return / max(1, steps_taken),
                    "dx_bin_counts": dict(dx_bins),
                    "dy_bin_counts": dict(dy_bins),
                    "vx_bin_counts": dict(vx_bins),
                    "vy_bin_counts": dict(vy_bins),
                    "theta_bin_counts": dict(th_bins),
                    "omega_bin_counts": dict(om_bins),
                    "vx_abs_p50": float(np.percentile(np.abs(vx_vals), 50)) if vx_vals else None,
                    "vy_abs_p50": float(np.percentile(np.abs(vy_vals), 50)) if vy_vals else None,
                    "theta_abs_p50": float(np.percentile(np.abs(th_vals), 50)) if th_vals else None,
                    "omega_abs_p50": float(np.percentile(np.abs(om_vals), 50)) if om_vals else None,
                    "vx_abs_p90": float(np.percentile(np.abs(vx_vals), 90)) if vx_vals else None,
                    "vy_abs_p90": float(np.percentile(np.abs(vy_vals), 90)) if vy_vals else None,
                    "theta_abs_p90": float(np.percentile(np.abs(th_vals), 90)) if th_vals else None,
                    "omega_abs_p90": float(np.percentile(np.abs(om_vals), 90)) if om_vals else None,
                    "near_vx_abs_p90": float(np.percentile(np.abs(near_vx_vals), 90)) if near_vx_vals else None,
                    "near_vy_abs_p90": float(np.percentile(np.abs(near_vy_vals), 90)) if near_vy_vals else None,
                    "near_theta_abs_p90": float(np.percentile(np.abs(near_th_vals), 90)) if near_th_vals else None,
                    "near_omega_abs_p90": float(np.percentile(np.abs(near_om_vals), 90)) if near_om_vals else None,
                })
                self._append_episode_csv({
                    "ep": int(ep + 1),
                    "stage": int(self.stage_idx),
                    "done_reason": done_reason,
                    "action_repeat": int(self.action_repeat),
                    "hits": int(ep_hits),
                    "crash": int(crashed),
                    "return": float(total_return),
                    "steps": int(steps_taken),
                    "avg_steps_per_hit": avg_steps_per_hit,
                    "med_steps_per_hit": med_steps_per_hit,
                    "epsilon": float(self.epsilon),
                    "alpha": float(self.alpha),
                    "q_mean": q_mean,
                    "q_std": q_std,
                    "q_min": q_min,
                    "q_max": q_max,
                    "r_hit": r_sums["hit"],
                    "r_progress": r_sums["progress"],
                    "progress_steps": progress_step_count,
                    "r_step": r_sums["step"],
                    "r_near_boundary": r_sums["near_boundary"],
                    "r_oob": r_sums["oob"],
                    "mean_return_per_step": total_return / max(1, steps_taken),
                    "dx_bin_counts": dict(dx_bins),
                    "dy_bin_counts": dict(dy_bins),
                    "vx_bin_counts": dict(vx_bins),
                    "vy_bin_counts": dict(vy_bins),
                    "theta_bin_counts": dict(th_bins),
                    "omega_bin_counts": dict(om_bins),
                    "vx_abs_p50": float(np.percentile(np.abs(vx_vals), 50)) if vx_vals else None,
                    "vy_abs_p50": float(np.percentile(np.abs(vy_vals), 50)) if vy_vals else None,
                    "theta_abs_p50": float(np.percentile(np.abs(th_vals), 50)) if th_vals else None,
                    "omega_abs_p50": float(np.percentile(np.abs(om_vals), 50)) if om_vals else None,
                    "vx_abs_p90": float(np.percentile(np.abs(vx_vals), 90)) if vx_vals else None,
                    "vy_abs_p90": float(np.percentile(np.abs(vy_vals), 90)) if vy_vals else None,
                    "theta_abs_p90": float(np.percentile(np.abs(th_vals), 90)) if th_vals else None,
                    "omega_abs_p90": float(np.percentile(np.abs(om_vals), 90)) if om_vals else None,
                    "near_vx_abs_p90": float(np.percentile(np.abs(near_vx_vals), 90)) if near_vx_vals else None,
                    "near_vy_abs_p90": float(np.percentile(np.abs(near_vy_vals), 90)) if near_vy_vals else None,
                    "near_theta_abs_p90": float(np.percentile(np.abs(near_th_vals), 90)) if near_th_vals else None,
                    "near_omega_abs_p90": float(np.percentile(np.abs(near_om_vals), 90)) if near_om_vals else None,
                })

            # Save checkpoint
            if (ep + 1) % save_every == 0:
                self.save()
            
            # Print progress
            if (ep + 1) % print_every == 0 or (ep + 1) == 1:
                q_mean = float(np.mean(self.Q))
                q_std = float(np.std(self.Q))
                q_max = float(np.max(self.Q))
                q_min = float(np.min(self.Q))
                
                print(
                    f"Ep {ep+1:4d}/{num_episodes} | "
                    f"stage={self.stage_idx} | "
                    f"hits={ep_hits:2d} | "
                    f"crash={crashed} | "
                    f"return={total_return:8.2f} | "
                    f"epsilon={self.epsilon:.3f} | "
                    f"q_mean={q_mean:.3f} q_min={q_min:.1f} q_max={q_max:.1f}"
                )

        # Final save
        self.save()
        print("\n[Training] Complete!")

    # ============================================================
    # Persistence
    # ============================================================
    def save(self):
        try:
            np.save(self.q_path, self.Q)
        except Exception as e:
            print(f"[save] failed: {e}")

    def load(self, q_path=None, target_mode=None):
        if target_mode is not None:
            self.target_mode = target_mode
        if q_path is not None:
            self.q_path = q_path
        try:
            if os.path.exists(self.q_path):
                arr = np.load(self.q_path)
                if arr.shape == (self.n_states, self.n_actions):
                    self.Q = arr.astype(np.float32)
                    print(f"[load] loaded Q-table from {self.q_path}")
                else:
                    print(f"[load] wrong shape {arr.shape}, expected {(self.n_states, self.n_actions)}")
        except Exception as e:
            print(f"[load] failed: {e}")

    