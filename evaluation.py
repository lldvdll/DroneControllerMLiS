"""
Evaluation Module for SARSA Drone Controller

Provides comprehensive evaluation metrics including:
- Success rate and crash rate
- Target completion statistics
- Stability and control metrics
- Path efficiency analysis
"""

import json
import math
import numpy as np
from typing import Dict, Any, List, Tuple, Optional
from drone import Drone
from SARSA_controller import CustomController


# ============================================================
# Helper Functions
# ============================================================

def _out_of_bounds(drone: Drone, bounds: Tuple[float, float, float, float]) -> bool:
    """Check if drone is outside specified bounds"""
    xmin, xmax, ymin, ymax = bounds
    return (drone.x < xmin or drone.x > xmax or 
            drone.y < ymin or drone.y > ymax)


def _all_targets_done(drone: Drone) -> bool:
    """Check if all targets have been reached"""
    try:
        return len(drone.target_coordinates) == 0
    except (AttributeError, TypeError):
        return False


def _safe_stats(values: List[float]) -> Dict[str, float]:
    """Compute mean and std, handling empty lists"""
    if len(values) == 0:
        return {"mean": float("nan"), "std": float("nan")}
    arr = np.asarray(values, dtype=float)
    return {
        "mean": float(np.mean(arr)), 
        "std": float(np.std(arr, ddof=0))
    }


def _median_iqr(values: List[float]) -> Dict[str, float]:
    """Compute median and interquartile range"""
    if len(values) == 0:
        return {
            "median": float("nan"), 
            "q25": float("nan"), 
            "q75": float("nan")
        }
    arr = np.asarray(values, dtype=float)
    return {
        "median": float(np.median(arr)),
        "q25": float(np.quantile(arr, 0.25)),
        "q75": float(np.quantile(arr, 0.75)),
    }


# ============================================================
# Evaluation Function
# ============================================================

def evaluate(
    controller: CustomController,
    eval_mode: str = "fixed",
    n_episodes: int = 30,
    seed: int = 0,
    max_steps: Optional[int] = None,
    bounds: Optional[Tuple[float, float, float, float]] = None,
    dt: Optional[float] = None,
    return_mode="report"  # "report" or "return"
) -> Dict[str, Any]:
    """
    Evaluate controller with greedy policy (no exploration)
    
    Args:
        controller: The controller to evaluate
        eval_mode: Target mode ("fixed", "random", or "curriculum")
        n_episodes: Number of evaluation episodes
        seed: Random seed for reproducibility
        max_steps: Maximum steps per episode (default: controller.max_steps)
        bounds: World bounds (default: controller.full_bounds)
        dt: Simulation timestep (default: controller.dt)
    
    Returns:
        Dictionary containing evaluation metrics
    """
    # Save original settings
    old_mode = controller.target_mode
    
    # Set evaluation mode
    controller.target_mode = eval_mode
    
    # Use defaults if not specified
    if dt is None:
        dt = controller.dt
    if bounds is None:
        bounds = controller.full_bounds
    if max_steps is None:
        max_steps = controller.max_steps
    
    # Metric storage
    success_flags: List[int] = []
    targets_reached: List[int] = []
    crashes: List[int] = []
    ep_steps: List[int] = []
    time_to_target_all: List[int] = []
    
    # Stability metrics
    pitch_mean_abs: List[float] = []
    pitch_max_abs: List[float] = []
    thrust_saturation: List[float] = []
    osc_sign_changes: List[int] = []
    pitch_var: List[float] = []
    
    # Path efficiency
    path_efficiency: List[float] = []
    
    # Set random seed
    np.random.seed(seed)
    
    # Run evaluation episodes
    for ep in range(n_episodes):
        # Episode-specific seed
        np.random.seed(seed * 10000 + ep)
        
        # Initialize episode
        drone = controller.init_drone()
        n_init_targets = len(drone.target_coordinates)
        done = False
        crashed = 0
        
        # Time-to-target tracking
        last_hit_step = 0
        per_ep_ttt: List[int] = []
        
        # Stability tracking
        pitch_abs_trace: List[float] = []
        pitch_trace: List[float] = []
        omega_trace: List[float] = []
        sat_count = 0
        
        # Path tracking
        travel = 0.0
        straight = 0.0
        prev_x, prev_y = float(drone.x), float(drone.y)
        seg_start = (prev_x, prev_y)
        seg_target = drone.get_next_target()
        
        # Oscillation tracking
        prev_omega = float(drone.pitch_velocity)
        sign_changes = 0
        
        # Episode simulation
        steps_taken = 0
        rep = int(getattr(controller, "action_repeat_eval", 1))
        rep = max(rep, 1)
        
        while steps_taken < max_steps:
            # Get action from controller
            thrusts = controller.get_thrusts(drone)
            
            # Apply the same thrusts for rep simulator steps
            for t in range(rep):
                if steps_taken >= max_steps:
                    break
                
                # Check thrust saturation
                t1, t2 = thrusts
                if (abs(t1) < 1e-9 or abs(t1 - 1.0) < 1e-9 or 
                    abs(t2) < 1e-9 or abs(t2 - 1.0) < 1e-9):
                    sat_count += 1
            
                # Apply action
                drone.set_thrust(thrusts)
                drone.step_simulation(dt)
                steps_taken += 1
            
                # Track path length
                x, y = float(drone.x), float(drone.y)
                travel += math.hypot(x - prev_x, y - prev_y)
                prev_x, prev_y = x, y
                
                # Track stability
                pitch_abs_trace.append(abs(float(drone.pitch)))
                pitch_trace.append(float(drone.pitch))
                omega = float(drone.pitch_velocity)
                omega_trace.append(omega)
                
                # Track oscillations (sign changes in angular velocity)
                if omega != 0.0 and prev_omega != 0.0:
                    if (omega > 0) != (prev_omega > 0):
                        sign_changes += 1
                prev_omega = omega
            
                # Check if target was hit
                if drone.has_reached_target_last_update:
                    hit_step = steps_taken
                    per_ep_ttt.append(hit_step - last_hit_step)
                    last_hit_step = hit_step
                
                    # Update straight-line distance
                    if seg_target is not None:
                        tx, ty = seg_target
                        sx, sy = seg_start
                        straight += math.hypot(tx - sx, ty - sy)
                
                    # Update segment for next target
                    seg_start = (float(drone.x), float(drone.y))
                    seg_target = drone.get_next_target()
            
                # Check terminal conditions
                if _out_of_bounds(drone, bounds):
                    crashed = 1
                    done = True
                    break
            
                if _all_targets_done(drone):
                    done = True
                    break
            
            if done:
                break
        
        # Episode statistics
        ep_steps.append(steps_taken)
        crashes.append(crashed)
        
        # Targets reached
        n_left = len(drone.target_coordinates)
        reached = n_init_targets - n_left
        targets_reached.append(reached)
        
        # Success: all targets reached without crashing
        success = int((n_left == 0) and (crashed == 0))
        success_flags.append(success)
        
        # Store time-to-target
        time_to_target_all.extend(per_ep_ttt)
        
        # Stability metrics
        if pitch_abs_trace:
            pitch_mean_abs.append(float(np.mean(pitch_abs_trace)))
            pitch_max_abs.append(float(np.max(pitch_abs_trace)))
        else:
            pitch_mean_abs.append(float("nan"))
            pitch_max_abs.append(float("nan"))
        
        if steps_taken > 0:
            thrust_saturation.append(float(sat_count / steps_taken))
        else:
            thrust_saturation.append(float("nan"))
        
        osc_sign_changes.append(int(sign_changes))
        
        if pitch_trace:
            pitch_var.append(float(np.var(np.asarray(pitch_trace))))
        else:
            pitch_var.append(float("nan"))
        
        # Path efficiency
        if straight > 1e-9:
            path_efficiency.append(float(travel / straight))
        else:
            path_efficiency.append(float("nan"))
            
    # Compile evaluation report
    report: Dict[str, Any] = {
        "config": {
            "eval_mode": eval_mode,
            "n_episodes": n_episodes,
            "seed": seed,
            "bounds": bounds,
            "dt": dt,
            "max_steps": max_steps,
            "action_repeat": int(getattr(controller, "action_repeat_eval", 1)),
        },
        "core": {
            "success_rate": float(np.mean(success_flags)) if success_flags else float("nan"),
            "targets_reached": _safe_stats([float(x) for x in targets_reached]),
            "crash_rate": float(np.mean(crashes)) if crashes else float("nan"),
            "steps": _safe_stats([float(x) for x in ep_steps]),
            "time_to_target": _median_iqr([float(x) for x in time_to_target_all]),
            "path_efficiency": _safe_stats([x for x in path_efficiency if not np.isnan(x)]),
        },
        "stability": {
            "mean_abs_pitch": _safe_stats(pitch_mean_abs),
            "max_abs_pitch": _safe_stats(pitch_max_abs),
            "thrust_saturation_rate": _safe_stats(thrust_saturation),
            "osc_sign_changes": _safe_stats([float(x) for x in osc_sign_changes]),
            "pitch_variance": _safe_stats(pitch_var),
        },
        "raw": {
            "success_flags": success_flags,
            "targets_reached": targets_reached,
            "crashes": crashes,
            "ep_steps": ep_steps,
        }
    }
    
    # Restore original settings
    controller.target_mode = old_mode
        
    # Return mode
    if return_mode == 'report':
        return report

    if return_mode == 'metrics':
        metrics = {
            'success_rate': report['core']['success_rate'],
            'targets_reached': report['core']['targets_reached']['mean'],
            'crash_rate': report['core']['crash_rate'],
            'path_efficiency': report['core']['path_efficiency']['mean'],
            'time_to_target': report['core']['time_to_target']['median']
        }
        return metrics


# ============================================================
# Reporting Functions
# ============================================================

def print_report(report: Dict[str, Any], title: str = "Evaluation") -> None:
    """
    Print formatted evaluation report
    
    Args:
        report: Evaluation report dictionary
        title: Title for the report
    """
    core = report["core"]
    stab = report["stability"]
    
    def fmt_ms(d: Dict[str, float]) -> str:
        """Format mean ± std"""
        return f'{d["mean"]:.3f} ± {d["std"]:.3f}'
    
    print(f"\n{'=' * 60}")
    print(f"{title:^60}")
    print('=' * 60)
    
    # Core metrics
    print("\nCore Metrics:")
    print(f'  Success rate:      {core["success_rate"]:.3f}')
    print(f'  Crash rate:        {core["crash_rate"]:.3f}')
    print(f'  Targets reached:   {fmt_ms(core["targets_reached"])}')
    print(f'  Steps per episode: {fmt_ms(core["steps"])}')
    
    ttt = core["time_to_target"]
    print(f'  Time to target (steps):')
    print(f'    Median: {ttt["median"]:.1f}')
    print(f'    Q25-Q75: [{ttt["q25"]:.1f}, {ttt["q75"]:.1f}]')
    
    pe = core["path_efficiency"]
    print(f'  Path efficiency (travel/straight): {pe["mean"]:.3f} ± {pe["std"]:.3f}')
    
    # Stability metrics
    print("\nStability & Control:")
    print(f'  Mean |pitch|:           {fmt_ms(stab["mean_abs_pitch"])}')
    print(f'  Max |pitch|:            {fmt_ms(stab["max_abs_pitch"])}')
    print(f'  Thrust saturation:      {fmt_ms(stab["thrust_saturation_rate"])}')
    print(f'  Angular vel. sign changes: {fmt_ms(stab["osc_sign_changes"])}')
    print(f'  Pitch variance:         {fmt_ms(stab["pitch_variance"])}')
    
    print('=' * 60)


def save_report(report: Dict[str, Any], path: str) -> None:
    """
    Save evaluation report to JSON file
    
    Args:
        report: Evaluation report dictionary
        path: Output file path
    """
    try:
        with open(path, "w", encoding="utf-8") as f:
            json.dump(report, f, indent=2, ensure_ascii=False)
        print(f"[Eval] Report saved to {path}")
    except Exception as e:
        print(f"[Error] Failed to save report: {e}")


# ============================================================
# Additional Analysis Functions
# ============================================================

def compare_evaluations(reports: List[Dict[str, Any]], labels: List[str]) -> None:
    """
    Print comparison of multiple evaluation reports
    
    Args:
        reports: List of evaluation report dictionaries
        labels: Labels for each report
    """
    print(f"\n{'=' * 80}")
    print(f"{'COMPARISON':^80}")
    print('=' * 80)
    print(f"{'Metric':<30}", end='')
    for label in labels:
        print(f"{label:>15}", end='')
    print()
    print('-' * 80)
    
    # Success rate
    print(f"{'Success Rate':<30}", end='')
    for report in reports:
        val = report["core"]["success_rate"]
        print(f"{val:>15.3f}", end='')
    print()
    
    # Crash rate
    print(f"{'Crash Rate':<30}", end='')
    for report in reports:
        val = report["core"]["crash_rate"]
        print(f"{val:>15.3f}", end='')
    print()
    
    # Targets reached
    print(f"{'Targets Reached (mean)':<30}", end='')
    for report in reports:
        val = report["core"]["targets_reached"]["mean"]
        print(f"{val:>15.3f}", end='')
    print()
    
    # Path efficiency
    print(f"{'Path Efficiency (mean)':<30}", end='')
    for report in reports:
        val = report["core"]["path_efficiency"]["mean"]
        print(f"{val:>15.3f}", end='')
    print()
    
    print('=' * 80)


if __name__ == "__main__":
    # Example usage
    print("This module is meant to be imported.")
    print("Use train.py with --eval-after flag for evaluation.")