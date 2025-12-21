import os
import numpy as np
from heuristic_controller import HeuristicController
from experiment_logger import ExperimentLogger

# --- CONFIGURATION ---
LOG_FILE = os.path.join(os.path.dirname(__file__), "results", "heuristic_ky_experiment.csv")
CONTROLLER = HeuristicController
logger = ExperimentLogger(LOG_FILE)

# Testing different 'ky' (Vertical P-Gain) values
param_values = [0.5, 1.0, 2.0, 5.0]  
EPISODES_PER_PARAM = 1
MAX_STEPS_PER_EPISODE = 5000
N_TARGETS = 50

# Run experiments (iterate parameter values)
for ky_val in param_values:
    param_str = f"ky={ky_val}"
    print(f"Testing: {param_str}")
    
    for episode in range(EPISODES_PER_PARAM):
        # Initialise controller and drone
        controller = CONTROLLER()
        controller.ky = ky_val  # Overwrite parameter(s)
        drone = controller.init_drone(mode='random', num_targets=N_TARGETS)
        
        # log initial target
        target_counter = 0
        logger.start_target_attempt(drone, target_index=target_counter)
        
        # 4. Run Simulation Steps
        for step in range(MAX_STEPS_PER_EPISODE):
            # Get action
            thrusts = controller.get_thrusts(drone)
            
            # Step simulation
            drone.set_thrust(thrusts)
            drone.step_simulation(controller.get_time_interval())
            
            # log step
            logger.log_step(drone)
            
            # check if target reached
            if drone.has_reached_target_last_update:
                # log the target attempt and reset
                logger.finish_target_attempt(
                    model_name=controller.__class__.__name__, 
                    param_setting=param_str, 
                    episode_num=episode, 
                    success=True
                )
                target_counter += 1
                
                # reset target attempt, or exit if no more targets
                if len(drone.target_coordinates) > 0:
                    logger.start_target_attempt(drone, target_index=target_counter)
                else:
                    break
        
        # If the loop finished but we were still chasing a target (timeout), log anyway with success=0
        if logger.current_attempt is not None:
             logger.finish_target_attempt(
                model_name=controller.__class__.__name__, 
                param_setting=param_str, 
                episode_num=episode, 
                success=False
            )

print(f"Sweep Complete. Data saved to {LOG_FILE}")