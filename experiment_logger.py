import csv
import os
import math
from datetime import datetime

class ExperimentLogger:
    def __init__(self, filename="results.csv"):
        self.filename = filename
        self.current_attempt = None
        
        # specific columns requested
        self.header = [
            "Timestamp", "Model", "Param_Setting", "Episode", 
            "Target_Index",
            "Start_X", "Start_Y", 
            "Target_X", "Target_Y", 
            "Distance", 
            "Steps_Taken", "Total_Thrust", "Success"
        ]

        # Initialize CSV with header if new file
        if not os.path.exists(self.filename):
            with open(self.filename, mode='w', newline='') as f:
                writer = csv.writer(f)
                writer.writerow(self.header)

    def start_target_attempt(self, drone, target_index):
        """Called immediately when a drone starts chasing a NEW target."""
        self.current_attempt = TargetAttempt(drone, target_index)

    def log_step(self, drone):
        """Called every simulation frame."""
        if self.current_attempt:
            self.current_attempt.add_step(drone)

    def finish_target_attempt(self, model_name, param_setting, episode_num, success=True):
        """Called when target is hit OR max steps reached."""
        if not self.current_attempt:
            return
        
        stats = self.current_attempt.get_stats(success)
        row = [
            datetime.now().strftime("%H:%M:%S"),
            model_name,
            param_setting,
            episode_num,
            stats['target_idx'],
            f"{stats['start_x']:.2f}",
            f"{stats['start_y']:.2f}",
            f"{stats['target_x']:.2f}",
            f"{stats['target_y']:.2f}",
            f"{stats['distance']:.2f}",
            stats['steps'],
            f"{stats['thrust']:.2f}",
            int(success)
        ]

        # save row to file - for long runs this avoids data loss
        with open(self.filename, mode='a', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(row)
            
        # Clear current attempt
        self.current_attempt = None

# experiment_logger.py (Partial Update)

class TargetAttempt:
    """Helper class to hold data for ONE target attempt."""
    # UPDATE: Add target_index here to match the call in ExperimentLogger
    def __init__(self, drone, target_index=0):
        # log starting location of drone
        self.start_x = float(drone.x)
        self.start_y = float(drone.y)
        
        # log target location and distance from drone
        if len(drone.target_coordinates) > 0:
            t = drone.target_coordinates[0]
            self.target_x = float(t[0])
            self.target_y = float(t[1])
        else:
            self.target_x, self.target_y = 0.0, 0.0

        # Calculate Euclidean distance
        self.distance = math.sqrt((self.target_x - self.start_x)**2 + (self.target_y - self.start_y)**2)
            
        # initialise stats
        self.steps = 0
        self.thrust = 0.0  # Renamed from total_thrust to match your file
        self.target_idx = target_index # Store the index

    def add_step(self, drone):
        self.steps += 1
        self.thrust += float(drone.thrust_left + drone.thrust_right)

    def get_stats(self, success):
        return {
            "target_idx": self.target_idx, # Include this in the stats
            "start_x": self.start_x,
            "start_y": self.start_y,
            "target_x": self.target_x,
            "target_y": self.target_y,
            "distance": self.distance,
            "steps": self.steps,
            "thrust": self.thrust,
            "success": success
        }