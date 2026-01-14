import numpy as np
import json
from flight_controller import FlightController
from drone import Drone


class Policy():
    def __init__(self, input_size, output_size):
        # Initialize weights to zeros
        self.weights = np.zeros((input_size, output_size))        
    
    def forward(self, state):
        return state @ self.weights  # Linear model
    
    def backward(self):
        pass
    
    def update(self):
        """
        Updates policy weights from returns using REINFORCE
        """
        pass

class ContinuousLinearController(FlightController):
    def __init__(self):
                
        # Load config
        with open('continuous_linear_config.json', 'r') as f:
            self.config = json.load(f)
        
        # Initialise policy
        input_size = self.config['input_size']
        output_size = self.config['output_size']
        self.policy = Policy(input_size, output_size)

    def get_state(self, drone: Drone):
        """
        Get state from drone object
        """
        # Get drone state parameters
        target = drone.get_next_target()
        dx = (target[0] - drone.x)
        dy = (target[1] - drone.y)
        vx = drone.velocity_x
        vy = drone.velocity_y
        pitch = drone.pitch
        pitch_vel = drone.pitch_velocity
        bias = 1.0  # Bias for linear model
        
        # Normalise - to help stabilise learning
        state = np.array([dx, dy, vx, vy, pitch, pitch_vel, bias])
        norms = np.array([2.0, 2.0, 5.0, 5.0, 1.0, 1.0, 1.0])
        state = state / norms
        
        return state
    
    def get_action(self, state, mode='test'):
        """
        Get action from policy
            mode='train': Add gaussian noise (exploration)
            mode='test': Use mean directly (exploitation)
        Return action pair: [total_thrust, roll]
        """
        # Get the mean (mu) from the policy
        mu = self.policy.forward(state)
        
        if mode == 'train':
            sigma = self.config['hyperparameters']['sigma']
            action = np.random.normal(loc=mu, scale=sigma)
            return action
        else:
            return mu
    
    def convert_action_to_thrust(self, action):
        """
        Convert action to thrusts
            0: Total thrust
            1: Roll
        """
        t_total = action[0]
        roll = action[1]
        
        # Mixing logic to convert to thrusts
        left = (t_total + roll) / 2.0
        right = (t_total - roll) / 2.0
        thrusts = (left, right)
        
        # Clip to ensure valid motor values [0, 1], since sampling during training can result in invalid values
        thrusts = np.clip(thrusts, 0, 1)
        
        return thrusts

    def get_thrusts(self, drone: Drone):
        """
        Get action from model and convert to thrusts
        """
        state = self.get_state(drone)
        action = self.get_action(state, mode='test')
        return self.convert_action_to_thrust(action)

    def get_reward(self, drone: Drone):
        """
        Calculate rewards for the episode
        """
        cfg = self.config['rewards']
        reward = 0.0
        
        # Distance Penalty
        target = drone.get_next_target()
        dist = np.linalg.norm([target[0] - drone.x, target[1] - drone.y])
        reward -= dist * cfg['distance_weight']
        
        # Hit Bonus
        if drone.has_reached_target_last_update:
            reward += cfg['hit_bonus']
            
        return reward
    
    def get_returns(self, rewards):
        """
        Monty Carlo returns
            Reverse discounted sum
            Center mean (baseline subtrraction) - need good=positive, bad=negative for REINFORCE
            Normalise by std - keeps returns in a range to stabilise gradients
        """
        gamma = self.config['hyperparameters']['gamma']
        returns = []
        G = 0
        
        # Iterate backwards: G_t = r_t + gamma * G_{t+1}
        for r in reversed(rewards):
            G = r + gamma * G
            returns.insert(0, G)
        
        # Normalization 
        returns = np.array(returns)
        returns = (returns - returns.mean()) / (returns.std() + 1e-8)
            
        return returns

    def train(self):
        """
        Run episode loop and update policy
            Loop over episodes
                Initialise environment
                Get States and Actions
                Step through environment
                Calculate rewards
            Calculate returns
            Update policy
        """
        print(f"Starting Training: {self.config['experiment_name']}")
        
        n_episodes = self.config['hyperparameters']['n_episodes']
        max_steps = self.config['hyperparameters']['max_steps']

        for episode in range(n_episodes):
            
            states = []
            actions = []
            rewards = []
            drone = self.init_drone()
            
            for step in range(max_steps):
                
                # Get states and actions
                state = self.get_state(drone)
                action = self.get_action(state, mode='train')
                
                # Step environment
                thrust = self.convert_action_to_thrust(action)
                drone.set_thrust(thrust)
                drone.step_simulation(self.get_time_interval())
                
                # Get rewards
                reward = self.get_reward(drone)
                
                # Store data
                states.append(state)
                actions.append(action)
                rewards.append(reward)

            # Calculate returns
            returns = self.get_returns(rewards)
            
            # Update policy
            self.policy.update(states, actions, returns)
            
            # Log
            if episode % 50 == 0:
                print(f"Episode {episode}: Total Reward: {sum(rewards):.2f}")
                
        self.save()

    def save(self):
        
        pass

    def load(self):
        pass