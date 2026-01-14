import numpy as np
import json
from flight_controller import FlightController
from drone import Drone
import matplotlib.pyplot as plt
import os

BASE_PATH = os.path.join(
    os.path.dirname(__file__), 
    "experiments", 
    "continuous_linear"
    )

class ExperimentLogger:
    """
    Diagnostics for training process
        Keeps track of rewards, returns, and gradients
        Generates plots
    """
    def __init__(self, experiment_name):
        self.episode_rewards = []
        self.episode_returns = []
        self.gradient_norms = []
        self.filepath = os.path.join(BASE_PATH, f"{experiment_name}_training_log")

    def log_episode(self, total_reward, discounted_return, grad_norm=None):
        self.episode_rewards.append(total_reward)
        self.episode_returns.append(discounted_return)
        if grad_norm is not None:
            self.gradient_norms.append(grad_norm)

    def save_plots(self):
        fig, axs = plt.subplots(3, 1, figsize=(10, 15))
        
        # Plot 1: Raw Rewards (What the game gave us)
        axs[0].plot(self.episode_rewards, label='Total Reward', color='blue', alpha=0.6)
        # Add a moving average 
        lag = 20
        if len(self.episode_rewards) > lag:
            avg = np.convolve(self.episode_rewards, np.ones(lag)/lag, mode='valid')
            axs[0].plot(range(lag-1, len(self.episode_rewards)), avg, color='red', label=f'{lag}-Ep Avg')
        axs[0].set_title('Episode Rewards (Survival + Targets)')
        axs[0].set_xlabel('Episode')
        axs[0].legend()
        axs[0].grid(True)

        # Plot 2: Discounted Returns (What the Agent sees)
        axs[1].plot(self.episode_returns, label='G_0 (Discounted)', color='green', alpha=0.6)
        axs[1].set_title('Discounted Returns (G_0)')
        axs[1].set_xlabel('Episode')
        axs[1].grid(True)

        # Plot 3: Gradient Norms (Stability)
        if self.gradient_norms:
            axs[2].plot(self.gradient_norms, label='Gradient Norm', color='purple')
            axs[2].set_title('Gradient Magnitudes (Learning Stability)')
            axs[2].set_xlabel('Update Step')
            axs[2].set_yscale('log') # Log scale is better for gradients
            axs[2].grid(True)

        plt.tight_layout()
        plt.savefig(f"{self.filepath}.png")
        plt.close()
        print(f"Plots saved to {self.filepath}.png")

class Policy():
    def __init__(self, input_size, output_size):
        # Initialize weights to zeros
        self.weights = np.zeros((input_size, output_size))        
    
    def forward(self, state):
        return state @ self.weights  # Linear model
    
    def backward(self, states, actions, returns, sigma, learning_rate):
        """
        Performs the REINFORCE update on the weights.
        Returns: The magnitude (norm) of the gradient for logging.
        """        
        # rerun forward pass
        mus = states @ self.weights 
        
        # Action error (exploration noise)
        error = actions - mus
        
        # Gradient of Log-Probability (direction to change policy in)
        grad_log_pi = error / (sigma ** 2)
        
        # Credit assignment (good/+ve >> uphill/Reinforce, bad/-ve >> downhill/suppress)
        weighted_grads = grad_log_pi * returns[:, np.newaxis]
        
        # Sum over time steps to get total gradient for the batch
        # Converts "change in action" (dlogp/dmu) to "change in weights" (dlogp/dW)
        gradient_matrix = states.T @ weighted_grads
        
        # Caluclate update for gradient ascent
        # Note: Divide by episode length incase they vary - keeps learning rate stable
        update_step = (learning_rate * gradient_matrix) / len(states)
        
        # Update the weights
        self.weights += update_step
        
        # Return the "Size" of the update for diagnostics
        return np.linalg.norm(update_step)

class ContinuousLinearController(FlightController):
    def __init__(self):
                
        # Load config
        with open('continuous_linear_config.json', 'r') as f:
            self.config = json.load(f)
        
        # Initialise policy - 7 states in, 2 actions out
        self.policy = Policy(7, 2)
        
        # Initialise logger
        self.logger = ExperimentLogger(self.config['experiment_name'])

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
            self.update_policy(states, actions, returns)
            
            # Log
            if episode % 50 == 0:
                print(f"Episode {episode}: Total Reward: {sum(rewards):.2f}")
                
        self.save()

    def update_policy(self, states, actions, rewards):
        """
        Update policy weights 
            Get returns
            Update policy
        """
        returns = self.get_returns(rewards)
        
        states = np.array(states)
        actions = np.array(actions)
        returns = np.array(returns)
        
        self.policy.backward(
            states, 
            actions, 
            returns, 
            self.config['hyperparameters']['sigma'], 
            self.config['hyperparameters']['learning_rate']
        )

    def save(self):
        path = os.path.join(BASE_PATH, self.config['experiment_name'], '.npy')
        np.save(path, self.policy.weights)
        print("Weights saved.")

    def load(self, filename=None):
        if filename is None:
            path = os.path.join(BASE_PATH, self.config['experiment_name'], '.npy')
        else:
            path = os.path.join(BASE_PATH, filename, '.npy')
        try:
            self.policy.weights = np.load(path)
            print("Weights loaded.")
        except:
            print("No weights found.")