import numpy as np
import json
from flight_controller import FlightController
from drone import Drone
import matplotlib.pyplot as plt
import os

BASE_PATH = os.path.join(
    os.path.dirname(__file__), 
    "experiments", 
    "linear_reinforce"
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

    def log_episode(self, episode_reward, initial_return, grad_norm):
        self.episode_rewards.append(episode_reward)
        self.episode_returns.append(initial_return)
        self.gradient_norms.append(grad_norm)

    def generate_plots(self):
        fig, axs = plt.subplots(2, 1, figsize=(8, 8))
        
        # Plot 1: Raw Rewards
        axs[0].plot(self.episode_rewards, label='Total Reward', color='blue', alpha=0.6)
        
        # Moving Average
        lag = 20
        if len(self.episode_rewards) > lag:
            # Simple moving average using convolution
            avg = np.convolve(self.episode_rewards, np.ones(lag)/lag, mode='valid')
            # Align x-axis
            axs[0].plot(range(lag-1, len(self.episode_rewards)), avg, color='red', label='20-Ep Avg')
            
        axs[0].set_title('Episode Rewards')
        axs[0].set_xlabel('Episode')
        axs[0].legend()
        axs[0].grid(True)

        # # Plot 2: Discounted Returns
        # axs[1].plot(self.episode_returns, label='G_0', color='green', alpha=0.6)
        # axs[1].set_title('Discounted Returns (G_0)')
        # axs[1].set_xlabel('Episode')
        # axs[1].grid(True)

        # Plot 3: Gradients
        if self.gradient_norms:
            axs[1].plot(self.gradient_norms, label='Gradient Norm', color='purple')
            axs[1].set_title('Gradient Norms')
            axs[1].set_xlabel('Update Step')
            axs[1].set_yscale('log')
            axs[1].grid(True)

        plt.tight_layout()
        plt.savefig(f"{self.filepath}.png")
        plt.show()
        print(f"Plots saved to {self.filepath}.png")

class Policy():
    def __init__(self, input_size, output_size, init_mode='random'):
        print(f"Weight Initialisation Mode: {init_mode}")
        self.weights = np.random.randn(input_size, output_size)  # Initialize weights with random values
        self.weights /= (np.linalg.norm(self.weights, axis=0) * 5)  # Normalise the weights so they start nearer 0
        if init_mode == 'random':
            pass
        # Initialise weights to prefer hovering
        elif init_mode == 'hover':
            self.weights[-1, 0] += 1.  # bias term for thrust set to counter gravity
        
        # Print weights formatting to 3 decimal places
        with np.printoptions(suppress=True, precision=3):
            print(f"Initial Weights: {self.weights}")
        
    
    def forward(self, state):
        return state @ self.weights  # Linear model
    
    def backward(self, states, actions, returns, sigma, learning_rate):
        """
        Performs the REINFORCE update on the weights.
        Returns: The magnitude (norm) of the gradient for logging.
        """        
        # Ensure numpy arrays
        states = np.array(states)
        actions = np.array(actions)
        returns = np.array(returns)
        
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

class LinearReinforceController(FlightController):
    def __init__(self):
                
        # Load config
        with open('linear_reinforce_config.json', 'r') as f:
            self.config = json.load(f)
        
        # Initialise policy - 7 states in, 2 actions out
        self.policy = Policy(7, 2, init_mode=self.config['hyperparameters']['weight_init_mode'])
        
        # Initialise logger
        self.logger = ExperimentLogger(self.config['experiment_name'])
        
        # Store previous distance for delta distance reward
        self.prev_dist = None
        

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
        
        # Calculate distance to target
        target = drone.get_next_target()
        dist = np.linalg.norm([target[0] - drone.x, target[1] - drone.y])
        
        # Delta distance - simple +/-x based on moving toward or away from target
        if cfg['delta_distance'] is not None:
            if self.prev_dist is None:
                pass  # Just do nothing on the first step
            else:
                delta_dist = 1.0 if self.prev_dist - dist > 0.0 else -1.0
                reward += (delta_dist * cfg['delta_distance']) 
            self.prev_dist = dist  # Update the stored distance
         
        # Distance Penalty
        if cfg['distance_weight'] is not None:
            reward -= dist * cfg['distance_weight']
        
        # Hit Bonus
        if cfg['hit_bonus'] is not None:
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
        baseline_mode = self.config['hyperparameters']['baseline_mode']
        if baseline_mode == 'zero':
            baseline = 0.0
        elif baseline_mode == 'mean':
            baseline = returns.mean()
        returns = (returns - baseline) / (returns.std() + 1e-8)
            
        return returns
    
    def run_episode(self):
            
        states = []
        actions = []
        rewards = []
        drone = self.init_drone(mode='random')
        
        # Iniitalise previous distance at the start of each episode. 
        # Otherwise it carries over unpredictably into rewards
        target = drone.get_next_target()
        self.prev_dist = np.linalg.norm([target[0] - drone.x, target[1] - drone.y])
        
        # Step through episode
        for step in range(self.config['hyperparameters']['max_steps']):
            
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
            
        return states, actions, rewards
        

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
        
        for episode in range(self.config['hyperparameters']['n_episodes']):
            
            # Run episode
            states, actions, rewards = self.run_episode()
            
            # Log
            total_reward = sum(rewards)
            if episode % 10 == 0:
                print(f"Episode {episode}: Total Reward: {total_reward:.2f}")
                
            # Save weights periodically, incase it craps out
            if episode % 100 == 0:
                self.save()
                
            # Calculate returns
            returns = self.get_returns(rewards)
            
            # Update policy
            grad_norm = self.policy.backward(
                states, 
                actions, 
                returns, 
                self.config['hyperparameters']['sigma'], 
                self.config['hyperparameters']['learning_rate']
            )
            
            # Update logger
            self.logger.log_episode(total_reward, returns[0], grad_norm)
                
        # Save final weights
        self.save()
                
        # Plot results
        self.logger.generate_plots()

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
        # Save weights
        path = os.path.join(BASE_PATH, self.config['experiment_name']+'_weights.npy')
        np.save(path, self.policy.weights)
        # Save config
        path = os.path.join(BASE_PATH, self.config['experiment_name']+'_config.json')
        with open(path, 'w') as f:
            json.dump(self.config, f, indent=4)
        print("Weights saved.")

    def load(self, filename=None):
        if filename is None:
            path = os.path.join(BASE_PATH, self.config['experiment_name']+'_weights.npy')
        else:
            path = os.path.join(BASE_PATH, filename+'_weights.npy')
        try:
            self.policy.weights = np.load(path)
            print("Weights loaded.")
        except:
            print("No weights found.")