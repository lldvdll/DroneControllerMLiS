import numpy as np
import json
from flight_controller import FlightController
from drone import Drone
import matplotlib.pyplot as plt
import os

BASE_PATH = os.path.join(
    os.path.dirname(__file__), 
    "experiments", 
    "neural_reinforce"
    )

class ExperimentLogger:
    def __init__(self, experiment_name):
        # Long-term history (Batch Averages)
        self.history = {
            'rewards': [],
            'lengths': [],
            'grads': [],
            'actions': []
        }
        
        # Short-term buffer (Raw Episode Data)
        self.buffer = {
            'rewards': [],
            'lengths': [],
            'actions': []
        }
        
        self.filepath = os.path.join(BASE_PATH, f"{experiment_name}_training_log")

    def record_episode(self, reward, length, action):
        """Called at the end of EVERY episode"""
        self.buffer['rewards'].append(reward)
        self.buffer['lengths'].append(length)
        self.buffer['actions'].append(action)

    def log_optimization_step(self, grad_norm):
        """Called after policy.backward() to aggregate and clear buffer"""
        if not self.buffer['rewards']:
            return 0.0, 0.0  # Handle empty buffer safety
            
        # Aggregate
        avg_reward = np.mean(self.buffer['rewards'])
        avg_length = np.mean(self.buffer['lengths'])
        avg_action = np.mean(self.buffer['actions'], axis=0)
        
        # Store in History
        self.history['rewards'].append(avg_reward)
        self.history['lengths'].append(avg_length)
        self.history['grads'].append(grad_norm)
        self.history['actions'].append(avg_action)
        
        # Clear Buffer
        self.buffer = {'rewards': [], 'lengths': [], 'actions': []}
        
        return avg_reward, avg_length

    def generate_plots(self):
        fig, axs = plt.subplots(2, 2, figsize=(12, 6))
        
        # Rewards
        axs[0, 0].plot(self.history['rewards'], label='Batch Avg', color='blue', alpha=0.5)
        # Add trendline
        if len(self.history['rewards']) > 20:
            avg = np.convolve(self.history['rewards'], np.ones(20)/20, mode='valid')
            axs[0, 0].plot(range(19, len(self.history['rewards'])), avg, color='red', label='Trend')
        axs[0, 0].set_title('Average Reward')
        axs[0, 0].legend()
        axs[0, 0].grid(True, alpha=0.3)

        # Survival
        axs[0, 1].plot(self.history['lengths'], color='green', alpha=0.6)
        axs[0, 1].set_title('Avg Survival Time')
        axs[0, 1].grid(True, alpha=0.3)

        # Actions
        actions = np.array(self.history['actions'])
        if len(actions) > 0:
            axs[1, 0].plot(actions[:, 0], label='Thrust (0-1)', color='orange')
            axs[1, 0].plot(actions[:, 1], label='Roll (-1 to 1)', color='purple')
            axs[1, 0].set_title('Avg Action Convergence')
            axs[1, 0].legend()
            axs[1, 0].grid(True, alpha=0.3)

        # Gradients
        axs[1, 1].plot(self.history['grads'], color='black', alpha=0.5)
        axs[1, 1].set_title('Gradient Norms')
        axs[1, 1].set_yscale('log')
        axs[1, 1].grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig(f"{self.filepath}.png")
        plt.show()
        print(f"Plots saved to {self.filepath}.png")

class Policy():
    def __init__(self, input_size, hidden_size, output_size, weight_scaling=1.0):
        """
        Initialize the weights (W1, W2) and biases (b1, b2).
        """
        # Hidden layer
        xavier_scale1 = np.sqrt(2.0 / (input_size + hidden_size))  # Xavier initialisation
        self.W1 = np.random.randn(hidden_size, input_size) * xavier_scale1 * weight_scaling
        self.b1 = np.zeros(hidden_size)
        
        # Output layer
        xavier_scale2 = np.sqrt(2.0 / (hidden_size + output_size))  # Uniform Xavier initialisation
        self.W2 = np.random.randn(output_size, hidden_size) * xavier_scale2 * weight_scaling
        self.b2 = np.zeros(output_size)
        
        # Cheat, force hover
        self.b2 = np.array([0.0, 0.0])
        
        print(f'Input size: {input_size}, Hidden size: {hidden_size}, Output size: {output_size}')
        print(f'Hidden Layer: W1: {self.W1.shape}, b1: {self.b1.shape}')
        print(f'Output Layer: W2: {self.W2.shape}, b2: {self.b2.shape}')
        pass
    
    def forward(self, state):
        """
        Takes a state vector, runs it through the network, 
        and returns action probabilities.
        Notation: 
            W: weights matrix
            b: bias vector
            s: state vector (input)
            z: pre-activation hidden layer vector
            h: post-activation hidden layer vector
            p: action probabilities vector (output)
        """
        # Hidden layer
        z1 = state @ self.W1.T + self.b1
        h1 = np.maximum(0, z1)  # ReLU, for gradient stability and faster learning
        
        # Output layer
        z2 = h1 @ self.W2.T + self.b2
        
        # Softmax
        mu_thrust = 1 / (1 + np.exp(-z2[0]))  # For thrust, sigmoid (0, 1)
        mu_roll = np.tanh(z2[1])  # For roll, tanh (-1, 1)
        
        return np.array([mu_thrust, mu_roll]), h1, z2
    
    def backward(self, states, actions, returns, sigma, learning_rate):
        """
        Performs the REINFORCE update on the weights.
        Returns: The magnitude (norm) of the gradient for logging.
        """        
        # Ensure numpy arrays
        states = np.array(states)
        actions = np.array(actions)
        returns = np.array(returns)
        
        grad_W1 = np.zeros_like(self.W1)
        grad_b1 = np.zeros_like(self.b1)
        grad_W2 = np.zeros_like(self.W2)
        grad_b2 = np.zeros_like(self.b2)
        
        N = len(states)
        
        for i in range(N):
            state = states[i]
            action = actions[i]
            ret = returns[i]
            
            # Re-run Forward to get intermediates
            mu, h1, z2 = self.forward(state)
            
            # Calculate REINFORCE Error Signal
            # "How much should we have shifted the mean?"
            # Error = (Action_Taken - Mean_Predicted) / Variance * Return
            grad_log_pi = (action - mu) / (sigma**2)
            weighted_error = grad_log_pi * ret
            
            # Output Layer Gradients (Chain Rule)
            d_out = np.zeros(2)
            
            # Thrust (Sigmoid Derivative: sig * (1 - sig))
            d_out[0] = weighted_error[0] * (mu[0] * (1 - mu[0]))
            
            # Roll (Tanh Derivative: 1 - tanh^2)
            d_out[1] = weighted_error[1] * (1 - mu[1]**2)
            
            # Gradients for W2 and b2
            # d_W2 = d_out * h1
            grad_W2 += np.outer(d_out, h1)
            grad_b2 += d_out
            
            # Backprop to Hidden Layer
            # Error at hidden = (d_out * W2)
            d_hidden = d_out @ self.W2
            
            # Hidden Layer Activation Derivative (ReLU)
            # Derivative is 1 if h1 > 0, else 0
            d_relu = d_hidden * (h1 > 0)
            
            # Gradients for W1 and b1
            grad_W1 += np.outer(d_relu, state)
            grad_b1 += d_relu
            
        # Gradient Ascent
        # Normalize by batch size for stability
        self.W1 += learning_rate * (grad_W1 / N)
        self.b1 += learning_rate * (grad_b1 / N)
        self.W2 += learning_rate * (grad_W2 / N)
        self.b2 += learning_rate * (grad_b2 / N)
        
        # Return gradient norm for logging
        total_norm = np.linalg.norm(grad_W1) + np.linalg.norm(grad_W2)
        return total_norm

class NeuralReinforceController(FlightController):
    def __init__(self):
                
        # Load config
        with open('neural_reinforce_config.json', 'r') as f:
            self.config = json.load(f)
        
        # Initialise policy - 7 states in, 2 actions out
        hidden_size = self.config['hyperparameters']['hidden_size']
        weight_scaling = self.config['hyperparameters']['weight_scaling']
        self.policy = Policy(input_size=7, hidden_size=hidden_size, output_size=2, weight_scaling=weight_scaling)
        
        # Initialise logger
        self.logger = ExperimentLogger(self.config['experiment_name'])
        
        # State/action stores for reward calculations
        self.prev_dist = None
        self.prev_velocity = np.zeros(2) 
        self.last_action = np.zeros(2)
        
    def get_max_simulation_steps(self):
            return 3000 

    def get_state(self, drone: Drone):
        """
        Get state from drone object
        """
        # Get drone state parameters
        target = drone.get_next_target()
        if self.config['hyperparameters']['curriculum'] == 'hover':
            target = (0, 0)
        dx = (target[0] - drone.x)
        dy = (target[1] - drone.y)
        vx = drone.velocity_x
        vy = drone.velocity_y
        pitch = drone.pitch
        pitch_vel = drone.pitch_velocity
        bias = 1.0  # Bias for linear model
        
        # Store velocity for reward calculation
        self.prev_velocity = np.array([vx, vy])
        
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
        mu, h1, z2 = self.policy.forward(state)
        
        # Sample from Gaussian  
        if mode == 'train':
            sigma = self.config['hyperparameters']['sigma']
            action = np.random.normal(loc=mu, scale=sigma)
        else:
            action = mu
            
        self.last_action = action
        return action
    
    def convert_action_to_thrust(self, action):
        """
        Convert action to thrusts
            0: Total thrust
            1: Roll
        """
        t_total = action[0] * 2.0  # Convert sigmoid range [0, 1] to full thrust range [0, 2]
        roll = action[1]
        
        # Mixing logic to convert to thrusts
        left = (t_total + roll) / 2.0
        right = (t_total - roll) / 2.0
        
        # Clip to ensure valid motor values [0, 1], since sampling during training can result in invalid values
        thrusts = (float(np.clip(left, 0.0, 1.0)), 
                float(np.clip(right, 0.0, 1.0)))
        
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
        TODO: Turn this into a class! To manage previous states, store logs, initialise based on config, and clean things up
        """
        cfg = self.config['rewards']
        reward = 0.0
        reward_log = {
            'distance_weight': 0.0,
            'delta_distance': 0.0,
            'pitch_penalty': 0.0,
            'spin_penalty': 0.0,
            'hit_bonus': 0.0,
            'crash_penalty': 0.0,
            'dx_penalty': 0.0,
            'velocity_alignment': 0.0,
            'drift_correction': 0.0
        }
        
        # Calculate distance to target
        target = drone.get_next_target()
        if self.config['hyperparameters']['curriculum'] == 'hover':
            target = (0, 0)
        dist_vector = np.array([target[0] - drone.x, target[1] - drone.y])
        dist = np.linalg.norm(dist_vector)
         
        # Distance Penalty
        if cfg['distance_weight'] is not None:
            part = (0.25 - dist) * cfg['distance_weight']  # Add bias, positive reward inside screen
            reward += part
            reward_log['distance_weight'] = part
        
        # Delta distance - simple +/-x based on moving toward or away from target
        if cfg['delta_distance'] is not None:
            if self.prev_dist is None:
                pass  # Just do nothing on the first step
            else:
                # Continuous change in target distance
                delta_dist = self.prev_dist - dist  
                part = (delta_dist * cfg['delta_distance']) 
                reward += part
                reward_log['delta_distance'] = part
        
        # Delta distance - simple +/-x based on moving toward or away from target
        if cfg['delta_direction'] is not None:
            if self.prev_dist is None:
                pass  # Just do nothing on the first step
            else:
                # Binary change in target distance. 
                # Note: when previous==current, the reward is negative, discouraging sitting still
                delta_direction = 1.0 if self.prev_dist > dist else -1.0
                if abs(self.prev_dist - dist) < 0.0001:
                    delta_direction = 2.0  # Give a big reward for staying in one place
                part = (delta_direction * cfg['delta_direction']) 
                reward += part
                reward_log['delta_direction'] = part
            
        # Velocity alignment - compares distance to target and velocity vectors, rewards alignment
        if cfg['velocity_alignment'] is not None:
            # Get change in velocity vector - unnormalised for magnitude, or "acceleration"
            velocity = np.array([drone.velocity_x, drone.velocity_y])
            delta_v = velocity - self.prev_velocity
            # Normalise distance to target vector and calculate alignment
            target_dir = dist_vector / dist
            alignment = np.dot(delta_v, target_dir)
            part = alignment * cfg['velocity_alignment']
            reward += part
            reward_log['velocity_alignment'] = part
              
        # Drift correction - if drifting away from target, reward actoins which correct drift
        if cfg['drift_correction'] is not None:
            vx = drone.velocity_x
            target_x_diff = target[0] - drone.x   
            moving_away = np.sign(vx) != np.sign(target_x_diff)  
            if moving_away:
                roll = self.last_action[1]
                correction = -(vx * roll)  # e.g. vx<0 and roll>0, moving left and rolling right, so signs reverse for correction
            else:
                correction = 0.0
            part = correction * cfg['drift_correction']
            reward += part
            reward_log['drift_correction'] = part
            
        # Pitch penalty
        if cfg['pitch_penalty'] is not None:
            part = -np.pow(drone.pitch, 2) * cfg['pitch_penalty']
            reward += part
            reward_log['pitch_penalty'] = part
        
        # Hit Bonus
        if cfg['hit_bonus'] is not None:
            if drone.has_reached_target_last_update:
                part = cfg['hit_bonus']
                # Modulator: 1.0 if stopped, 0.0 if speed > 1.0
                if self.config['hit_slow']:
                    speed = np.linalg.norm([drone.velocity_x, drone.velocity_y])  
                    stability = max(0.0, 1.0 - speed) 
                    part *= stability
                reward += part
                reward_log['hit_bonus'] = part
                
        # Penalise high pitch velocity
        if cfg.get('spin_penalty') is not None:
            part = -(drone.pitch_velocity ** 2) * cfg['spin_penalty']
            reward += part
            reward_log['spin_penalty'] = part
        
        # Crash penalty - early stopping
        if cfg.get('crash_penalty') is not None:
            limit = self.config['hyperparameters']['safe_zone']
            if abs(drone.x) > limit or abs(drone.y) > limit:
                part = -cfg['crash_penalty']
                reward += part
                reward_log['crash_penalty'] = part
                
        # Lateral speed penalty
        if cfg.get('dx_penalty') is not None:
            part = -(drone.velocity_x ** 2) * cfg['dx_penalty']
            reward += part
            reward_log['dx_penalty'] = part
        
        # Log total
        # reward_log['total'] = reward
        
        self.prev_dist = dist  # Update the stored distance
            
        return reward, reward_log
    
    def get_returns(self, rewards, normalise=True):
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
        if normalise:
            returns = self.normalise_returns(returns)
            
        return returns
    
    def normalise_returns(self, returns):
        """
        Seperate function for returns normalisation for batch processing
        """
        returns = np.array(returns)
        baseline_mode = self.config['hyperparameters']['baseline_mode']
        if baseline_mode == 'zero':
            baseline = 0.0
        elif baseline_mode == 'mean':
            baseline = returns.mean()
        returns = (returns - baseline) / (returns.std() + 1e-8)
        
        return returns
    
    
    def run_episode(self, limits=(-0.5, 0.5)):
        
        
        # Init drone with increasingly distant random targets
        drone = self.init_drone(mode='random', limits=limits)
        
        # Iniitalise previous state/action stores before each episode begins. 
        # Otherwise they carry over unpredictably into rewards
        target = drone.get_next_target()
        if self.config['hyperparameters']['curriculum'] == 'hover':
            target = (0, 0)
        self.prev_dist = np.linalg.norm([target[0] - drone.x, target[1] - drone.y])
        self.prev_velocity = np.zeros(2) 
        self.last_action = np.zeros(2)
        
        # Initialise stores
        states = []
        actions = []
        rewards = []
        reward_logs = []        
        
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
            reward, reward_log = self.get_reward(drone)
            reward_logs.append(reward_log)
            
            # Store data
            states.append(state)
            actions.append(action)
            rewards.append(reward)
            
            # Early stopping
            limit = self.config['hyperparameters']['safe_zone']
            if self.config['rewards']['crash_penalty'] is not None:
                if abs(drone.x) > limit or abs(drone.y) > limit:
                    break
            
        return states, actions, rewards, reward_logs
        

    def train(self):
        """
        Run episode loop and update policy
            Per episodes
                Initialise environment
                Get States and Actions
                Step through environment
                Calculate rewards
            Per batch
                Normmalise returns
                Update policy
        """
        print(f"Starting Training: {self.config['experiment_name']}")
        
        # Track best score for saving
        best_score = -np.inf
        
        # Run episode loop
        n_episodes = self.config['hyperparameters']['n_episodes']
        for episode in range(n_episodes):
            
            # Set curriculum and run episode
            target_mode = self.config['hyperparameters']['curriculum']
            if target_mode == 'hover':
                # Note: this isn't active, target (0, 0) needs to be hardcoded so they don't run out  
                # TODO: Implement properly, most likely in drone initialisation
                limits = (0.0, 0.0) # Target always in centre - don't use hit reward with this!
            elif target_mode == 'increasing':
                curriculum_max_episodes = self.config['hyperparameters']['curriculum_max_episodes']
                difficulty = min(1.0, episode / curriculum_max_episodes)  # Target distance increases over episodes
                limit = max(0.1*2, difficulty/2)  # min distance of target size*2, so it's not just sitting on them
                limits = (-limit, limit)
                difficulty = self.run_episode(limits=limits)
            elif target_mode == 'random':
                limits = (-0.5, 0.5)  # Target location is always random - the objective
            states, actions, rewards, reward_logs = self.run_episode(limits=limits)
            
            # Log episode
            avg_action = np.mean(actions, axis=0)
            self.logger.record_episode(
                reward=np.sum(rewards),
                length=len(states),
                action=avg_action
            )
                
            # Save weights periodically, incase it craps out
            if episode % 100 == 0:
                self.save()
                
            # Save best score
            if np.sum(rewards) > best_score:
                best_score = np.sum(rewards)
                self.save(best=True)
                
            # Calculate returns - espisodic returns for discounting, will normalise over batch
            returns = self.get_returns(rewards, normalise=False)
            
            # Store prtial batch
            batch_size = self.config['hyperparameters']['batch_size']
            if episode % batch_size == 0:  # Start a new batch
                batch_states = states
                batch_actions = actions
                batch_all_returns = returns
                batch_rewards = np.sum(rewards)
                
            else:  # Continue batch - note also accumulates at end of batch
                batch_states.extend(states)
                batch_actions.extend(actions)
                batch_all_returns.extend(returns)
                batch_rewards += np.sum(rewards)
                
            # End of batch: Normalise returns, update policy, and log
            if (episode + 1) % batch_size == 0:
                returns_norm = self.normalise_returns(batch_all_returns)
                grad_norm = self.policy.backward(  
                    batch_states, 
                    batch_actions, 
                    returns_norm, 
                    self.config['hyperparameters']['sigma'], 
                    self.config['hyperparameters']['learning_rate']
                )
                avg_rew, avg_len = self.logger.log_optimization_step(grad_norm)
                print(f"Batch {(episode + 1) / batch_size}: Average Reward: {avg_rew:.4f}, Average Length: {avg_len:.4f}")
                
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

    def save(self, best=False):
        # Choose best or latest save
        if best:
            path = os.path.join(BASE_PATH, self.config['experiment_name']+'_weights_best.npz')
        else:
            path = os.path.join(BASE_PATH, self.config['experiment_name']+'_weights.npz')
        # Save weights
        np.savez(path, 
                 W1=self.policy.W1, 
                 b1=self.policy.b1, 
                 W2=self.policy.W2, 
                 b2=self.policy.b2)
        # Save config
        path = os.path.join(BASE_PATH, self.config['experiment_name']+'_config.json')
        with open(path, 'w') as f:
            json.dump(self.config, f, indent=4)
        print("Weights saved.")

    def load(self, filename=None):
        if filename is None:
            if self.config['load'] == 'best':
                path = os.path.join(BASE_PATH, self.config['experiment_name']+'_weights_best.npz')
            elif self.config['load'] == 'latest':
                path = os.path.join(BASE_PATH, self.config['experiment_name']+'_weights.npz')
        else:
            path = os.path.join(BASE_PATH, filename+'_weights.npz')
        try:
            data = np.load(path)
            self.policy.W1 = data['W1']
            self.policy.b1 = data['b1']
            self.policy.W2 = data['W2']
            self.policy.b2 = data['b2']
            print("Weights loaded.")
        except:
            print("No weights found.")