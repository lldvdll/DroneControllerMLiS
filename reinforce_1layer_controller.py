"""
Policy gradient approach, using a 1 hidden layer neural net approach.
State space is continuous.
Action space is discrete - e.g. up, down, left, right, stop, hover, with associated fixed thrust vectors
Outputs probability over discrete actions
Exploration/Exploitation is implicit in sampling from the action distribution
Exploration rate should naturally decrease with training as the network learns to make better predictions
"""

from flight_controller import FlightController
from drone import Drone
from typing import Tuple 
import numpy as np
import matplotlib.pyplot as plt
import json



class ExperimentLogger:
    def __init__(self, n_actions, n_states):
        self.episode_rewards = []
        self.episode_lengths = []
        self.policy_entropies = []
        self.gradient_norms = []
        self.avg_state_vals = []   # Shape: [Episode, State_Dim]
        self.avg_action_probs = [] # Shape: [Episode, N_Actions]
        self.n_actions = n_actions
        self.n_states = n_states
    
    def log_episode(self, reward, length, avg_entropy, grad_norm, avg_state, avg_probs):
        self.episode_rewards.append(reward)
        self.episode_lengths.append(length)
        self.policy_entropies.append(avg_entropy)
        self.gradient_norms.append(grad_norm)
        self.avg_state_vals.append(avg_state)
        self.avg_action_probs.append(avg_probs)
        
    def plot(self, window=50):
        """
        Plots the training history.
        window: Moving average window size for smoothing
        """
        fig, axs = plt.subplots(2, 3, figsize=(18, 8))
        
        # Helper for moving average
        def moving_average(data, w):
            return np.convolve(data, np.ones(w), 'valid') / w

        # Rewards
        axs[0, 0].plot(self.episode_rewards, alpha=0.3, color='blue', label='Raw')
        if len(self.episode_rewards) > window:
            ma = moving_average(self.episode_rewards, window)
            axs[0, 0].plot(range(window-1, len(self.episode_rewards)), ma, color='red', label='Avg')
        axs[0, 0].set_title("Total Reward")
        axs[0, 0].legend()

        # Entropy (The "Confidence" Meter)
        axs[0, 1].plot(self.policy_entropies, color='green')
        axs[0, 1].set_title("Policy Entropy (Randomness)")
        axs[0, 1].set_xlabel("Episode")
        # Reference line for max entropy (log(5 actions) ≈ 1.6)
        axs[0, 1].axhline(y=1.6, color='black', linestyle='--', alpha=0.5, label='Max Random')

        # Episode Length
        axs[1, 0].plot(self.episode_lengths, color='purple')
        axs[1, 0].set_title("Episode Length (Survival Time)")

        # Gradient Norms (The "Stability" Meter)
        axs[1, 1].plot(self.gradient_norms, color='orange')
        axs[1, 1].set_title("Avg Gradient Norm")
        axs[1, 1].set_yscale('log') # Log scale helps see explosions
        
        # Input Scale
        state_data = np.array(self.avg_state_vals)
        if len(state_data) > 0:
            labels = ['dx', 'dy', 'vx', 'vy', r'cos$\theta$', r'sin$\theta$', 'av']
            labels = labels[:self.n_states]
            for i in range(self.n_states):
                axs[0, 2].plot(state_data[:, i], label=labels[i], alpha=0.7)
            
            axs[0, 2].set_title("Avg Input Magnitude (Should be < 1.0)")
            axs[0, 2].legend(fontsize='small', loc='upper right')
            axs[0, 2].grid(True, alpha=0.3)

        # Action Probabilities
        prob_data = np.array(self.avg_action_probs)
        if len(prob_data) > 0:
            x = range(len(prob_data))
            y = [prob_data[:, i] for i in range(self.n_actions)]
            labels = ["Hover", "Up", "Down", "Right", "Left"]
            
            axs[1, 2].stackplot(x, *y, labels=labels, alpha=0.6)
            axs[1, 2].set_title("Action Probability Distribution")
            axs[1, 2].set_ylim(0, 1.0)
            axs[1, 2].legend(loc='lower left', fontsize='small')
        
        
        plt.tight_layout()
        plt.show()
    
class PolicyNetwork:
    def __init__(self, input_size, hidden_size, output_size):
        """
        Initialize the weights (W1, W2) and biases (b1, b2).
        """
        self.output_size = output_size
        
        # This combines the "Xavier" scaling and the "Entropy" reduction
        # Reduces the weights so initial action probabilities are closer to unifrom
        target_std = (1.0 / np.sqrt(input_size)) * 0.1
        
        # Hidden layer
        self.W1 = np.random.normal(loc=0.0, scale=target_std, size=(hidden_size, input_size))
        self.b1 = np.zeros(hidden_size)
        
        # Output layer
        self.W2 = np.random.normal(loc=0.0, scale=target_std, size=(output_size, hidden_size))
        self.b2 = np.zeros(output_size)
        
        print(f'Input size: {input_size}, Hidden size: {hidden_size}, Output size: {output_size}')
        print(f'Hidden Layer: W1: {self.W1.shape}, b1: {self.b1.shape}')
        print(f'Output Layer: W2: {self.W2.shape}, b2: {self.b2.shape}')
        pass

    def forward(self, s):
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
        z1 = self.W1 @ s + self.b1
        h1 = np.tanh(z1)  # Just using tanh for simplicity
        
        # Output layer
        z2 = self.W2 @ h1 + self.b2
        
        # Softmax
        z2_norm = z2 - np.max(z2) # Subtracting max for numerical stability - so exp doesn't explode
        exp_logits = np.exp(z2_norm)  
        p = exp_logits/np.sum(exp_logits)
        
        return p, h1  # Return hidden state for backprop
    
    def backward(self, state, a_t, g_t, learning_rate=0.001):
        """
        Runs a backwards pass through the network, 
        updating the weights and biases.
        Notation: 
            W: weights matrix
            b: bias vector
            s: state vector (input)
            z: pre-activation hidden layer vector
            h: post-activation hidden layer vector
            p: action probabilities vector (output)
            g: return - note: this is normalised in calculate_returns
            J: loss function
            d_: derivative prefix
        """
        # TODO: May want to batch train over episode timeseries for efficiency
        
        # Re-run forward pass to get the probabilities and hidden states
        # TODO: The class should store these on each forward pass so we don't have to re-run
        p, h1 = self.forward(state)
        
        # Error at output (dJ/dz2)
        # Loss function J = -g_t * np.log(p)
        # The gradient of the Objective J w.r.t the logits z2
        one_hot_target = np.zeros(len(p))
        one_hot_target[a_t] = 1.0
        delta = (one_hot_target - p)
        dJ_dz2 = g_t * delta 

        # Backprop to layer 2 weights
        # Gradient for W2 is: (Error at Output) * (Input to Output Layer)
        dJ_dW2 = np.outer(dJ_dz2, h1)  # dJ/dW2 = (dJ/dz2) * (dz2/dW2), where dz2/dW2 is h1
        dJ_db2 = dJ_dz2                # dJ/db2 = (dJ/dz2) * (dz2/db2), where dz2/db2 is 1

        # Backprop Error to Hidden Layer (dJ/dh1)
        # Pass the error backwards through the weights to see who to blame in h1
        dJ_dh1 = self.W2.T @ dJ_dz2    # dJ/dh1 = (dJ/dz2) * (dz2/dh1), where dz2/dh1 is W2.T

        # Backprop through activation function tanh
        dJ_dz1 = dJ_dh1 * (1 - h1**2)  # dJ/dz1 = (dJ/dh1) * (dh1/dz1), where dh1/dz1 is 1 - tanh^2, h1 = tanh(z1)

        # Backprop to layer 1 weights
        dJ_dW1 = np.outer(dJ_dz1, state) # dJ/dW1 = (dJ/dz1) * (dz1/dW1), where dz1/dW1 is state
        dJ_db1 = dJ_dz1                  # dJ/db1 = (dJ/dz1) * (dz1/db1), where dz1/db1 is 1
        
        # For entropy logging
        total_norm = np.linalg.norm(np.concatenate([
            dJ_dW1.flatten(), 
            dJ_dW2.flatten(), 
            dJ_db1.flatten(), 
            dJ_db2.flatten()
        ]))

        # Update weights (gradient ascent - maximise rewards)
        self.W1 += learning_rate * dJ_dW1
        self.b1 += learning_rate * dJ_db1
        self.W2 += learning_rate * dJ_dW2
        self.b2 += learning_rate * dJ_db2
        
        return total_norm
    
class Reinforce1LayerController(FlightController):

    def __init__(self):
        
        # Load config with parameters etc
        with open('reinforce_1layer_config.json', 'r') as f:
            config = json.load(f)
        self.config = config
        print('Config:')
        print(config)        
        
        # Set learning rate
        self.filename = "reinforce_1layer_weights.npy"
        self.learning_rate = config.get('learning_rate')  # Probably want to pass this as a parameter?
        self.discount_factor = config.get('discount_factor')
        self.crash_range = config.get('crash_range')  # Set edge of screen for early stopping
        self.n_episodes = config.get('episodes')
        self.max_steps = config.get('max_steps')
        self.rewards = config.get('rewards')
        self.actions = config.get('actions')
        self.actions = {int(k):v for k,v in self.actions.items()}
        
        # Create policy network
        self.include_angles = True
        self.n_states = 7
        n_hidden = config.get('hidden_layer')
        self.policy = PolicyNetwork(self.n_states, n_hidden, self.actions.__len__())
        
    def get_max_simulation_steps(self):
            return 1000 # You can alter the amount of steps you want your program to run for here
        
    def get_state_vector(self, drone: Drone):
        """
        Converts the Drone object into a flat numpy array of numbers 
        that the Neural Network can read.
        """
        # calculate dx, dy
        target = drone.get_next_target()
        dx = (target[0] - drone.x) 
        dy = (target[1] - drone.y) 
        
        # get velocities
        vx = drone.velocity_x 
        vy = drone.velocity_y 
        
        # get angles and angular velocity
        cos_theta = np.cos(drone.pitch)
        sin_theta = np.sin(drone.pitch)
        angular_vel = drone.pitch_velocity 
        
        return np.array([dx, dy, vx, vy, cos_theta, sin_theta, angular_vel])
    
    def get_thrusts(self, drone: Drone, mode='test') -> Tuple[float, float]:
        """
        Runs the policy network, then sample from the action distribution
        Return thrust pair for action
        """
        state = self.get_state_vector(drone)
        action_probs, _ = self.policy.forward(state)
        # print(f"Probs: {np.round(action_probs, 2)}")
        
        if mode == 'train':
            # Exploration: Sample an action during training
            action_index = np.random.choice(len(action_probs), p=action_probs)
            thrusts = self.actions[action_index]
            return thrusts, action_index, action_probs
        elif mode == 'test':
            # Greedy: Pick the best during testing
            action_index = np.argmax(action_probs)
            thrusts = self.actions[action_index]
            return thrusts
    
    def calculate_returns(self, rewards, discount_factor=0.99):
        """
        Calculate the discounted returns for each step in the episode
        Reverse calculation for efficiency - each step is only caluclated once
        Return array of returns per step as numpy array for learning calcs
        """
        returns = []
        G = 0
        for reward in reversed(rewards):
            G = reward + discount_factor * G
            returns.append(G)
        returns = np.array(list(reversed(returns)))
        # Normalise returns to a) stabilise gradients and b) give learning good/bad directions
        eps = 1e-8 # Tiny number to prevent division by zero
        returns = (returns - np.mean(returns)) / (np.std(returns) + eps) 
        return returns
    
    def train(self):
        # Set training parameters - needs to reset early in training because it'll rarely hit targets
        episodes = self.n_episodes  # How many times to fly
        max_steps = self.max_steps # Max time per episode
        delta_time = self.get_time_interval()
        best_score = -np.inf  # Initialise best score
        logger = ExperimentLogger(len(self.actions), self.n_states)
        
        for episode in range(episodes):
            drone = self.init_drone(mode='random')
            
            # Logging
            entropies = []
            grad_norms = []
            episode_states = [] # To store abs values
            episode_probs = []
            
            # Initialise episode data stores for learning 
            states = []
            actions = []
            rewards = []
            
            # Calculate initial distance to target - for reward shaping, using np norm
            target = drone.get_next_target()
            dist0 = np.linalg.norm([target[0] - drone.x, target[1] - drone.y])
            
            # Log previous acction for striction penalties
            prev_action = 0
            prev_action_count = 1
            
            for step in range(max_steps):
                
                # Run forward pass
                state = self.get_state_vector(drone)
                thrusts, action_index, action_probs = self.get_thrusts(drone, mode='train')
                
                # Logging 
                # Calculate Entropy of current state: -sum(p * log(p))
                entropy = -np.sum(action_probs * np.log(action_probs + 1e-9))
                entropies.append(entropy)
                episode_states.append(np.abs(state))
                episode_probs.append(action_probs)
                
                # Step simulation
                drone.set_thrust(thrusts)
                drone.step_simulation(delta_time)
                
                ######## REWARDS ########
                # Calculate distance to target
                target = drone.get_next_target()
                dist1 = np.linalg.norm([target[0] - drone.x, target[1] - drone.y])
                
                r_hit = drone.has_reached_target_last_update  # Target aqcuired reward. bool, so 0 if no hit, 1 if hit
                r_ddist = (dist0 - dist1) * (not r_hit)  # Change in distance reward. Positive means closer. 0 if we hit to avoid teleporting cost
                # r_step = 1  # Step penalty
                # r_exit = abs(drone.x) > self.crash_range or abs(drone.y) > self.crash_range  # Out of bounds penalty bool, so 0 if not out, 1 if out
                if prev_action == action_index:
                    r_strict = prev_action_count
                    prev_action_count += 1
                else:
                    r_strict = 0
                    prev_action = action_index
                    prev_action_count = 1
                
                # Calculate reward
                reward = 0
                reward += r_hit   * self.rewards['hit']  # Huge reward for hitting the target
                reward += r_ddist * self.rewards['ddist']   # Change in distance to target, positive means closer
                # reward += r_step  * self.rewards['step']     # -1 step penalty
                # reward += r_exit  * self.rewards['exit']   # Huge penalty for exiting bounds, ensure it's always actually penalised for going off screen
                reward += r_strict * self.rewards['strict']
                
                # Clean-up
                dist0 = dist1
                #########################
                
                # Store episode data
                states.append(state)
                actions.append(action_index)
                rewards.append(reward)
                
                # Early stopping if it goes off screen, to save early training time
                # if abs(drone.x) > self.crash_range or abs(drone.y) > self.crash_range:
                #     break
            
            # Get returns
            returns = self.calculate_returns(rewards, discount_factor=self.discount_factor)
            # print(f"Returns: {returns[-5:]}")
            # print(f"Rewards: {rewards[-5:]}")
            
            # Backward pass - update weights
            for t in range(len(states)):
                s_t = states[t]
                a_t = actions[t]
                g_t = returns[t]
                g_norm = self.policy.backward(s_t, a_t, g_t, learning_rate=self.learning_rate)
                
                # Logging: Capture the norm returned by backward
                grad_norms.append(g_norm)
                
            total_score = sum(rewards)
            if total_score > best_score:
                best_score = total_score
                # Save the weights to a file (or memory)
                self.save()
                print(f"New High Score: {best_score:.2f} (Saved)")
                
            # Logging: Log averages for the episode
            avg_state_mag = np.mean(np.array(episode_states), axis=0)
            avg_action_dist = np.mean(np.array(episode_probs), axis=0)
            
            logger.log_episode(
                reward=sum(rewards),
                length=len(states),
                avg_entropy=np.mean(entropies),
                grad_norm=np.mean(grad_norms),
                avg_state=avg_state_mag,
                avg_probs=avg_action_dist
            )
                
            if episode % 50 == 0:
                print(f"Episode {episode}: Total Score = {total_score:.2f}")
                
            # if episode % 500 == 499:  # Plot every 500 episodes, skipping the first and ensuring the last
            #     logger.plot()
                
        logger.plot()  # Show diagnostic plots at end
    
    def save(self):
        # Helper to save weights
        np.savez(self.filename, 
                 W1=self.policy.W1, b1=self.policy.b1, 
                 W2=self.policy.W2, b2=self.policy.b2)
                 
    def load(self):
        # Helper to load weights (Call this in main.py)
        try:
            data = np.load(self.filename)
            self.policy.W1 = data['W1']
            self.policy.b1 = data['b1']
            self.policy.W2 = data['W2']
            self.policy.b2 = data['b2']
            print("Loaded Best Pilot weights!")
        except:
            print("No saved weights found, using random.")
    
    
    
    

# # Run: python reinforce_controller.py to test the implementations
# if __name__ == "__main__":
#     print("--- Testing PolicyNetwork ---")
    
#     # Setup dimensions
#     n_inputs = 7   # State vector size
#     n_hidden = 16  # Hidden neurons
#     n_outputs = 5  # Actions (e.g. Up, Down, Left, Right, Hover)
    
#     # Instantiate the brain
#     brain = PolicyNetwork(n_inputs, n_hidden, n_outputs)
#     print(f"Network created with input={n_inputs}, hidden={n_hidden}, output={n_outputs}")
    
#     # Create a fake state (random numbers)
#     fake_state = np.random.rand(n_inputs)
#     print(f"Input State: {fake_state}")
    
#     # Run forward pass
#     action_probs, _ = brain.forward(fake_state)
#     print(f"Output Probabilities: {action_probs}")
    
#     # Sanity Checks
#     sum_probs = np.sum(action_probs)
#     print(f"Sum of probs: {sum_probs:.6f} (Should be 1.0)")
    
#     # Check if shapes are correct
#     assert action_probs.shape == (n_outputs,), f"Wrong shape: {action_probs.shape}"
#     assert np.isclose(sum_probs, 1.0), "Probabilities do not sum to 1.0!"
    
#     print("TEST PASSED: Neural net is working.")
    
    
#     controller = Reinforce1LayerController()
    
#     # Fake rewards: 3 steps of penalty, then a win
#     fake_rewards = [-1.0, -1.0, -1.0, 100.0] 
    
#     # Calculate
#     returns = controller.calculate_returns(fake_rewards, discount_factor=0.99)
    
#     print("\n--- TESTING RETURNS ---")
#     print(f"Rewards: {fake_rewards}")
#     print(f"Returns: {returns}")
    
#     # Manual Check
#     expected = [-1 + 0.99*(-1 + 0.99*(-1 + 0.99*100)), # ~94.06
#                 -1 + 0.99*(-1 + 0.99*100),             # ~96.02
#                 -1 + 0.99*100,                         # 98.0
#                 100.0]                                 # 100.0
    
#     print(f"Expected: {np.round(expected, 2)}")
