import numpy as np
import json
from flight_controller import FlightController
from drone import Drone


class Policy():
    def __init__(self):
        pass
    
    def forward(self, state):
        pass
    
    def backward(self):
        pass
    
    def update(self):
        """
        Updates policy weights from returns using REINFORCE
        """
        pass

class ContinuousLinearController(FlightController):
    def __init__(self):
        super().__init__()
        
        # Load config
        with open('continuous_linear_config.json', 'r') as f:
            self.config = json.load(f)
        
        # Initialise policy
        self.policy = Policy()

    def get_state(self, drone: Drone):
        """
        Get state from drone object
        """
        pass
    
    def get_action(self, state, mode='test'):
        """
        Get action from policy
        """
        pass
    
    def convert_action_to_thrust(self, action):
        """
        Convert action to thrusts
        """
        pass

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
        pass
    
    def get_returns(self, rewards):
        """
        Convert rewards to returns for training
        """
        pass

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
                action = self.get_action(drone, state)
                
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