import pygame
from drone import Drone
from pygame import Rect
import numpy as np
import math
from typing import Tuple
from flight_controller import FlightController
from matplotlib import pyplot as plt

#---------------------WRITE YOUR OWN CODE HERE------------------------#
from heuristic_controller import HeuristicController
# from custom_controller import CustomController
from neural_reinforce_controller import NeuralReinforceController


def generate_controller() -> FlightController:
    # return HeuristicController() # <--- Replace this with your own written controller
    return NeuralReinforceController()

def is_training() -> bool:
    return True # <--- Replace this with True if you want to train, false otherwise
def is_saving() -> bool:
    return True # <--- Replace this with True if you want to save the results of training, false otherwise

#---------------------------------------------------------------------#
SCREEN_WIDTH = 720
SCREEN_HEIGHT = 480

# 'deterministic': the 4 fixed targets provided, 'random': generates NUM_TARGETS randomly distributed targets
TARGET_MODE = 'random'
NUM_TARGETS = 100

# Reward diagnostics - requires controller to have a get_reward() method return a specific data structure
PLOT_REWARDS = True  # Plot reward distribution after simulation
DISPLAY_REWARDS = True  # Show live rewards on game screen

def get_scale():
    return min(SCREEN_HEIGHT, SCREEN_WIDTH)

def convert_to_screen_coordinate(x,y):
    scale = get_scale()
    return (x*scale + SCREEN_WIDTH/2, -y*scale + SCREEN_HEIGHT/2)

def convert_to_screen_size(game_size):    
    scale = get_scale()
    return game_size*scale

def convert_to_game_coordinates(x,y):
    scale = get_scale()
    return ((x - SCREEN_WIDTH/2)/scale, (y - SCREEN_HEIGHT/2)/scale)

def main(controller: FlightController):
    
    global PLOT_REWARDS
    global DISPLAY_REWARDS

    # Initialise pygame
    pygame.init()
    clock = pygame.time.Clock()
    
    # Initialise fonts for rewards display
    pygame.font.init()
    font = pygame.font.SysFont('Arial', 16)

    # Load the relevant graphics into pygame
    drone_img = pygame.image.load('graphics/drone_small.png')
    background_img = pygame.image.load('graphics/background.png')
    target_img = pygame.image.load('graphics/target.png')

    # Create the screen
    screen = pygame.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT))
    
    # Initalise the drone
    drone = controller.init_drone(mode=TARGET_MODE, num_targets=NUM_TARGETS)  # I've changed this so that more targets are generated randomly
    
    simulation_step_counter = 0
    max_simulation_steps = controller.get_max_simulation_steps()
    delta_time = controller.get_time_interval()

    # Track cumulative rewards
    rewards_cum = {}
    reward_history = []

    # Run the simulation
    running = True
    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False


        # --- Begin Physics --- #
        # Get the thrust information from the controller
        drone.set_thrust(controller.get_thrusts(drone))
        # Update the simulation
        drone.step_simulation(delta_time)
        
        # Get rewards for display and plot
        if PLOT_REWARDS:
            try:
                _, rewards = controller.get_reward(drone)
                for k,v in rewards.items():  # Accumulate rewards for display
                    rewards_cum[k] = rewards_cum.get(k, 0) + v        
                reward_history.append(rewards_cum.copy())  # Track reward history for final plot
            except:
                DISPLAY_REWARDS = False
                PLOT_REWARDS = False
            

        # --- Begin Drawing --- #

        # Refresh the background
        screen.blit(background_img, (0,0))
        # Draw the current drone on the screen
        draw_drone(screen, drone, drone_img)
        # Draw the next target on the screen
        draw_target(drone.get_next_target(), screen, target_img)
        
        # Draw reward counter on the screen
        if DISPLAY_REWARDS:
            draw_rewards(screen, font, rewards_cum)

        # Actually displays the final frame on the screen
        pygame.display.flip()

        # Makes sure that the simulation runs at a target 60FPS
        clock.tick(60)

        # Checks whether to reset the current drone
        simulation_step_counter+=1
        if (simulation_step_counter>max_simulation_steps):
            drone = controller.init_drone() # Reset the drone
            simulation_step_counter = 0
            
    # Close the program
    pygame.quit()
    
    # Plot the reward history
    if PLOT_REWARDS:
        plot_rewards_history(reward_history)


def draw_rewards(screen, font, rewards):
    """Overlay cumulative rewards on screen with %'s to inform reward shaping"""
    x_offset = 10
    y_offset = 10

    # Sort keys so they don't jump around
    for key in sorted(rewards.keys()):
        val = rewards[key]
        text_str = f"{key}: {val:.2f}"
        
        # Render text (Black)
        text_surface = font.render(text_str, True, (0, 0, 0))
        
        # Draw to screen
        screen.blit(text_surface, (x_offset, y_offset))
        y_offset += 20 # Move down for next line    

def draw_target(target_point, screen, target_img):
    target_size = convert_to_screen_size(0.1)
    point_x, point_y = convert_to_screen_coordinate(*target_point)
    screen.blit(pygame.transform.scale(target_img, (int(target_size), int(target_size))), (point_x-target_size/2, point_y-target_size/2))

def draw_drone(screen: pygame.Surface, drone: Drone, drone_img: pygame.Surface):
    drone_x, drone_y = convert_to_screen_coordinate(drone.x, drone.y)
    drone_width = convert_to_screen_size(0.3)
    drone_height = convert_to_screen_size(0.15)
    drone_rect = Rect(drone_x-drone_width/2, drone_y-drone_height/2, drone_width, drone_height)
    drone_scaled_img = pygame.transform.scale(drone_img, (int(drone_width), int(drone_height)))
    drone_scaled_center = drone_scaled_img.get_rect(topleft = (drone_x-drone_width/2, drone_y-drone_height/2)).center
    rotated_drone_img = pygame.transform.rotate(drone_scaled_img, -drone.pitch * 180 / math.pi)
    drone_scaled_rect = rotated_drone_img.get_rect(center=drone_scaled_center)
    screen.blit(rotated_drone_img, drone_scaled_rect)
    
    
def plot_rewards_history(history):
    """
    Generates a stacked area plot of reward components over time.
    Separates positive and negative components for clarity.
    """
    if not history:
        print("No reward history to plot.")
        return

    # 1. Organize data
    keys = set().union(*history)
    # Filter out 'total' if your controller logs it, so we don't double count
    component_keys = [k for k in keys if k != 'total']
    steps = range(len(history))
    
    # 2. Setup Plot
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # 3. Stack Calculations
    pos_bottom = np.zeros(len(history))
    neg_bottom = np.zeros(len(history))
    
    # 4. Plot Stacks
    for key in component_keys:
        # Get data for this key across all steps
        values = np.array([step.get(key, 0.0) for step in history])
        
        # Split into positive and negative parts
        pos_vals = np.maximum(values, 0)
        neg_vals = np.minimum(values, 0)
        
        # Plot positive stack (upwards)
        p = ax.fill_between(steps, pos_bottom, pos_bottom + pos_vals, label=key, alpha=0.6)
        pos_bottom += pos_vals
        
        # Plot negative stack (downwards)
        # Use same color as positive part for consistency
        ax.fill_between(steps, neg_bottom, neg_bottom + neg_vals, color=p.get_facecolor(), alpha=0.6)
        neg_bottom += neg_vals

    # 5. Plot Total Net Reward (The Black Line)
    # This shows what the agent actually "feels" (Sum of all components)
    totals = np.array([sum(step.values()) for step in history])
    ax.plot(steps, totals, color='black', linewidth=1.5, linestyle='--', label='Net Total')

    # 6. Final Polish
    ax.set_title("Reward Evolution Over Time (Components)")
    ax.set_xlabel("Simulation Step")
    ax.set_ylabel("Reward Magnitude")
    ax.legend(loc='upper left', bbox_to_anchor=(1, 1))
    ax.grid(True, alpha=0.3)
    ax.axhline(0, color='black', linewidth=0.5)
    
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":

    controller = generate_controller()
    if is_training():
        controller.train()
        if is_saving():
            controller.save()        
    else:
        controller.load()
    
    main(controller)