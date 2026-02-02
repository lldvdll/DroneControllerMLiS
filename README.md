# Drone Flight Controller, Project 9 MLiS Part I

To get started you need to first setup python on your computer. The version of python used to develop this initial code is Python 3.6.12. The list of modules installed is given in requirements.txt. The only modules you need to run this code are `numpy` and `pygame`. You can also install `matplotlib` if you intend to plot any figures.

We recommend that the group doing this project using this repository as a template to create their own and invite all members. Collaboration via Git is the best way to let each member work on the code at the same time. Refer to the Git video on Moodle if you are not familiar with Git or GitHub.

Once everyone has access to the repository on GitHub, clone your repository to your own computer. You can use [GitHub Desktop](https://desktop.github.com/) to make this very easy.

## Installing Python packages

Once you have cloned the repository, open up the folder in your code editor. We recommend you to use [VS Code](https://code.visualstudio.com/). Open the folder where you downloaded the repository. Now, in VS Code, open a terminal by choosing `Terminal -> New Terminal`. This should open in the root folder of the repository. From here, install the dependencies with the command:
```sh
pip install -r requirements.txt
```

## Running the code

In order to run the code, open up a terminal and run
```sh
python main.py
```
You can also run this in the normal way supported by your IDE.

In order to run your own algorithm with the visualisation, you need to change the code at the top of `main.py`, which is detailed in the comments.

## Changes to main
IS_TRAINING = False to run pygame simulation
CONTROLLER specifies which controller to use.
generate_controller() contains a list of active models for each controller. Comment/uncomment, or specify new config or q_path to select a specific model.
Reward overlay and cummulative reward plot have been added. These may be toggled on/off with PLOT_REWARDS and DISPLAY_REWARDS.


## Task

We have included a file `custom_controller.py` for you to write your own custom controller. You can also create any new files you need for scripts in order to train your controller. You can simulate your own environments and even change the way in which targets are spawned in order to more effectively train your agent. An example of the code needed to run your controller is given below:

```python
# Number of time steps
max_time = 1000
# Create the drone
drone = controller.init_drone()
for i in range(max_time):
    # Controller decides on an action
    action = controller.get_thrusts(drone)
    # Apply the action to the environment
    drone.set_thrust(action)
    # Update the simulation
    drone.step_simulation(delta_time)
    # TODO: Add in any code for calculating a reward
```

## Bug Fixes

1. 26/11/2020: A quick fix to the implementations of the drag dynamics on the drone.

## Possible Extensions

1. Increase the difficulty of the simulation by making sure the drone stays within the boundary of the screen.
2. Decrease the value of the drag constants in drone.py to make the simulation more sensitive.
3. Introduce barriers for the drone to avoid while also hitting the target.
4. Extend the simulation to include two drones which avoid colliding with one another, but which still have to hit targets.


## Controllers
### FlightController
flight_controller.py has been edited to:
1. Add attributes required for evaluations
2. Update init_drone() with target modes: fixed (original 4 targets), hover (all at (0,0)), random (with constraints), random_simple (without constraints), increasing (random, but increasing per episode).

### REINFORCE
neural_reinforce_controller.py implements REINFORCE with a neural network. It contains:
1. NeuralReinforceController: Extends FlightController. Reads training, test, and reward config from neural_reinforce_congif.json. Can also pass custom config during init.
2. ExperimentLogger: Logs intermediate per episode metrics for training diagnostics. Plots/saves to experiments/neural_reinforce at end of training run.
3. Policy: Neural network model with back propagation. Called by NeuralReinforceController.train() for trajectory sampling and model updates. Called by NeuralReinforceController.get_thrusts() for testing.
4. RewardManager: Calculates rewards and maintains relevant state/action variables. Gets reward structure from neural_reinforce_congif.json - rewards.

Config in neural_reinforce_congif.json. Contains hyperparameters, reward settings, and save/load locations. "experiment_name" determines save/load settings, loading files from experiments/neural_reinforce in NeuralReinforceController(test_mode=True). During training, copies config, weights, and diagnostics to experiments/neural_reinforce. Doesn't check for duplicate names or overwrite!

### SARSA

<TODO: Complete this.>


## Additional files

1. train.py - used for training SARSA_controller
2. comparison.py - used to compare different models, specified in dictionary at top of file. A template for each controller type is included. Runs evaluate() on each model with same game and target settings, generates performance tables
3. evaluate.py - used to evaluate SARSA models. Generates ...<TODO: explain>
4. analysis.py - <TODO: Explain>
5. SARSA_eval.py - <TODO: Explain>
6. Heuristic_eval.py - <TODO: Explain>
7 Report Plot.py - <TODO: Explain>
