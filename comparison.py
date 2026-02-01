from evaluation import evaluate
import pandas as pd
import numpy as np
from heuristic_controller import HeuristicController
from neural_reinforce_controller import NeuralReinforceController
from SARSA_controller import CustomController as SARSAController


MODELS = {
    'Heuristic': {
        'controller': 'HeuristicController',
    },
    'SARSA slow': {
        'controller': 'SARSAController',
        'load_args': dict(q_path = "runs/20260129_115824/q_table.npy",
                          k_omega_arrest_motion = 0.35)
    },
    'SARSA fast': {
        'controller': 'SARSAController',
        'load_args': dict(q_path = "runs/20260129_231302/q_table.npy",
                          k_omega_arrest_motion = 0.5)
    },
    # 'Neural Reinforce (60_reward_shaping_phase1 best)': {
    #     'controller': 'NeuralReinforceController',
    #     'load_args': dict(filename='60_reward_shaping_phase1', mode='best')
    # },
    # 'Neural Reinforce (63_increase_gamma best)': {
    #     'controller': 'NeuralReinforceController',
    #     'load_args': dict(filename='63_increase_gamma', mode='best')
    # },
    'Neural Reinforce (73_reset_curriculum best)': {
        'controller': 'NeuralReinforceController',
        'load_args': dict(filename='73_reset_curriculum', mode='best')
    },
    # 'Neural Reinforce (90_continue_hover best)': {
    #     'controller': 'NeuralReinforceController',
    #     'load_args': dict(filename='90_continue_hover', mode='best')
    # },
    # 'Neural Reinforce (100_movement best)': {
    #     'controller': 'NeuralReinforceController',
    #     'load_args': dict(filename='102_movement_vb', mode='best')
    # },
}

def plot_evaluation(models):
    pass


def load_controller(spec):
    
    if spec['controller'] == 'HeuristicController':
        model = HeuristicController()
        
    elif spec['controller'] == 'NeuralReinforceController':
        model = NeuralReinforceController(test_mode=True)
        model.load(**spec['load_args'])
        
    elif spec['controller'] == 'SARSAController':
        model = SARSAController()
        model.q_path = spec['load_args']['q_path']
        model.k_omega_arrest_motion = spec['load_args']['k_omega_arrest_motion']
        model.load()
        
    model.target_mode = 'random'
        
    return model
    

def run_comparison():
    
    models = MODELS
    performance = {}
    for model_name, model in models.items():
        print(f'Evaluating model {model_name}...')
        
        # initialise model and load paramters
        model = load_controller(model)
        
        # get bounds
        full_bounds = (-0.75, 0.75, -0.5, 0.5)
        
        # evaluate
        metrics = evaluate(
            model,
            eval_mode='random',
            n_episodes=50,
            seed=np.random.randint(0, 1000),
            max_steps=6000,
            bounds=full_bounds,
            return_mode='metrics'
            # dt = 0.01
            )
        
        performance[model_name] = metrics
        
    df = pd.DataFrame(performance).T
    print(df)
        
    plot_evaluation(models)
    print('Done!')
    
    
if __name__ == "__main__":
    run_comparison()