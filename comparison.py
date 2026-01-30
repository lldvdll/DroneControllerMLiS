from evaluation import evaluate
import pandas as pd
from heuristic_controller import HeuristicController
from neural_reinforce_controller import NeuralReinforceController
from SARSA_controller import CustomController as SARSAController




MODELS = {
    'Heuristic': {
        'controller': 'HeuristicController',
        'init_args': dict(),
        'load_args': dict()
    },
    'Neural Reinforce (ph1 latest)': {
        'controller': 'NeuralReinforceController',
        'init_args': dict(),
        'load_args': dict(filename='60_reward_shaping_phase1', mode='latest')
    },
    'Neural Reinforce (ph1 best)': {
        'controller': 'NeuralReinforceController',
        'init_args': dict(),
        'load_args': dict(filename='60_reward_shaping_phase1', mode='best')
    },
    'Neural Reinforce (ph2 latest)': {
        'controller': 'NeuralReinforceController',
        'init_args': dict(),
        'load_args': dict(filename='63_increase_gamma', mode='latest')
    },
    'Neural Reinforce (ph2 best)': {
        'controller': 'NeuralReinforceController',
        'init_args': dict(),
        'load_args': dict(filename='63_increase_gamma', mode='best')
    },
    'SARSA': {
        'controller': 'SARSAController',
        'init_args': dict(),
        'load_args': dict(q_path = "runs/20260128_203007/q_table.npy")
    }
}

def plot_evaluation(models):
    pass


def load_controller(spec):
    
    if spec['controller'] == 'HeuristicController':
        model = HeuristicController()
        model.target_mode = None
    elif spec['controller'] == 'NeuralReinforceController':
        model = NeuralReinforceController()
        model.load(**spec['load_args'])
        model.target_mode = None
    elif spec['controller'] == 'SARSAController':
        model = SARSAController()
        model.target_mode = 'random'
        model.q_path = spec['load_args']['q_path']
        
    return model
    

def run_comparison():
    
    models = MODELS
    performance = {}
    for model_name, model in models.items():
        print(f'Evaluating model {model_name}...')
        
        # initialise model and load paramters
        model = load_controller(model)
        
        # evaluate
        metrics = evaluate(
            model,
            eval_mode='random',
            n_episodes=20,
            seed=40,
            max_steps=5000,
            # bounds=,
            return_mode='metrics',
            dt = 0.01
            )
        
        performance[model_name] = metrics
        
    df = pd.DataFrame(performance).T
    print(df)
        
    plot_evaluation(models)
    print('Done!')
    
    
if __name__ == "__main__":
    run_comparison()