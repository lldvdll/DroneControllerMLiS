from evaluation import evaluate
import pandas as pd
import numpy as np
import re
from heuristic_controller import HeuristicController
from neural_reinforce_controller import NeuralReinforceController
from SARSA_controller import CustomController as SARSAController


MODELS = {
    'Heuristic': {
        'controller': 'HeuristicController',
    },
    'SARSA Conservative': {
        'controller': 'SARSAController',
        'load_args': dict(q_path = "runs/20260129_115824/q_table.npy",
                          k_omega_arrest_motion = 0.35)
    },
    'SARSA Agressive': {
        'controller': 'SARSAController',
        'load_args': dict(q_path = "runs/20260129_231302/q_table.npy",
                          k_omega_arrest_motion = 0.5)
    },
    # 'Neural REINFROCE (60_reward_shaping_phase1 best)': {
    #     'controller': 'NeuralReinforceController',
    #     'load_args': dict(filename='60_reward_shaping_phase1', mode='best')
    # },
    # 'Neural REINFROCE (63_increase_gamma best)': {
    #     'controller': 'NeuralReinforceController',
    #     'load_args': dict(filename='63_increase_gamma', mode='best')
    # },
    # 'Neural REINFROCE (73_reset_curriculum best)': {
    #     'controller': 'NeuralReinforceController',
    #     'load_args': dict(filename='73_reset_curriculum', mode='best')
    # },
    # 'REINFROCE (90_continue_hover)': {
    #     'controller': 'NeuralReinforceController',
    #     'load_args': dict(filename='90_continue_hover', mode='best')
    # },
    'REINFROCE': {
        'controller': 'NeuralReinforceController',
        'load_args': dict(filename='106_movement', mode='best')
    },
    # 'Neural REINFROCE (100_movement best)': {
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
        
    return model


def process_comparison_table(df):
    
    df['success_rate'] = (df['success_rate'] * 100).round(1)
    df['crash_rate'] = (df['crash_rate'] * 100).round(1)
    df['targets_reached'] = df['targets_reached'].round(2).astype(str)+' \u00B1 '+df['targets_reached_std'].round(2).astype(str)
    df['path_efficiency'] = df['path_efficiency'].round(2).astype(str)+' \u00B1 '+df['path_efficiency_std'].round(2).astype(str)
    df['pitch_variance'] = df['pitch_variance'].round(3).astype(str)+' \u00B1 '+df['pitch_variance_std'].round(3).astype(str)
    df = df[['crash_rate', 'targets_reached', 'time_to_target', 'path_efficiency', 'pitch_variance']]
        
    df = df.rename(columns={
        'success_rate': 'Success Rate (%)',
        'crash_rate': 'Crash Rate (%)',
        'targets_reached': 'Targets Reached',
        'time_to_target': 'Time to Target (Median)',
        'path_efficiency': 'Path Efficiency',
        'pitch_variance': 'Stability (Pitch Variance)'
    })
    
    return df


def df_to_latex_table(df, name):
    latex = (
        df.style
        .format(precision=2)
        .set_properties(**{"text-align": "right"})
        .to_latex(
            hrules=True,
            position="H",
            caption="My results",
            label="tab:results"
        )
    )
    
    latex = latex.replace('\u00B1', r'$\pm$')

    with open(f"results/comparison_stats_{name}.tex", "w") as f:
        f.write(latex)

def run_comparison(
        name, 
        n_targets_random=5, 
        n_episodes=50, 
        max_steps=5000, 
        bounds=(-0.75, 0.75, -0.5, 0.5)
        ):
    
    models = MODELS
    performance = {}
    for model_name, model in models.items():
        print(f'Evaluating model {model_name}...')
        
        # Initialise model and load paramters
        model = load_controller(model)
        model.target_mode = 'random'
        model.n_targets_random = n_targets_random
        
        # Fixed targets evaluation to generate trajectory plots
        rep_fixed = evaluate(
            model,
            eval_mode="fixed",
            n_episodes=1,
            max_steps=max_steps,
            bounds=bounds,
            log_trajectory = True,
            traj_filename = f"results/trajectories_{model_name}_{name}"
        )
        
        # Evaluate model on random targets
        metrics = evaluate(
            model,
            eval_mode='random',
            n_episodes=n_episodes,
            seed=np.random.randint(0, 1000),
            max_steps=max_steps,
            bounds=bounds,
            return_mode='metrics'
            )
        
        performance[model_name] = metrics
        
    # Process performance metrics into styled comparison table
    df = pd.DataFrame(performance).T
    df = process_comparison_table(df)
    print(df)
        
    # Save table for report
    df_to_latex_table(df, name)
    
    
if __name__ == "__main__":
    n_episodes = 50
    run_comparison(
        'bounded',  
        n_episodes=n_episodes,
        n_targets_random=5, 
        bounds=(-0.75, 0.75, -0.5, 0.5)
        )
    run_comparison(
        'unbounded', 
        n_episodes=n_episodes,
        n_targets_random=10, 
        bounds=(-10, 10, -10, 10)
        )