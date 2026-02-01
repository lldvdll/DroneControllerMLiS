# python3 SARSA_eval.py
from evaluation import evaluate, save_report
from SARSA_controller import CustomController

def main():
    ctrl = CustomController()
    ctrl.q_path = "runs/20260129_115824/q_table.npy"
    ctrl.load()

    # Fixed targets evaluation
    rep_fixed = evaluate(
        ctrl,
        eval_mode="fixed",
        n_episodes=1,
        max_steps=5000,
        bounds=ctrl.full_bounds,
        log_trajectory = True,
        traj_filename = "Evaluation/Sarsa_slow_eval_fixed_trajectory"
    )
    save_report(rep_fixed, "Evaluation/Sarsa_slow_eval_fixed_metrics_v2.json")

    # Random targets evaluation
    rep_random = evaluate(
        ctrl,
        eval_mode="random",
        n_episodes=100,
        seed=43,
        max_steps=6000,
        bounds=ctrl.full_bounds,
        log_trajectory = False
    )
    save_report(rep_random, "Evaluation/Sarsa_slow_eval_random_metrics_v2.json")

if __name__ == "__main__":
    main()