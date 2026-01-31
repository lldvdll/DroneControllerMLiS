# python3 SARSA_eval.py
from evaluation import evaluate, save_report
from SARSA_controller import CustomController

def main():
    ctrl = CustomController()
    ctrl.q_path = "runs/20260129_231302/q_table.npy"
    ctrl.load()

    # FIXED targets evaluation
    rep_fixed = evaluate(
        ctrl,
        eval_mode="fixed",
        n_episodes=1,
        max_steps=5000,
        bounds=ctrl.full_bounds,
        log_trajectory = True,
        traj_filename = "Evaluation/Sarsa_fast_eval_fixed_trajectory"
    )
    save_report(rep_fixed, "Evaluation/Sarsa_fast_eval_fixed_5000_metrics.json")

    # RANDOM targets evaluation
    rep_random = evaluate(
        ctrl,
        eval_mode="random",
        n_episodes=100,
        seed=43,
        max_steps=5000,
        bounds=ctrl.full_bounds,
        log_trajectory = False
    )
    save_report(rep_random, "Evaluation/Sarsa_fast_eval_random_5000_metrics.json")

if __name__ == "__main__":
    main()