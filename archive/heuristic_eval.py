# python3 heuristic_eval.py
from evaluation import evaluate, save_report
from heuristic_controller import HeuristicController

def main():
    ctrl = HeuristicController()

    # Fixed targets evaluation
    rep_fixed = evaluate(
        ctrl,
        eval_mode="fixed",
        n_episodes=1,
        max_steps= 3000,
        bounds=ctrl.full_bounds,
        log_trajectory = True,
        traj_filename = "Evaluation/heuristic_eval_fixed_trajectory"
    )
    save_report(rep_fixed, "Evaluation/heuristic_eval_fixed_metrics.json")

    # Random targets evaluation
    rep_random = evaluate(
        ctrl,
        eval_mode="random",
        n_episodes=100,
        seed=43,
        max_steps= 7000,
        bounds=ctrl.full_bounds,
        log_trajectory = False
    )
    save_report(rep_random, "Evaluation/heuristic_eval_random_metrics.json")

if __name__ == "__main__":
    main()