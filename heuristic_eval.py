from evaluation import evaluate, print_report, save_report
from heuristic_controller import HeuristicController

def main():
    ctrl = HeuristicController()

    # Provide dt/max_steps explicitly because HeuristicController doesn't define ctrl.dt / ctrl.max_steps
    # dt: use the standard sim step (match your training/eval setup; 0.01 is what SARSA uses)
    dt = 0.01
    max_steps = ctrl.get_max_simulation_steps()

    # FIXED targets evaluation
    rep_fixed = evaluate(
        ctrl,
        eval_mode="fixed",
        n_episodes=30,
        seed=42,
        max_steps=max_steps,
        bounds=ctrl.full_bounds,
        dt=dt,
    )
    save_report(rep_fixed, "heuristic_eval_fixed_30ep_4T.json")

    # RANDOM targets evaluation
    rep_random = evaluate(
        ctrl,
        eval_mode="random",
        n_episodes=30,
        seed=43,
        max_steps=max_steps,
        bounds=ctrl.full_bounds,
        dt=dt,
    )
    save_report(rep_random, "heuristic_eval_random_30ep_5T.json")

if __name__ == "__main__":
    main()
