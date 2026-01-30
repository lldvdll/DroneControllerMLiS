# python3 SARSA_eval.py
from evaluation import evaluate, save_report
from SARSA_controller import CustomController

def main():
    ctrl = CustomController()
    ctrl.q_path = "runs/20260129_115824/q_table.npy"
    ctrl.load()

    # FIXED targets evaluation
    rep_fixed = evaluate(
        ctrl,
        eval_mode="fixed",
        n_episodes=50,
        seed=42,
        max_steps=3000,
        bounds=ctrl.full_bounds,
        dt=ctrl.dt,
    )
    save_report(rep_fixed, "Model_20260129_115824_eval_fixed_50ep_3000.json")

    # RANDOM targets evaluation
    rep_random = evaluate(
        ctrl,
        eval_mode="random",
        n_episodes=50,
        seed=43,
        max_steps=3000,
        bounds=ctrl.full_bounds,
        dt=ctrl.dt,
    )
    save_report(rep_random, "Model_20260129_115824_eval_random_50ep_3000.json")

if __name__ == "__main__":
    main()