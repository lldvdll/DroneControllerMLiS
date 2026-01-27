"""
Training Script for SARSA Drone Controller

Features:
- Multiple target modes (fixed, random, curriculum)
- Automatic experiment tracking in runs/ directory
- Save Q-table, plots, and evaluation reports
- Resume training from checkpoints

Usage:
    python3 train.py --mode curriculum --episodes 12000 --eval-after
    python3 train.py --load runs/20260117_004246/q_table.npy --eval-after --episodes 0
"""

import argparse
import os
from datetime import datetime
from SARSA_controller import CustomController
from evaluation import evaluate, print_report, save_report
from pathlib import Path
import analysis


def parse_args():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(
        description="Train SARSA drone controller with experiment tracking",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    # Training mode
    parser.add_argument(
        "--mode", 
        type=str, 
        default="curriculum",
        choices=["fixed", "random", "curriculum"],
        help="Target generation mode"
    )
    
    # Training parameters
    parser.add_argument(
        "--episodes", 
        type=int, 
        default=5000,
        help="Number of training episodes"
    )
    
    parser.add_argument(
        "--save-every", 
        type=int, 
        default=200,
        help="Save checkpoint every N episodes"
    )
    
    parser.add_argument(
        "--print-every", 
        type=int, 
        default=50,
        help="Print progress every N episodes"
    )
    
    # Model persistence
    parser.add_argument(
        "--load", 
        type=str,
        default=None,
        help="Path to existing Q-table to load (e.g., runs/2024-01-15_143022/q_table.npy)"
    )
    
    parser.add_argument(
        "--run-name",
        type=str,
        default=None,
        help="Custom name for this run (default: timestamp)"
    )
    
    # Evaluation
    parser.add_argument(
        "--eval-after", 
        action="store_true",
        help="Run evaluation after training and save reports"
    )
    
    parser.add_argument(
        "--eval-episodes", 
        type=int, 
        default=30,
        help="Number of evaluation episodes"
    )
    
    parser.add_argument(
        "--eval-seed",
        type=int,
        default=42,
        help="Random seed for evaluation"
    )
    
    # Analysis
    parser.add_argument(
        "--no-analysis",
        action="store_true",
        help="Skip generating analysis plots"
    )
    
    parser.add_argument(
        "--analysis-window",
        type=int,
        default=50,
        help="Rolling window size for analysis plots"
    )
    
    return parser.parse_args()


def main():
    """Main training workflow with experiment tracking"""
    args = parse_args()
    
    # Create run directory with timestamp
    if args.run_name:
        run_id = args.run_name
    else:
        run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    run_dir = os.path.join("runs", run_id)
    os.makedirs(run_dir, exist_ok=True)
    
    # Paths for saving
    q_path = os.path.join(run_dir, "q_table.npy")
    log_path = os.path.join(run_dir, "q_table_train_log.jsonl")
    csv_log_path = os.path.join(run_dir, "q_table_train_log.csv")
    eval_fixed_path = os.path.join(run_dir, "eval_fixed.json")
    eval_random_path = os.path.join(run_dir, "eval_random.json")
    
    # Print configuration
    print("=" * 70)
    print("SARSA Drone Controller Training")
    print("=" * 70)
    print(f"Run ID:        {run_id}")
    print(f"Output dir:    {run_dir}")
    print(f"Target mode:   {args.mode}")
    print(f"Episodes:      {args.episodes}")
    print(f"Load from:     {args.load if args.load else 'None (fresh start)'}")
    print("=" * 70)
    
    # Initialise controller
    ctrl = CustomController()
    ctrl.target_mode = args.mode
    ctrl.q_path = q_path
    ctrl.log_path = log_path
    ctrl.csv_log_path = csv_log_path
    
    # Load existing Q-table if specified
    if args.load:
        if os.path.exists(args.load):
            # Temporarily change path to load from specified location
            temp_path = ctrl.q_path
            ctrl.q_path = args.load
            ctrl.load()
            ctrl.q_path = temp_path  # Restore save path
            print(f"[Load] Q-table loaded from {args.load}")
        else:
            print(f"[Warning] Q-table not found at {args.load}, starting fresh")
    
    # Train
    print("\n[Train] Starting training...")
    print("-" * 70)
    ctrl.train(
        num_episodes=args.episodes,
        save_every=args.save_every,
        print_every=args.print_every,
    )
    
    # Save final Q-table
    ctrl.save()
    print(f"\n[Save] Q-table saved to {q_path}")
    
    # Generate analysis plots
    if not args.no_analysis:
        if os.path.exists(log_path):
            print(f"\n[Analysis] Generating diagnostic plots from {log_path}...")
            try:
                analysis.run_analysis(
                    log_path=Path(log_path), 
                    out_dir=Path(run_dir), 
                    window=args.analysis_window
                )
            except ImportError:
                print("[Analysis] Warning: analysis.py not found, skipping plots")
            except Exception as e:
                print(f"[Analysis] Error generating plots: {e}")
        else:
            print(f"[Analysis] No log file found at {log_path}")
    
    # Evaluation
    if args.eval_after:
        print("\n" + "=" * 70)
        print("EVALUATION (Greedy Policy)")
        print("=" * 70)
        
        # Fixed targets evaluation
        print("\n[Eval] Running on FIXED targets...")
        rep_fixed = evaluate(
            ctrl, 
            eval_mode="fixed", 
            n_episodes=args.eval_episodes, 
            seed=args.eval_seed
        )
        print_report(rep_fixed, title="Fixed Targets")
        save_report(rep_fixed, eval_fixed_path)
        
        # Random targets evaluation
        print("\n[Eval] Running on RANDOM targets...")
        rep_random = evaluate(
            ctrl, 
            eval_mode="random", 
            n_episodes=args.eval_episodes, 
            seed=args.eval_seed + 1
        )
        print_report(rep_random, title="Random Targets")
        save_report(rep_random, eval_random_path)
    
    # Summary
    print("\n" + "=" * 70)
    print("TRAINING COMPLETE!")
    print("=" * 70)
    print(f"Results saved in: {run_dir}/")
    print(f"  - Q-table:           q_table.npy")
    print(f"  - Training log:      q_table_train_log.jsonl")
    if not args.no_analysis and os.path.exists(log_path):
        print(f"  - Plots:             *.png")
    if args.eval_after:
        print(f"  - Eval fixed:        eval_fixed.json")
        print(f"  - Eval random:       eval_random.json")
    print("=" * 70)


if __name__ == "__main__":
    main()












































