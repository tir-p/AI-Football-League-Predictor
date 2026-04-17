from common import RESULTS_DIR, ensure_results_dir, save_json
from run_ablation import run_experiment as run_ablation_experiment
from run_calibration_analysis import run_experiment as run_calibration_experiment
from run_feature_expansion import run_experiment as run_feature_expansion_experiment
from run_feature_importance import run_experiment as run_feature_importance_experiment
from run_model_comparison import run_experiment as run_model_comparison_experiment
from run_out_of_sample_validation import run_experiment as run_out_of_sample_validation_experiment


def main():
    ensure_results_dir()

    summary = {
        "feature_expansion": run_feature_expansion_experiment(),
        "ablation": run_ablation_experiment(),
        "model_comparison": run_model_comparison_experiment(),
        "calibration": run_calibration_experiment(),
        "out_of_sample_validation": run_out_of_sample_validation_experiment(),
        "feature_importance": run_feature_importance_experiment(),
    }
    save_json(RESULTS_DIR / "all_experiment_summaries.json", summary)

    print("All planned experiments completed.")
    print(f"Results written to: {RESULTS_DIR}")


if __name__ == "__main__":
    main()
