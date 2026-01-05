"""Test ensemble and TTA on existing models."""

from configs.config import Config
from data_factory.data_factory import DataFactory
from data_factory.ROP_dataset import ROPDataset
from train_and_val_worker import TrainAndEvalWorker
from utils.utils import set_seed


def main():
    """Test ensemble and TTA."""
    
    print("=" * 80)
    print("Testing Ensemble + TTA on Existing Models")
    print("=" * 80)
    
    config = Config()
    set_seed(config.random_state)
    
    # Initialize DataFactory
    print("\n[1/3] Loading test data...")
    data_factory = DataFactory(config)
    train_df, test_df = data_factory.create_metadata_dataframe()
    
    # Create test dataset
    test_dataset = ROPDataset(
        dataframe=test_df,
        config=config,
        is_train=False
    )
    
    print(f"Test set: {len(test_df)} images")
    
    # Initialize worker
    worker = TrainAndEvalWorker(config)
    
    # Test 1: Best single model without TTA (baseline)
    print("\n" + "=" * 80)
    print("[2/3] TEST 1: Best Single Model (Fold 1) - No TTA")
    print("=" * 80)
    
    best_model_path = "outputs/checkpoints/v1.0/best_model_fold1.pth"
    results_baseline = worker.evaluate(
        test_dataset=test_dataset,
        model_path=best_model_path
    )
    
    # Test 2: Best single model WITH TTA
    print("\n" + "=" * 80)
    print("[3/3] TEST 2: Best Single Model (Fold 1) + TTA")
    print("=" * 80)
    
    results_tta = worker.evaluate_with_tta(
        test_dataset=test_dataset,
        model_path=best_model_path
    )
    
    # Test 3: Ensemble of all 5 folds WITHOUT TTA
    print("\n" + "=" * 80)
    print("[4/4] TEST 3: Ensemble (5 folds) - No TTA")
    print("=" * 80)
    
    model_paths = [
        f"outputs/checkpoints/v1.0/best_model_fold{i}.pth" 
        for i in range(1, 6)
    ]
    
    results_ensemble = worker.evaluate_ensemble(
        test_dataset=test_dataset,
        model_paths=model_paths,
        use_tta=False
    )
    
    # Test 4: Ensemble of all 5 folds WITH TTA
    print("\n" + "=" * 80)
    print("[5/5] TEST 4: Ensemble (5 folds) + TTA")
    print("=" * 80)
    
    results_ensemble_tta = worker.evaluate_ensemble(
        test_dataset=test_dataset,
        model_paths=model_paths,
        use_tta=True
    )
    
    # Summary comparison
    print("\n" + "=" * 80)
    print("SUMMARY COMPARISON")
    print("=" * 80)
    
    print(f"\n{'Method':<30} {'Dice':<10} {'IoU':<10} {'Exudates':<12} {'Haemorrhages':<12}")
    print("-" * 80)
    
    print(f"{'Single Model (Baseline)':<30} {results_baseline['mean_dice']:.4f}     {results_baseline['mean_iou']:.4f}     "
          f"{results_baseline['dice_exudates']:.4f}       {results_baseline['dice_haemorrhages']:.4f}")
    
    print(f"{'Single Model + TTA':<30} {results_tta['mean_dice']:.4f}     {results_tta['mean_iou']:.4f}     "
          f"{results_tta['dice_exudates']:.4f}       {results_tta['dice_haemorrhages']:.4f}")
    
    print(f"{'Ensemble (5 folds)':<30} {results_ensemble['mean_dice']:.4f}     {results_ensemble['mean_iou']:.4f}     "
          f"{results_ensemble['dice_exudates']:.4f}       {results_ensemble['dice_haemorrhages']:.4f}")
    
    print(f"{'Ensemble + TTA':<30} {results_ensemble_tta['mean_dice']:.4f}     {results_ensemble_tta['mean_iou']:.4f}     "
          f"{results_ensemble_tta['dice_exudates']:.4f}       {results_ensemble_tta['dice_haemorrhages']:.4f}")
    
    # Calculate improvements
    improvement_tta = (results_tta['mean_dice'] - results_baseline['mean_dice']) * 100
    improvement_ensemble = (results_ensemble['mean_dice'] - results_baseline['mean_dice']) * 100
    improvement_both = (results_ensemble_tta['mean_dice'] - results_baseline['mean_dice']) * 100
    
    print("\n" + "=" * 80)
    print("IMPROVEMENTS vs BASELINE")
    print("=" * 80)
    print(f"TTA only:           +{improvement_tta:.2f}% Dice")
    print(f"Ensemble only:      +{improvement_ensemble:.2f}% Dice")
    print(f"Ensemble + TTA:     +{improvement_both:.2f}% Dice")
    print("=" * 80)


if __name__ == "__main__":
    main()
