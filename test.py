import glob
import os
import shutil
import pandas as pd
from concurrent.futures import ProcessPoolExecutor, as_completed
from tqdm import tqdm
from DriveIdent.lib.core.backend_adapter import train, predict

split_dir = "D:/Windows/Desktop/MSuT/Hausarbeit - Fahrerprofile/splits"
data_dir = "D:/Windows/Desktop/MSuT/Hausarbeit - Fahrerprofile/recordings"

train_files = sorted(glob.glob(f"{split_dir}/split_*_train.lbl"))


# ========================================
# Worker
# ========================================
def run_split(train_path):

    split_number = train_path.split("_")[1]
    test_path = f"{split_dir}/split_{split_number}_test.lbl"

    train_df = pd.read_csv(train_path)
    test_df = pd.read_csv(test_path)

    n_train_triplets = len(train_df) // 3
    n_test_triplets = len(test_df) // 3

    artifacts_dir = f"artifacts_split_{split_number}"
    os.makedirs(artifacts_dir, exist_ok=True)

    results = []

    out, err = train(
        data_dir=data_dir,
        labels=train_df,
        artifacts_dir=artifacts_dir,
        progress_callback=None,
        use_grid_search=False
    )

    if not out:
        return []

    out, err, res = predict(
        data_dir=data_dir,
        test_labels_file=test_df,
        artifacts_dir=artifacts_dir,
        progress_callback=None
    )

    if not out:
        return []

    for model_name, predictions in res.items():
        pred_df = pd.DataFrame(predictions)
        n_correct = pred_df["korrekt"].sum()
        accuracy = n_correct / len(pred_df)

        results.append({
            "split": int(split_number),
            "model": model_name,
            "train_triplets": n_train_triplets,
            "test_triplets": n_test_triplets,
            "n_correct": int(n_correct),
            "n_test_samples": len(pred_df),
            "accuracy": accuracy
        })

    shutil.rmtree(artifacts_dir, ignore_errors=True)

    return results


# ========================================
# PARALLEL EXECUTION + PROGRESS
# ========================================

def main():

    max_workers = 4
    all_results = []

    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        futures = [executor.submit(run_split, tp) for tp in train_files]

        with tqdm(total=len(futures), desc="Processing Splits") as pbar:
            for future in as_completed(futures):
                all_results.extend(future.result())
                pbar.update(1)

    results_df = pd.DataFrame(all_results)
    results_df.to_csv("evaluation_results.csv", index=False)

    print("FERTIG.")


if __name__ == "__main__":
    main()