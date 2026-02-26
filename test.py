import os
import glob
import shutil
import pandas as pd
from concurrent.futures import ProcessPoolExecutor, as_completed
from tqdm import tqdm
from DriveIdent.lib.core.backend_adapter import train, predict
import itertools
import csv
import argparse

'''
This Script is used to test the accuracy of the predictions across all possible combinations of training and test splits available for the projects recordings.
Note: This script only works in combination with the exact amount and names of the recording files listed in the arrays below. It is not intended to be used with other recordings.
'''

split_dir = "splits"

florian = [
    "recording_2025_12_11_12_29_26_florian.csv",
    "recording_2026_02_10__13_18_02_florian.csv",
    "recording_2026_02_10__13_37_56_florian.csv",
    "recording_2026_02_10__14_03_00_florian_night.csv",
    "recording_2026_02_10__14_22_43_florian_night.csv",
    "recording_2026_02_10__15_13_03_florian.csv",
    "recording_2026_02_10__15_31_22_florian.csv",
    "recording_2026_02_10__15_56_32_florian.csv",
]

matthias = [
    "recording_2025_12_11_12_38_11_matthias.csv",
    "recording_2026_02_10__13_25_22_matthias.csv",
    "recording_2026_02_10__13_44_54_matthias.csv",
    "recording_2026_02_10__14_10_05_matthias_night.csv",
    "recording_2026_02_10__14_29_48_matthias_night.csv",
    "recording_2026_02_10__15_19_00_matthias.csv",
    "recording_2026_02_10__15_38_03_matthias.csv",
    "recording_2026_02_10__16_02_05_matthias.csv",
]

fabian = [
    "recording_2025_12_11_12_46_32_fabian.csv",
    "recording_2026_02_10__13_10_22_fabian.csv",
    "recording_2026_02_10__13_51_14_fabian.csv",
    "recording_2026_02_10__14_16_19_fabian_night.csv",
    "recording_2026_02_10__15_25_06_fabian.csv",
    "recording_2026_02_10__15_43_36_fabian.csv",
    "recording_2026_02_10__15_49_12_fabian_night.csv",
    "recording_2026_02_10__16_08_13_fabian.csv",
]

drivers = {
    "florian": florian,
    "matthias": matthias,
    "fabian": fabian
}

assert all(len(v) == 8 for v in drivers.values())

def split():
    '''
    Combines triplets of recordings (one per driver) to every possible test/split combination where the amount of training triplets is 2 to 7 triplets (246 in total = sum of 8 over n, where n is 2 to 7)
    '''
    print("Erstelle Trainingssplits")
    os.makedirs(split_dir, exist_ok=True)

    split_id = 1
    indices = list(range(8))

    for n in range(2, 8):  # 2 bis 7
        for train_indices in itertools.combinations(indices, n):

            test_indices = [i for i in indices if i not in train_indices]

            train_file = os.path.join(split_dir, f"split_{split_id:03d}_train.lbl")
            test_file = os.path.join(split_dir, f"split_{split_id:03d}_test.lbl")

            # TRAIN FILE
            with open(train_file, "w", newline="") as f:
                writer = csv.writer(f)
                writer.writerow(["File", "Label"])
                for driver, files in drivers.items():
                    for i in train_indices:
                        writer.writerow([files[i], driver])

            # TEST FILE
            with open(test_file, "w", newline="") as f:
                writer = csv.writer(f)
                writer.writerow(["File", "Label"])
                for driver, files in drivers.items():
                    for i in test_indices:
                        writer.writerow([files[i], driver])

            split_id += 1

    print(f"Erstellt: {split_id - 1} Splits")


def run_split(train_path, data_dir, fail_on_err = True):
    ''' Trains models and predicts drivers for a single train/test split '''
    print(data_dir)
    split_number = train_path.split("_")[1]
    test_path = f"{split_dir}/split_{split_number}_test.lbl"

    train_df = pd.read_csv(train_path)
    test_df = pd.read_csv(test_path)

    n_train_triplets = len(train_df) // 3
    n_test_triplets = len(test_df) // 3

    artifacts_dir = f"artifacts_split_{split_number}"
    os.makedirs(artifacts_dir, exist_ok=True)

    results = []
    try:
        out, err = train(
            data_dir=data_dir,
            labels=train_df,
            artifacts_dir=artifacts_dir,
            progress_callback=None,
            use_grid_search=False
        )

        if not out:
            raise Exception(err)

        out, err, res = predict(
            data_dir=data_dir,
            test_labels_file=test_df,
            artifacts_dir=artifacts_dir,
            progress_callback=None
        )

        if not out:
            raise Exception(err) 

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

    except Exception as err:
        print(err)
        if fail_on_err:
            raise Exception(err)
        return []
                
    finally:
        shutil.rmtree(artifacts_dir, ignore_errors=True)

    return results

def main():
    
    parser = argparse.ArgumentParser(description="Führt Vorhersagen über alle möglichen Kombinationen von 2 - 7 Trainingstriplets durch")
    parser.add_argument("--data-dir", type=str, help="Ordner mit CSV-Recordings")
    args = parser.parse_args()
    data_dir = args.data_dir

    split()
    train_files = sorted(glob.glob(f"{split_dir}/split_*_train.lbl"))

    max_workers = 4
    all_results = []

    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        futures = [executor.submit(run_split, tp, data_dir) for tp in train_files]

        with tqdm(total=len(futures), desc="Processing Splits") as pbar:
            for future in as_completed(futures):
                all_results.extend(future.result())
                pbar.update(1)

    results_df = pd.DataFrame(all_results)
    results_df.to_csv("evaluation_results.csv", index=False)

    print("FERTIG.")


if __name__ == "__main__":
    main()