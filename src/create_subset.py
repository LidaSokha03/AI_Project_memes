from pathlib import Path

import pandas as pd


PROJECT_ROOT = Path.cwd()
METADATA_PATH = PROJECT_ROOT / "data" / "interim" / "metadata.csv"
OUTPUT_PATH = PROJECT_ROOT / "data" / "interim" / "metadata_4000.csv"
IDS_PATH = (PROJECT_ROOT / "data" / "splits" / "working_subset_4000_ids.csv")

RANDOM_SEED = 42
SUBSET_SIZE = 4000


def main():
    metadata = pd.read_csv(METADATA_PATH)
    valid = metadata[metadata["is_valid"] == True].copy()
    subset = valid.sample(n=SUBSET_SIZE,random_state=RANDOM_SEED).copy()
    subset = subset.reset_index(drop=True)
    subset.to_csv(OUTPUT_PATH,index=False)

    IDS_PATH.parent.mkdir(parents=True,exist_ok=True)
    subset[["meme_id"]].to_csv(IDS_PATH,index=False)

    print(f"Saved {len(subset)} images to:")
    print(OUTPUT_PATH)

    print("\nSaved shared subset IDs to:")
    print(IDS_PATH)

    print("\nDataset distribution:")
    print(subset["dataset"].value_counts())


if __name__ == "__main__":
    main()
