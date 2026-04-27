from pathlib import Path
import pandas as pd


REPO_ROOT = Path(__file__).resolve().parents[1]

LABELS_PATH = REPO_ROOT / "data" / "labels.csv"
OUT_PATH = REPO_ROOT / "data" / "ranking_pairs.csv"


def main():
    labels = pd.read_csv(LABELS_PATH)

    rows = []

    for _, row in labels.iterrows():
        query = str(row["query_card_id"])
        good = str(row["good_card_id"])
        bad = str(row["bad_card_id"])

        candidates = str(row["candidate_card_ids"]).split("|")

        mids = [
            c for c in candidates
            if c != good and c != bad
        ]

        # good > mids
        for mid in mids:
            rows.append({
                "query_card_id": query,
                "preferred_card_id": good,
                "worse_card_id": mid,
                "comparison_type": "good>mid",
            })

        # mids > bad
        for mid in mids:
            rows.append({
                "query_card_id": query,
                "preferred_card_id": mid,
                "worse_card_id": bad,
                "comparison_type": "mid>bad",
            })

        # good > bad
        rows.append({
            "query_card_id": query,
            "preferred_card_id": good,
            "worse_card_id": bad,
            "comparison_type": "good>bad",
        })

    out = pd.DataFrame(rows)
    out.to_csv(OUT_PATH, index=False)

    print(f"Input label rows: {len(labels)}")
    print(f"Output ranking pairs: {len(out)}")
    print(f"Saved to: {OUT_PATH}")
    print()
    print(out["comparison_type"].value_counts())


if __name__ == "__main__":
    main()