import numpy as np
from sklearn.feature_selection import mutual_info_classif
from sklearn.preprocessing import LabelEncoder


def select_features_mi(features, labels, ratio=0.05, random_state=42, top_n_preview=15):
    """Select top features using Mutual Information."""
    print("\n" + "=" * 70)
    print("MUTUAL INFORMATION FEATURE SELECTION")
    print("=" * 70)

    le = LabelEncoder()
    labels_enc = le.fit_transform(labels)

    n_total = features.shape[1]
    n_select = max(1, int(n_total * ratio))

    print(f"Total features: {n_total}")
    print(f"Selecting: {n_select} ({ratio * 100:.1f}%)")

    features_filled = features.fillna(0)
    mi_scores = mutual_info_classif(
        features_filled.values,
        labels_enc,
        discrete_features=False,
        random_state=random_state,
        n_neighbors=3,
    )

    score_dict = dict(zip(features.columns, mi_scores))
    sorted_feats = sorted(score_dict.items(), key=lambda x: x[1], reverse=True)
    selected = [feature for feature, _ in sorted_feats[:n_select]]

    if top_n_preview:
        print(f"\nTop {min(top_n_preview, len(sorted_feats))} features:")
        for i, (feature, score) in enumerate(sorted_feats[:top_n_preview], 1):
            print(f"  {i:2d}. {feature:45s}: {score:.6f}")

    return selected, score_dict

