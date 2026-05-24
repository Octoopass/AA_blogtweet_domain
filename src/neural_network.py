"""
Approach: Neural Network setup on combined blog+tweet feature vectors.
Feature vectors are reused directly from mixed_train.py.
"""

import copy
import pickle
import warnings
from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    precision_recall_fscore_support,
)
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler

warnings.filterwarnings("ignore")

try:
    import torch
    import torch.nn as nn
    from torch.utils.data import DataLoader, TensorDataset, WeightedRandomSampler
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False

import mixed_train
from utils.feature_selection import select_features_mi
from utils.paths import ensure_parent_dir, project_path, results_path


class Config:
    # Data files
    BLOG_CSV = str(project_path("training_data", "selected_blogs.csv"))
    TWEET_CSV = str(project_path("training_data", "selected_tweets.csv"))

    # Shared feature-vector settings
    USE_BLOG_TITLE = True
    TWEET_GROUP_SIZE = 10
    USE_SEMANTIC_CLUSTERING = False
    SEMANTIC_CLUSTER_K = 10
    SEMANTIC_MODEL_NAME = "paraphrase-multilingual-MiniLM-L12-v2"
    TEST_SIZE = 0.2
    RANDOM_STATE = 42
    USE_4GRAMS = True
    TOP_4GRAMS = 1000
    USE_FEATURE_SELECTION = True
    FEATURE_SELECTION_RATIO = 0.05

    # Domain weighting
    USE_DOMAIN_WEIGHTING = False
    TWEET_WEIGHT = 3.0
    BLOG_WEIGHT = 1.0

    # NN settings
    MODEL_TYPE = "all"  # "mlp", "bilstm", "all"
    EPOCHS = 50
    BATCH_SIZE = 32
    LEARNING_RATE = 1e-3
    EARLY_STOPPING_PATIENCE = 8

    # MLP
    MLP_HIDDEN_DIMS = [512, 256, 128]
    MLP_DROPOUT = 0.3

    # BiLSTM
    BILSTM_HIDDEN_SIZE = 256
    BILSTM_NUM_LAYERS = 2
    BILSTM_DROPOUT = 0.3

    # Outputs
    RESULTS_FILE = str(results_path("neural_network", "evaluation_results_nn.txt"))
    BEST_MODEL_FILE = str(results_path("neural_network", "nn_best_model.pkl"))


def sync_mixed_train_config():
    """Mirror local config into mixed_train.Config."""
    base = mixed_train.Config
    base.BLOG_CSV = Config.BLOG_CSV
    base.TWEET_CSV = Config.TWEET_CSV
    base.USE_BLOG_TITLE = Config.USE_BLOG_TITLE
    base.TWEET_GROUP_SIZE = Config.TWEET_GROUP_SIZE
    base.USE_SEMANTIC_CLUSTERING = Config.USE_SEMANTIC_CLUSTERING
    base.SEMANTIC_CLUSTER_K = Config.SEMANTIC_CLUSTER_K
    base.SEMANTIC_MODEL_NAME = Config.SEMANTIC_MODEL_NAME
    base.TEST_SIZE = Config.TEST_SIZE
    base.RANDOM_STATE = Config.RANDOM_STATE
    base.USE_4GRAMS = Config.USE_4GRAMS
    base.TOP_4GRAMS = Config.TOP_4GRAMS
    base.USE_FEATURE_SELECTION = Config.USE_FEATURE_SELECTION
    base.FEATURE_SELECTION_RATIO = Config.FEATURE_SELECTION_RATIO
    base.USE_DOMAIN_WEIGHTING = Config.USE_DOMAIN_WEIGHTING
    base.TWEET_WEIGHT = Config.TWEET_WEIGHT
    base.BLOG_WEIGHT = Config.BLOG_WEIGHT


def prepare_feature_data():
    """Build train/test feature matrices with the same pipeline."""
    sync_mixed_train_config()
    extractor = mixed_train.make_feature_extractor()

    # Blog data
    blog_df = pd.read_csv(Config.BLOG_CSV)
    blog_texts = []
    blog_labels = []
    for _, row in blog_df.iterrows():
        text = row[mixed_train.Config.TEXT_COLUMN]
        title = row.get(mixed_train.Config.BLOG_TITLE_COLUMN, "") if Config.USE_BLOG_TITLE else ""
        processed = extractor.preprocess_blog(text, title)
        if processed:
            blog_texts.append(processed)
            blog_labels.append(row[mixed_train.Config.AUTHOR_COLUMN])

    # Tweet data with grouping/clustering
    tweet_df = pd.read_csv(Config.TWEET_CSV)
    grouped_tweets = extractor.group_tweets(tweet_df, Config.TWEET_GROUP_SIZE)
    tweet_texts = []
    tweet_labels = []
    for group in grouped_tweets:
        combined = " ".join([extractor.preprocess_tweet(t) for t in group["tweets"]])
        if combined.strip():
            tweet_texts.append(combined)
            tweet_labels.append(group["author"])

    # Train/test split on tweets
    tweet_texts_train, tweet_texts_test, tweet_labels_train, tweet_labels_test = train_test_split(
        tweet_texts,
        tweet_labels,
        test_size=Config.TEST_SIZE,
        random_state=Config.RANDOM_STATE,
        stratify=tweet_labels,
    )

    # Keep 4-gram vocab exactly aligned with mixed_train
    if Config.USE_4GRAMS:
        extractor.build_4gram_vocabulary(blog_texts, tweet_texts_train)
    else:
        extractor.vocab_4grams = []
        extractor.feature_names = extractor._build_fixed_feature_names()

    # Feature extraction
    blog_features, blog_labels_series = extractor.process_texts(blog_texts, blog_labels, "blog")
    tweet_train_features, tweet_train_labels_series = extractor.process_texts(
        tweet_texts_train, tweet_labels_train, "tweet-train"
    )
    tweet_test_features, tweet_test_labels_series = extractor.process_texts(
        tweet_texts_test, tweet_labels_test, "tweet-test"
    )

    # Feature selection
    if Config.USE_FEATURE_SELECTION:
        combined_train_features = pd.concat([blog_features, tweet_train_features], ignore_index=True)
        combined_train_labels = pd.concat([blog_labels_series, tweet_train_labels_series], ignore_index=True)
        selected_features, _ = select_features_mi(
            combined_train_features,
            combined_train_labels,
            ratio=Config.FEATURE_SELECTION_RATIO,
            random_state=Config.RANDOM_STATE,
        )
        blog_features = blog_features[selected_features]
        tweet_train_features = tweet_train_features[selected_features]
        tweet_test_features = tweet_test_features[selected_features]
    else:
        selected_features = list(blog_features.columns)

    # Final train/test
    X_train = pd.concat([blog_features, tweet_train_features], axis=0, ignore_index=True).fillna(0)
    y_train = pd.concat([blog_labels_series, tweet_train_labels_series], axis=0, ignore_index=True)
    X_test = tweet_test_features.fillna(0)
    y_test = tweet_test_labels_series.reset_index(drop=True)

    domain_labels_train = pd.Series(["blog"] * len(blog_features) + ["tweet"] * len(tweet_train_features))

    return (
        X_train,
        y_train,
        X_test,
        y_test,
        domain_labels_train,
        extractor,
        selected_features,
    )


class MLPNet(nn.Module):
    def __init__(self, input_dim, num_classes):
        super().__init__()
        layers = []
        prev = input_dim
        for hidden in Config.MLP_HIDDEN_DIMS:
            layers.extend([nn.Linear(prev, hidden), nn.ReLU(), nn.Dropout(Config.MLP_DROPOUT)])
            prev = hidden
        layers.append(nn.Linear(prev, num_classes))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)


class BiLSTMNet(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        dropout = Config.BILSTM_DROPOUT if Config.BILSTM_NUM_LAYERS > 1 else 0.0
        self.lstm = nn.LSTM(
            input_size=1,
            hidden_size=Config.BILSTM_HIDDEN_SIZE,
            num_layers=Config.BILSTM_NUM_LAYERS,
            batch_first=True,
            bidirectional=True,
            dropout=dropout,
        )
        self.dropout = nn.Dropout(Config.BILSTM_DROPOUT)
        self.classifier = nn.Linear(Config.BILSTM_HIDDEN_SIZE * 2, num_classes)

    def forward(self, x):
        # x: (batch, n_features) -> (batch, n_features, 1)
        x = x.unsqueeze(-1)
        out, _ = self.lstm(x)
        last_out = out[:, -1, :]
        return self.classifier(self.dropout(last_out))


@dataclass
class TrainedNNModel:
    model_type: str
    model: object
    label_encoder: LabelEncoder
    scaler: StandardScaler
    selected_features: list

    def predict(self, X_df):
        X_scaled = self.scaler.transform(X_df.fillna(0))
        X_tensor = torch.tensor(X_scaled, dtype=torch.float32)
        self.model.eval()
        with torch.no_grad():
            logits = self.model(X_tensor.to(next(self.model.parameters()).device))
            preds = torch.argmax(logits, dim=1).cpu().numpy()
        return self.label_encoder.inverse_transform(preds)


def build_domain_sample_weights(domain_labels):
    weights_map = {"tweet": Config.TWEET_WEIGHT, "blog": Config.BLOG_WEIGHT}
    weights = np.array([weights_map.get(str(d).lower(), 1.0) for d in domain_labels], dtype=np.float32)
    mean_weight = np.mean(weights)
    if mean_weight > 0:
        weights = weights / mean_weight
    return weights


def evaluate_encoded_predictions(y_true_labels, y_pred_labels, classes):
    accuracy = accuracy_score(y_true_labels, y_pred_labels)
    precision_macro, recall_macro, f1_macro, _ = precision_recall_fscore_support(
        y_true_labels, y_pred_labels, average="macro", zero_division=0
    )
    precision_weighted, recall_weighted, f1_weighted, _ = precision_recall_fscore_support(
        y_true_labels, y_pred_labels, average="weighted", zero_division=0
    )
    report = classification_report(y_true_labels, y_pred_labels, zero_division=0)
    cm = confusion_matrix(y_true_labels, y_pred_labels, labels=classes)
    return {
        "accuracy": accuracy,
        "recall_macro": recall_macro,
        "f1_score_macro": f1_macro,
        "precision_macro": precision_macro,
        "recall_weighted": recall_weighted,
        "f1_score_weighted": f1_weighted,
        "precision_weighted": precision_weighted,
        "classification_report": report,
        "confusion_matrix": cm,
    }


def train_torch_model(model_type, X_train, y_train, X_test, y_test, domain_labels_train):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    label_encoder = LabelEncoder()
    y_train_encoded = label_encoder.fit_transform(y_train)
    y_test_encoded = label_encoder.transform(y_test)
    num_classes = len(label_encoder.classes_)

    X_train_tensor = torch.tensor(X_train_scaled, dtype=torch.float32)
    y_train_tensor = torch.tensor(y_train_encoded, dtype=torch.long)
    X_test_tensor = torch.tensor(X_test_scaled, dtype=torch.float32)
    y_test_tensor = torch.tensor(y_test_encoded, dtype=torch.long)

    train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
    test_dataset = TensorDataset(X_test_tensor, y_test_tensor)

    if Config.USE_DOMAIN_WEIGHTING and domain_labels_train is not None:
        weights = build_domain_sample_weights(domain_labels_train)
        sampler = WeightedRandomSampler(
            weights=torch.DoubleTensor(weights),
            num_samples=len(weights),
            replacement=True,
        )
        train_loader = DataLoader(train_dataset, batch_size=Config.BATCH_SIZE, sampler=sampler)
    else:
        train_loader = DataLoader(train_dataset, batch_size=Config.BATCH_SIZE, shuffle=True)

    test_loader = DataLoader(test_dataset, batch_size=Config.BATCH_SIZE, shuffle=False)

    if model_type == "mlp":
        model = MLPNet(input_dim=X_train.shape[1], num_classes=num_classes).to(device)
    elif model_type == "bilstm":
        model = BiLSTMNet(num_classes=num_classes).to(device)
    else:
        raise ValueError(f"Unsupported model type: {model_type}")

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=Config.LEARNING_RATE)

    best_state = None
    best_f1 = -1.0
    epochs_without_improvement = 0

    for epoch in range(Config.EPOCHS):
        model.train()
        for xb, yb in train_loader:
            xb = xb.to(device)
            yb = yb.to(device)
            optimizer.zero_grad()
            logits = model(xb)
            loss = criterion(logits, yb)
            loss.backward()
            optimizer.step()

        # Validation on tweet test split 
        model.eval()
        all_preds = []
        all_targets = []
        with torch.no_grad():
            for xb, yb in test_loader:
                xb = xb.to(device)
                logits = model(xb)
                preds = torch.argmax(logits, dim=1).cpu().numpy()
                all_preds.extend(preds.tolist())
                all_targets.extend(yb.numpy().tolist())

        _, _, val_f1, _ = precision_recall_fscore_support(
            all_targets, all_preds, average="macro", zero_division=0
        )

        if val_f1 > best_f1:
            best_f1 = val_f1
            best_state = copy.deepcopy(model.state_dict())
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1
            if epochs_without_improvement >= Config.EARLY_STOPPING_PATIENCE:
                break

    if best_state is not None:
        model.load_state_dict(best_state)

    # Final evaluation
    model.eval()
    with torch.no_grad():
        logits = model(X_test_tensor.to(device))
        y_pred_encoded = torch.argmax(logits, dim=1).cpu().numpy()

    y_pred_labels = label_encoder.inverse_transform(y_pred_encoded)
    results = evaluate_encoded_predictions(y_test, y_pred_labels, label_encoder.classes_)
    results["n_authors"] = len(label_encoder.classes_)

    trained = TrainedNNModel(
        model_type=model_type,
        model=model,
        label_encoder=label_encoder,
        scaler=scaler,
        selected_features=list(X_train.columns),
    )
    return trained, results


def save_results(all_results, best_model_name):
    ensure_parent_dir(Config.RESULTS_FILE)
    with open(Config.RESULTS_FILE, "w", encoding="utf-8") as f:
        f.write("=" * 70 + "\n")
        f.write("APPROACH 2 - NEURAL NETWORK RESULTS\n")
        f.write("=" * 70 + "\n\n")
        f.write(f"Best Model: {best_model_name.upper()}\n\n")

        for model_name, result in all_results.items():
            if result is None:
                continue
            f.write("-" * 70 + "\n")
            f.write(f"{model_name.upper()}\n")
            f.write("-" * 70 + "\n")
            f.write(f"Accuracy: {result['accuracy']*100:.2f}%\n")
            f.write(f"Macro Precision: {result['precision_macro']*100:.2f}%\n")
            f.write(f"Macro Recall: {result['recall_macro']*100:.2f}%\n")
            f.write(f"Macro F1: {result['f1_score_macro']*100:.2f}%\n\n")
            f.write(result["classification_report"] + "\n")


def main():
    if not TORCH_AVAILABLE:
        raise ImportError("PyTorch is required for nn_approach2.py. Please install torch first.")

    print("=" * 70)
    print("APPROACH 2 - NEURAL NETWORK TRAINING")
    print("Feature vectors reused from mixed_train.py")
    print("=" * 70)

    (
        X_train,
        y_train,
        X_test,
        y_test,
        domain_labels_train,
        extractor,
        selected_features,
    ) = prepare_feature_data()

    print(f"\nTrain instances: {len(X_train)}")
    print(f"Test instances: {len(X_test)}")
    print(f"Feature count: {X_train.shape[1]}")
    print(f"Authors: {y_train.nunique()}")

    model_types = ["mlp", "bilstm"] if Config.MODEL_TYPE == "all" else [Config.MODEL_TYPE]
    all_results = {}
    all_models = {}

    for model_type in model_types:
        print(f"\nTraining NN model: {model_type.upper()}")
        model, results = train_torch_model(
            model_type=model_type,
            X_train=X_train,
            y_train=y_train,
            X_test=X_test,
            y_test=y_test,
            domain_labels_train=domain_labels_train,
        )
        model.selected_features = selected_features
        all_models[model_type] = model
        all_results[model_type] = results
        print(f"  Accuracy: {results['accuracy']*100:.2f}%")
        print(f"  Macro F1: {results['f1_score_macro']*100:.2f}%")

    best_model_name = max(
        all_results.items(),
        key=lambda x: x[1]["accuracy"] if x[1] else 0,
    )[0]
    best_model = all_models[best_model_name]
    best_results = all_results[best_model_name]

    # Save summary + best model artifact
    save_results(all_results, best_model_name)
    model_state_cpu = {k: v.detach().cpu() for k, v in best_model.model.state_dict().items()}
    ensure_parent_dir(Config.BEST_MODEL_FILE)
    with open(Config.BEST_MODEL_FILE, "wb") as f:
        pickle.dump(
            {
                "best_model_name": best_model_name,
                "all_results": all_results,
                "label_encoder": best_model.label_encoder,
                "scaler": best_model.scaler,
                "selected_features": best_model.selected_features,
                "model_state_dict": model_state_cpu,
                "model_type": best_model.model_type,
            },
            f,
        )

    print("\n" + "=" * 70)
    print(f"Best model: {best_model_name.upper()}")
    print(f"Best accuracy: {best_results['accuracy']*100:.2f}%")
    print("=" * 70)

    return best_model, best_results, all_results


if __name__ == "__main__":
    main()
