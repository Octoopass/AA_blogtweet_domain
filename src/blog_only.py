import pickle
import warnings

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.ensemble import BaggingClassifier, VotingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    precision_recall_fscore_support,
)
from sklearn.multiclass import OneVsRestClassifier
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.svm import SVC

try:
    from xgboost import XGBClassifier
    XGBOOST_AVAILABLE = True
except ImportError:
    XGBClassifier = None
    XGBOOST_AVAILABLE = False

from utils.feature_selection import select_features_mi
from utils.paths import ensure_parent_dir, project_path, results_path
from utils.stylometric_features import UnifiedFeatureExtractor

warnings.filterwarnings("ignore")


class Config:
    """Configuration for blog-only to tweet authorship attribution."""

    BLOG_CSV = str(project_path("training_data", "selected_blogs.csv"))
    TWEET_CSV = str(project_path("training_data", "selected_tweets.csv"))

    TEXT_COLUMN = "Text"
    AUTHOR_COLUMN = "Author Name"
    BLOG_TITLE_COLUMN = "Title"

    USE_BLOG_TITLE = True
    TWEET_GROUP_SIZE = 10
    RANDOM_STATE = 42

    USE_4GRAMS = True
    TOP_4GRAMS = 1000
    USE_FEATURE_SELECTION = True
    FEATURE_SELECTION_RATIO = 0.05

    MODEL_TYPE = "all"  # svm, logistic_regression, xgboost, bagging, voting, all

    SVM_C = 1.0
    SVM_KERNEL = "linear"

    LR_C = 1.0
    LR_MAX_ITER = 1000

    XGB_MAX_DEPTH = 6
    XGB_LEARNING_RATE = 0.1
    XGB_N_ESTIMATORS = 100

    BAGGING_N_ESTIMATORS = 10
    BAGGING_MAX_SAMPLES = 1.0
    BAGGING_BASE_ESTIMATOR = "decision_tree"

    VOTING_METHOD = "soft"
    VOTING_ESTIMATORS = ["logistic_regression", "svm", "xgboost"]

    MODEL_FILE = str(results_path("blog_only", "blog_only_model.pkl"))
    SCALER_FILE = str(results_path("blog_only", "feature_scaler.pkl"))
    LABEL_ENCODER_FILE = str(results_path("blog_only", "label_encoder.pkl"))
    FEATURE_NAMES_FILE = str(results_path("blog_only", "feature_names.pkl"))
    RESULTS_FILE = str(results_path("blog_only", "evaluation_results_blog_only.txt"))
    CONFUSION_MATRIX_IMG = str(results_path("blog_only", "confusion_matrix_blog_only.png"))


def make_feature_extractor():
    return UnifiedFeatureExtractor(
        top_4grams=Config.TOP_4GRAMS,
        use_4grams=Config.USE_4GRAMS,
        text_column=Config.TEXT_COLUMN,
        author_column=Config.AUTHOR_COLUMN,
        use_semantic_clustering=False,
        random_state=Config.RANDOM_STATE,
        tweet_group_size=Config.TWEET_GROUP_SIZE,
    )


class BlogOnlyAuthorshipModel:
    """Train on blog features only and evaluate on grouped tweet features."""

    def __init__(self, model_type=None, random_state=None):
        self.model_type = model_type or Config.MODEL_TYPE
        self.random_state = random_state or Config.RANDOM_STATE
        valid_models = ["svm", "logistic_regression", "xgboost", "bagging", "voting", "all"]
        if self.model_type not in valid_models:
            raise ValueError(f"Invalid model_type: {self.model_type}. Must be one of {valid_models}")

        self.label_encoder = LabelEncoder()
        self.scaler = StandardScaler()
        self.classifier = None
        self.feature_extractor = None
        self.selected_features = None
        self.training_history = {
            "model_type": self.model_type,
            "n_authors": None,
            "n_features": None,
            "n_blog_instances": None,
            "n_tweet_test_instances": None,
            "train_accuracy": None,
        }

    def _create_classifier(self):
        if self.model_type == "svm":
            return SVC(
                kernel=Config.SVM_KERNEL,
                C=Config.SVM_C,
                random_state=self.random_state,
                probability=False,
            )

        if self.model_type == "logistic_regression":
            lr = LogisticRegression(
                C=Config.LR_C,
                max_iter=Config.LR_MAX_ITER,
                random_state=self.random_state,
                solver="lbfgs",
            )
            return OneVsRestClassifier(lr)

        if self.model_type == "xgboost":
            if not XGBOOST_AVAILABLE:
                raise ImportError("XGBoost is not available")
            return XGBClassifier(
                max_depth=Config.XGB_MAX_DEPTH,
                learning_rate=Config.XGB_LEARNING_RATE,
                n_estimators=Config.XGB_N_ESTIMATORS,
                random_state=self.random_state,
                eval_metric="mlogloss",
            )

        if self.model_type == "bagging":
            base_estimator = self._create_bagging_base_estimator()
            try:
                return BaggingClassifier(
                    estimator=base_estimator,
                    n_estimators=Config.BAGGING_N_ESTIMATORS,
                    max_samples=Config.BAGGING_MAX_SAMPLES,
                    random_state=self.random_state,
                    n_jobs=-1,
                )
            except TypeError:
                return BaggingClassifier(
                    base_estimator=base_estimator,
                    n_estimators=Config.BAGGING_N_ESTIMATORS,
                    max_samples=Config.BAGGING_MAX_SAMPLES,
                    random_state=self.random_state,
                    n_jobs=-1,
                )

        if self.model_type == "voting":
            estimators = []
            for name in Config.VOTING_ESTIMATORS:
                estimator = self._create_voting_estimator(name)
                if estimator is not None:
                    estimators.append(estimator)
            if not estimators:
                raise ValueError("No valid estimators for voting classifier")
            return VotingClassifier(
                estimators=estimators,
                voting=Config.VOTING_METHOD,
                n_jobs=-1,
            )

        raise ValueError(f"Unsupported model_type: {self.model_type}")

    def _create_bagging_base_estimator(self):
        if Config.BAGGING_BASE_ESTIMATOR == "logistic_regression":
            lr = LogisticRegression(
                C=Config.LR_C,
                max_iter=Config.LR_MAX_ITER,
                random_state=self.random_state,
                solver="lbfgs",
            )
            return OneVsRestClassifier(lr)

        if Config.BAGGING_BASE_ESTIMATOR == "svm":
            return SVC(
                kernel=Config.SVM_KERNEL,
                C=Config.SVM_C,
                random_state=self.random_state,
                probability=True,
            )

        if Config.BAGGING_BASE_ESTIMATOR == "decision_tree":
            from sklearn.tree import DecisionTreeClassifier

            return DecisionTreeClassifier(random_state=self.random_state, max_depth=10)

        raise ValueError(f"Invalid BAGGING_BASE_ESTIMATOR: {Config.BAGGING_BASE_ESTIMATOR}")

    def _create_voting_estimator(self, name):
        if name == "logistic_regression":
            lr = LogisticRegression(
                C=Config.LR_C,
                max_iter=Config.LR_MAX_ITER,
                random_state=self.random_state,
                solver="lbfgs",
            )
            return ("lr", OneVsRestClassifier(lr))

        if name == "svm":
            return (
                "svm",
                SVC(
                    kernel=Config.SVM_KERNEL,
                    C=Config.SVM_C,
                    random_state=self.random_state,
                    probability=Config.VOTING_METHOD == "soft",
                ),
            )

        if name == "xgboost":
            if not XGBOOST_AVAILABLE:
                print("[WARNING] XGBoost is not available, skipping it in voting ensemble")
                return None
            return (
                "xgb",
                XGBClassifier(
                    max_depth=Config.XGB_MAX_DEPTH,
                    learning_rate=Config.XGB_LEARNING_RATE,
                    n_estimators=Config.XGB_N_ESTIMATORS,
                    random_state=self.random_state,
                    eval_metric="mlogloss",
                ),
            )

        if name == "decision_tree":
            from sklearn.tree import DecisionTreeClassifier

            return ("dt", DecisionTreeClassifier(random_state=self.random_state, max_depth=10))

        raise ValueError(f"Invalid voting estimator: {name}")

    def train(self, X_train, y_train):
        print("\n" + "=" * 70)
        print("BLOG-ONLY AUTHORSHIP ATTRIBUTION MODEL TRAINING")
        print(f"Model: {self.model_type.upper().replace('_', ' ')}")
        print("Training on: Blog data only")
        print("=" * 70)

        X_train = X_train.fillna(0)
        y_encoded = self.label_encoder.fit_transform(y_train)

        self.classifier = self._create_classifier()
        X_scaled = self.scaler.fit_transform(X_train)
        self.classifier.fit(X_scaled, y_encoded)

        train_pred = self.classifier.predict(X_scaled)
        train_accuracy = accuracy_score(y_encoded, train_pred)
        self.training_history.update(
            {
                "n_authors": len(self.label_encoder.classes_),
                "n_features": X_train.shape[1],
                "n_blog_instances": X_train.shape[0],
                "train_accuracy": train_accuracy,
            }
        )

        print("\nTraining complete")
        print(f"Training accuracy: {train_accuracy * 100:.2f}%")
        return self

    def predict(self, X_test):
        if self.classifier is None:
            raise ValueError("Model is not trained")
        X_test = X_test.fillna(0)
        X_scaled = self.scaler.transform(X_test)
        encoded_pred = self.classifier.predict(X_scaled)
        return self.label_encoder.inverse_transform(encoded_pred)

    def evaluate(self, X_test, y_test, save_results=True):
        print("\n" + "=" * 70)
        print("BLOG-ONLY MODEL EVALUATION ON TWEET TEST DATA")
        print("=" * 70)

        y_pred = self.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        precision_macro, recall_macro, f1_macro, _ = precision_recall_fscore_support(
            y_test, y_pred, average="macro", zero_division=0
        )
        precision_weighted, recall_weighted, f1_weighted, _ = precision_recall_fscore_support(
            y_test, y_pred, average="weighted", zero_division=0
        )
        report = classification_report(y_test, y_pred, zero_division=0)
        cm = confusion_matrix(y_test, y_pred, labels=self.label_encoder.classes_)

        n_authors = self.training_history["n_authors"]
        results = {
            "accuracy": accuracy,
            "precision_macro": precision_macro,
            "recall_macro": recall_macro,
            "f1_score_macro": f1_macro,
            "precision_weighted": precision_weighted,
            "recall_weighted": recall_weighted,
            "f1_score_weighted": f1_weighted,
            "n_authors": n_authors,
            "random_baseline": 100 / n_authors,
            "improvement": accuracy * 100 - (100 / n_authors),
            "classification_report": report,
            "confusion_matrix": cm,
        }

        print(f"Test Accuracy: {accuracy * 100:.2f}%")
        print(f"Macro Precision: {precision_macro * 100:.2f}%")
        print(f"Macro Recall: {recall_macro * 100:.2f}%")
        print(f"Macro F1-Score: {f1_macro * 100:.2f}%")
        print("\nClassification report:")
        print(report)

        if save_results:
            self._save_results(results)
            if len(self.label_encoder.classes_) <= 20:
                self._plot_confusion_matrix(cm)

        return results

    def _save_results(self, results):
        ensure_parent_dir(Config.RESULTS_FILE)
        with open(Config.RESULTS_FILE, "w", encoding="utf-8") as file:
            file.write("=" * 70 + "\n")
            file.write("BLOG-ONLY AUTHORSHIP ATTRIBUTION RESULTS\n")
            file.write("=" * 70 + "\n\n")
            file.write(f"Model Type: {self.model_type}\n")
            file.write("Training Setup:\n")
            file.write(f"  - Blog data: ALL instances ({self.training_history['n_blog_instances']})\n")
            file.write("  - Tweet data: ALL grouped tweet instances used as test set\n")
            file.write(f"  - Tweet test instances: {self.training_history['n_tweet_test_instances']}\n\n")
            file.write("Performance Metrics (Tweet Test Set)\n")
            file.write("-" * 70 + "\n")
            file.write(f"Accuracy: {results['accuracy'] * 100:.2f}%\n")
            file.write(f"Random baseline: {results['random_baseline']:.2f}%\n")
            file.write(f"Improvement over random: {results['improvement']:.2f}%\n\n")
            file.write("Macro-averaged Metrics:\n")
            file.write(f"  Precision: {results['precision_macro'] * 100:.2f}%\n")
            file.write(f"  Recall: {results['recall_macro'] * 100:.2f}%\n")
            file.write(f"  F1-Score: {results['f1_score_macro'] * 100:.2f}%\n\n")
            file.write("Weighted-averaged Metrics:\n")
            file.write(f"  Precision: {results['precision_weighted'] * 100:.2f}%\n")
            file.write(f"  Recall: {results['recall_weighted'] * 100:.2f}%\n")
            file.write(f"  F1-Score: {results['f1_score_weighted'] * 100:.2f}%\n\n")
            file.write("Classification Report:\n")
            file.write(results["classification_report"])
        print(f"[SUCCESS] Results saved to: {Config.RESULTS_FILE}")

    def _plot_confusion_matrix(self, cm):
        plt.figure(figsize=(12, 10))
        sns.heatmap(
            cm,
            annot=True,
            fmt="d",
            cmap="Blues",
            xticklabels=self.label_encoder.classes_,
            yticklabels=self.label_encoder.classes_,
        )
        plt.title("Confusion Matrix - Blog-Only to Tweet Authorship Attribution")
        plt.ylabel("True Author")
        plt.xlabel("Predicted Author")
        plt.tight_layout()
        ensure_parent_dir(Config.CONFUSION_MATRIX_IMG)
        plt.savefig(Config.CONFUSION_MATRIX_IMG, dpi=300, bbox_inches="tight")
        plt.close()
        print(f"[SUCCESS] Confusion matrix saved to: {Config.CONFUSION_MATRIX_IMG}")

    def save_model(self, filepath=None):
        filepath = filepath or Config.MODEL_FILE
        ensure_parent_dir(filepath)
        with open(filepath, "wb") as file:
            pickle.dump(
                {
                    "label_encoder": self.label_encoder,
                    "scaler": self.scaler,
                    "classifier": self.classifier,
                    "feature_extractor": self.feature_extractor,
                    "selected_features": self.selected_features,
                    "training_history": self.training_history,
                    "model_type": self.model_type,
                },
                file,
            )
        print(f"[SUCCESS] Model saved to: {filepath}")


def load_blog_texts(extractor):
    print("\n[1/5] Loading blog data...")
    blog_df = pd.read_csv(Config.BLOG_CSV)
    blog_texts = []
    blog_labels = []
    for _, row in blog_df.iterrows():
        title = row.get(Config.BLOG_TITLE_COLUMN, "") if Config.USE_BLOG_TITLE else ""
        text = extractor.preprocess_blog(row[Config.TEXT_COLUMN], title)
        if text:
            blog_texts.append(text)
            blog_labels.append(row[Config.AUTHOR_COLUMN])
    print(f"  Blog instances: {len(blog_texts)}")
    return blog_texts, blog_labels


def load_tweet_test_texts(extractor):
    print("\n[2/5] Loading tweet data for test...")
    tweet_df = pd.read_csv(Config.TWEET_CSV)
    grouped_tweets = extractor.group_tweets(tweet_df, Config.TWEET_GROUP_SIZE)
    tweet_texts = []
    tweet_labels = []
    for group in grouped_tweets:
        combined = " ".join(extractor.preprocess_tweet(tweet) for tweet in group["tweets"])
        if combined.strip():
            tweet_texts.append(combined)
            tweet_labels.append(group["author"])
    print(f"  Tweet test groups: {len(tweet_texts)}")
    return tweet_texts, tweet_labels


def filter_to_common_authors(blog_texts, blog_labels, tweet_texts, tweet_labels):
    blog_authors = set(blog_labels)
    tweet_authors = set(tweet_labels)
    common_authors = blog_authors & tweet_authors
    if not common_authors:
        raise ValueError("No overlapping authors between blog train data and tweet test data")

    filtered_blog = [
        (text, label) for text, label in zip(blog_texts, blog_labels)
        if label in common_authors
    ]
    filtered_tweet = [
        (text, label) for text, label in zip(tweet_texts, tweet_labels)
        if label in common_authors
    ]
    print("\nCommon author filtering:")
    print(f"  Common authors: {len(common_authors)}")
    print(f"  Blog train instances: {len(filtered_blog)}")
    print(f"  Tweet test instances: {len(filtered_tweet)}")

    if not filtered_blog or not filtered_tweet:
        raise ValueError("No usable instances after filtering to common authors")

    return (
        [item[0] for item in filtered_blog],
        [item[1] for item in filtered_blog],
        [item[0] for item in filtered_tweet],
        [item[1] for item in filtered_tweet],
    )


def prepare_features(extractor, blog_texts, blog_labels, tweet_texts, tweet_labels):
    if Config.USE_4GRAMS:
        print("\n[3/5] Building 4-gram vocabulary from blog train data only...")
        extractor.build_4gram_vocabulary(blog_texts, [])
    else:
        print("\n[3/5] Skipping 4-gram vocabulary")
        extractor.vocab_4grams = []
        extractor.feature_names = extractor._build_fixed_feature_names()

    print("\n[4/5] Extracting features...")
    blog_features, blog_labels_series = extractor.process_texts(blog_texts, blog_labels, "blog-train")
    tweet_features, tweet_labels_series = extractor.process_texts(tweet_texts, tweet_labels, "tweet-test")

    if Config.USE_FEATURE_SELECTION:
        print("\n[5/5] Feature selection on blog train data only...")
        selected_features, _ = select_features_mi(
            blog_features,
            blog_labels_series,
            ratio=Config.FEATURE_SELECTION_RATIO,
            random_state=Config.RANDOM_STATE,
        )
        blog_features = blog_features[selected_features]
        tweet_features = tweet_features[selected_features]
    else:
        print("\n[5/5] Skipping feature selection")
        selected_features = list(blog_features.columns)

    print(f"  Selected features: {len(selected_features)}")
    return blog_features, blog_labels_series, tweet_features, tweet_labels_series, selected_features


def train_and_compare_all_models(X_train, y_train, X_test, y_test, extractor, selected_features):
    models_to_test = ["svm", "logistic_regression", "bagging", "voting"]
    if XGBOOST_AVAILABLE:
        models_to_test.insert(2, "xgboost")

    all_results = {}
    all_models = {}
    for index, model_type in enumerate(models_to_test, 1):
        print("\n" + "=" * 70)
        print(f"[{index}/{len(models_to_test)}] Training {model_type.upper().replace('_', ' ')}")
        print("=" * 70)
        try:
            model = BlogOnlyAuthorshipModel(model_type=model_type)
            model.feature_extractor = extractor
            model.selected_features = selected_features
            model.train(X_train, y_train)
            model.training_history["n_tweet_test_instances"] = X_test.shape[0]
            results = model.evaluate(X_test, y_test, save_results=False)
            all_results[model_type] = results
            all_models[model_type] = model
        except Exception as exc:
            print(f"[ERROR] {model_type} failed: {exc}")
            all_results[model_type] = None

    comparison_df = build_comparison_df(all_results)
    print("\n" + "=" * 70)
    print("MODEL COMPARISON RESULTS")
    print("=" * 70)
    print(comparison_df.to_string(index=False))

    best_model_name = max(
        (name for name, result in all_results.items() if result is not None),
        key=lambda name: all_results[name]["accuracy"],
    )
    best_model = all_models[best_model_name]
    _save_all_models_results(all_results, comparison_df, best_model_name)
    best_model.save_model(Config.MODEL_FILE)
    _plot_model_comparison(comparison_df)
    return all_results, comparison_df, best_model_name, best_model


def build_comparison_df(all_results):
    rows = []
    for model_type, results in all_results.items():
        if results is None:
            continue
        rows.append(
            {
                "Model": model_type.replace("_", " ").title(),
                "Accuracy (%)": f"{results['accuracy'] * 100:.2f}",
                "Precision (Macro) (%)": f"{results['precision_macro'] * 100:.2f}",
                "Recall (Macro) (%)": f"{results['recall_macro'] * 100:.2f}",
                "F1-Score (Macro) (%)": f"{results['f1_score_macro'] * 100:.2f}",
                "Precision (Weighted) (%)": f"{results['precision_weighted'] * 100:.2f}",
                "Recall (Weighted) (%)": f"{results['recall_weighted'] * 100:.2f}",
                "F1-Score (Weighted) (%)": f"{results['f1_score_weighted'] * 100:.2f}",
            }
        )
    return pd.DataFrame(rows)


def _save_all_models_results(all_results, comparison_df, best_model_name):
    ensure_parent_dir(Config.RESULTS_FILE)
    with open(Config.RESULTS_FILE, "w", encoding="utf-8") as file:
        file.write("=" * 70 + "\n")
        file.write("ALL MODELS COMPARISON - BLOG-ONLY AUTHORSHIP ATTRIBUTION\n")
        file.write("=" * 70 + "\n\n")
        file.write("Training Setup:\n")
        file.write("  - Train: blog data only\n")
        file.write("  - Test: grouped tweet data only\n")
        file.write("  - Semantic clustering: disabled\n")
        file.write("  - Domain weighting: disabled\n\n")
        file.write("Model Comparison Table\n")
        file.write("-" * 70 + "\n")
        file.write(comparison_df.to_string(index=False))
        file.write("\n\n")
        file.write(f"Best Model: {best_model_name.upper().replace('_', ' ')}\n\n")

        for model_name, results in all_results.items():
            if results is None:
                continue
            file.write("-" * 70 + "\n")
            file.write(f"{model_name.upper().replace('_', ' ')}\n")
            file.write("-" * 70 + "\n")
            file.write(f"Accuracy: {results['accuracy'] * 100:.2f}%\n")
            file.write(f"Macro Precision: {results['precision_macro'] * 100:.2f}%\n")
            file.write(f"Macro Recall: {results['recall_macro'] * 100:.2f}%\n")
            file.write(f"Macro F1-Score: {results['f1_score_macro'] * 100:.2f}%\n\n")
            file.write("Classification Report:\n")
            file.write(results["classification_report"])
            file.write("\n\n")
    print(f"[SUCCESS] All model results saved to: {Config.RESULTS_FILE}")


def _plot_model_comparison(comparison_df):
    if comparison_df.empty:
        return
    metrics = ["Accuracy (%)", "Recall (Macro) (%)", "F1-Score (Macro) (%)"]
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    for index, metric in enumerate(metrics):
        ax = axes[index]
        values = [float(value) for value in comparison_df[metric]]
        models = comparison_df["Model"]
        bars = ax.bar(range(len(models)), values, color="steelblue", alpha=0.8)
        ax.set_xticks(range(len(models)))
        ax.set_xticklabels(models, rotation=45, ha="right")
        ax.set_ylabel("Percentage (%)")
        ax.set_title(metric)
        ax.set_ylim([0, 100])
        ax.grid(axis="y", alpha=0.3)
        best_index = values.index(max(values))
        bars[best_index].set_color("darkgreen")
    plt.tight_layout()
    comparison_img = Config.CONFUSION_MATRIX_IMG.replace(".png", "_comparison.png")
    ensure_parent_dir(comparison_img)
    plt.savefig(comparison_img, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"[SUCCESS] Model comparison plot saved to: {comparison_img}")


def main():
    print("=" * 70)
    print("BLOG-ONLY TO TWEET AUTHORSHIP ATTRIBUTION")
    print("Training: Blog data only")
    print("Testing: Grouped tweet data")
    print("Semantic clustering: disabled")
    print("Domain weighting: disabled")
    print("=" * 70)

    extractor = make_feature_extractor()
    blog_texts, blog_labels = load_blog_texts(extractor)
    tweet_texts, tweet_labels = load_tweet_test_texts(extractor)
    blog_texts, blog_labels, tweet_texts, tweet_labels = filter_to_common_authors(
        blog_texts,
        blog_labels,
        tweet_texts,
        tweet_labels,
    )
    X_train, y_train, X_test, y_test, selected_features = prepare_features(
        extractor,
        blog_texts,
        blog_labels,
        tweet_texts,
        tweet_labels,
    )

    print("\nDataset summary:")
    print(f"  Blog train instances: {X_train.shape[0]}")
    print(f"  Tweet test instances: {X_test.shape[0]}")
    print(f"  Features: {X_train.shape[1]}")
    print(f"  Authors: {y_train.nunique()}")

    if Config.MODEL_TYPE == "all":
        all_results, comparison_df, best_model_name, best_model = train_and_compare_all_models(
            X_train,
            y_train,
            X_test,
            y_test,
            extractor,
            selected_features,
        )
        return best_model, all_results[best_model_name], all_results, comparison_df

    model = BlogOnlyAuthorshipModel(model_type=Config.MODEL_TYPE)
    model.feature_extractor = extractor
    model.selected_features = selected_features
    model.train(X_train, y_train)
    model.training_history["n_tweet_test_instances"] = X_test.shape[0]
    results = model.evaluate(X_test, y_test, save_results=True)
    model.save_model()
    return model, results


if __name__ == "__main__":
    result = main()
