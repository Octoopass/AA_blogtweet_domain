import json
from pathlib import Path

from utils.paths import PROJECT_ROOT


SECTION_KEYS = {
    "data_selection": {
        "num_authors",
        "tweet_word_limit",
        "blog_word_limit",
        "source_tweet_file",
        "source_blog_file",
        "selected_data_dir",
        "selection_seed",
        "base_seed",
        "skip_selection",
    },
    "models": {
        "compare_unified",
        "compare_tweet_only",
        "compare_neural_network",
        "compare_distilbert",
        "unified_model_type",
        "tweet_only_model_type",
        "neural_network_model_type",
    },
    "training": {
        "num_runs",
        "test_size",
        "tweet_group_size",
        "use_blog_title",
        "use_domain_weighting",
        "tweet_weight",
        "blog_weight",
    },
    "features": {
        "use_4grams",
        "top_4grams",
        "use_feature_selection",
        "feature_selection_ratio",
        "tweet_use_feature_selection",
        "use_semantic_clustering",
        "semantic_cluster_k",
        "semantic_model_name",
    },
    "output": {
        "results_dir",
        "averaged_results_file",
        "detailed_results_file",
        "comparison_summary_file",
    },
    "distilbert": {
        "distilbert_model_name",
        "distilbert_max_length",
        "distilbert_batch_size",
        "distilbert_learning_rate",
        "distilbert_num_epochs",
        "distilbert_use_blog_data",
        "distilbert_use_tweet_data",
    },
}

PATH_KEYS = {
    "source_tweet_file",
    "source_blog_file",
    "selected_data_dir",
    "results_dir",
}


def load_experiment_config(default_config, config_path=None):
    """Load a JSON experiment config and merge it into default_config."""
    config = default_config.copy()
    if not config_path:
        return config

    path = Path(config_path)
    if not path.is_absolute():
        path = PROJECT_ROOT / path

    with path.open("r", encoding="utf-8") as file:
        raw_config = json.load(file)

    flat_config = _flatten_config(raw_config)
    for key, value in flat_config.items():
        if value is None:
            continue
        if key in PATH_KEYS:
            value = str(_resolve_project_path(value))
        config[key] = value

    return config


def apply_cli_overrides(config, args, ignored_keys=None):
    """Apply argparse values that were explicitly provided."""
    ignored_keys = set(ignored_keys or [])
    for key, value in vars(args).items():
        if key in ignored_keys or value is None:
            continue
        config[key] = value
    return config


def _flatten_config(raw_config):
    flat = {}
    for key, value in raw_config.items():
        if isinstance(value, dict) and key in SECTION_KEYS:
            for nested_key, nested_value in value.items():
                if nested_key not in SECTION_KEYS[key]:
                    raise ValueError(f"Unknown config key: {key}.{nested_key}")
                flat[nested_key] = nested_value
        else:
            flat[key] = value
    return flat


def _resolve_project_path(value):
    path = Path(value)
    if path.is_absolute():
        return path
    return PROJECT_ROOT / path
