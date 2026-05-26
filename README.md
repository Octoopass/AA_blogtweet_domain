# AA Blog-Tweet Authorship Attribution

Tweet authorship attribution experiments for a cross-domain setting where training data can include both Blog and Tweet text.

## Structure

```text
AA_tweet/
├── config/experiment_config.json
├── data/
│   ├── Tweet30.csv
│   └── Blog30.csv
├── src/
│   ├── evaluate.py
│   ├── mixed_train.py
│   ├── tweet_only.py
│   ├── neural_network.py
│   ├── distilbert.py
│   ├── adversarial_domain/
│   └── utils/
├── training_data/   # generated
└── results/         # generated
```

Expected CSV columns:

- `Author Name`
- `Text`
- `Title` for blog data

## Setup

```bash
python -m venv .venv
pip install pandas numpy scikit-learn matplotlib seaborn nltk emoji
```

```bash
pip install xgboost sentence-transformers
pip install torch transformers accelerate
```

NLTK data:

```bash
python -m nltk.downloader punkt punkt_tab words stopwords averaged_perceptron_tagger averaged_perceptron_tagger_eng
```

In case NLTK version does not support the newer package names:

```bash
python -m nltk.downloader punkt words stopwords averaged_perceptron_tagger
```

## Config

Edit:

```text
config/experiment_config.json
```

Important fields:

- `data_selection.num_authors`
- `data_selection.tweet_word_limit`
- `data_selection.blog_word_limit`
- `data_selection.selection_seed`
- `training.num_runs`
- `training.use_domain_weighting`
- `features.use_semantic_clustering`
- `models.compare_unified`
- `models.compare_tweet_only`
- `models.compare_neural_network`
- `models.compare_distilbert`
- `models.unified_model_type`
- `models.tweet_only_model_type`
- `models.neural_network_model_type`

`selection_seed` defaults to `null` for random author/data selection. `base_seed` is only used for training-run seeds.

Neural Network, DistilBERT, domain weighting, and semantic clustering are disabled by default.

## Run

Full workflow from config:

```bash
python src/evaluate.py --config config/experiment_config.json
```

Workflow with 5 random author selections:

```bash
python src/random_selection_evaluate.py --config config/experiment_config.json --selection-runs 5 --inner-num-runs 1 --output-dir results/random_selection_averaged
```

`random_selection_evaluate.py` runs `evaluate.py` five times. Each outer run uses
a different random `selection_seed`, writes a separate selected dataset, and then
aggregates the five `model_comparison_summary.csv` files.

Only select data:

```bash
python src/utils/select_data.py --config config/experiment_config.json
```

Quick test:

```bash
python src/utils/select_data.py --num-authors 5 --tweet-word-limit 1500 --blog-word-limit 1500
python src/evaluate.py --skip-selection --num-runs 1 --unified-model-type svm --tweet-only-model-type svm
```

Enable optional model branches:

```bash
python src/evaluate.py --config config/experiment_config.json --compare-neural-network --neural-network-model-type mlp
python src/evaluate.py --config config/experiment_config.json --compare-distilbert
```

Optional machine-learning classifier settings:

```bash
python src/evaluate.py --config config/experiment_config.json --use-domain-weighting
python src/evaluate.py --config config/experiment_config.json --use-semantic-clustering
python src/evaluate.py --config config/experiment_config.json --unified-model-type svm --use-domain-weighting --use-semantic-clustering
```

## Individual Scripts

```bash
python src/mixed_train.py
python src/blog_only.py
python src/tweet_only.py
python src/neural_network.py
python src/distilbert.py
python src/adversarial_domain/adversarial_domain_adaptation.py
```

## Outputs

Evaluation writes to:

```text
results/averaged/
```

Individual write to:

```text
results/unified/
results/tweet_only/
results/neural_network/
results/distilbert/
results/adversarial/
```

## Notes

- All data collected are in English. Retweet and quote (retweet but other author) are not collected.
- If `sentence-transformers` is unavailable, semantic clustering falls back to TF-IDF/SVD.
- If `xgboost` is unavailable, use `svm`, `logistic_regression`, `bagging`, or `voting` without XGBoost.
