import re
import string
from collections import Counter

import numpy as np
import pandas as pd
from nltk import pos_tag
from nltk.corpus import stopwords, words
from nltk.tokenize import word_tokenize

from utils.semantic_clustering import SemanticTweetClusterer


class UnifiedFeatureExtractor:
    """Unified stylometric feature extractor for blog and tweet texts."""

    CHARS = list(string.ascii_lowercase)
    SPECIAL_CHARS = [",", ".", "!", "?", ";", ":", "-", "(", ")", '"', "'", "#", "%", "&", "*"]
    PUNCTUATIONS = [",", ".", ";", ":", "!", "?", "-", "(", ")", "[", "]", '"', "'"]
    COMMON_POS = [
        "NN", "NNS", "NNP", "NNPS", "VB", "VBD", "VBG", "VBN", "VBP", "VBZ",
        "JJ", "JJR", "JJS", "RB", "RBR", "RBS", "PRP", "PRP$", "DT", "IN", "CC", "TO",
    ]
    FUNCTION_WORDS = [
        "the", "a", "an", "and", "or", "but", "if", "because",
        "in", "on", "at", "to", "for", "of", "with", "by", "from",
        "is", "are", "was", "were", "be", "been", "being",
        "have", "has", "had", "do", "does", "did",
        "will", "would", "could", "should", "may", "might", "can",
        "this", "that", "these", "those", "it", "they", "them",
    ]

    def __init__(
        self,
        top_4grams=500,
        use_4grams=True,
        text_column="Text",
        author_column="Author Name",
        use_semantic_clustering=False,
        semantic_cluster_k=10,
        semantic_model_name="paraphrase-multilingual-MiniLM-L12-v2",
        random_state=42,
        tweet_group_size=10,
    ):
        self.top_4grams = top_4grams
        self.use_4grams = use_4grams
        self.text_column = text_column
        self.author_column = author_column
        self.use_semantic_clustering = use_semantic_clustering
        self.semantic_cluster_k = semantic_cluster_k
        self.semantic_model_name = semantic_model_name
        self.random_state = random_state
        self.tweet_group_size = tweet_group_size
        self.english_words = set(words.words())
        self.stop_words = set(stopwords.words("english"))
        self.vocab_4grams = None
        self.feature_names = None
        self.semantic_clusterer = None

    def _build_fixed_feature_names(self):
        names = []
        names += [f"char_{c}_freq" for c in self.CHARS]
        names += [f"special_{c}_freq" for c in self.SPECIAL_CHARS]
        names += ["chars_per_word"]
        names += [f"word_len_{i}_freq" for i in range(1, 11)]
        names += ["word_len_10plus_freq"]
        names += ["digit_freq", "two_digit_freq", "three_digit_freq"]
        names += ["function_words_freq"]
        names += [f"has_punct_{p}" for p in self.PUNCTUATIONS]
        names += [f"pos_{p}_freq" for p in self.COMMON_POS]
        names += ["misspelling_freq"]
        if self.use_4grams and self.vocab_4grams:
            names += [f"4gram_{ng}" for ng in self.vocab_4grams]
        return names

    def preprocess_tweet(self, text):
        if pd.isna(text):
            return ""
        text = re.sub(r"http[s]?://\S+", "", text)
        text = re.sub(r"@\w+", "", text)
        text = re.sub(r"#(\w+)", r"\1", text)
        return text.strip()

    def preprocess_blog(self, text, title=""):
        if pd.isna(text) or str(text).strip() == "":
            return ""
        if title and pd.notna(title) and str(title).strip():
            return f"{title} {text}".strip()
        return text.strip()

    def extract_fixed_features(self, text):
        if pd.isna(text) or not str(text).strip():
            return None

        features = {}
        total_chars = len(text)
        char_counts = Counter(text.lower())

        for c in self.CHARS:
            features[f"char_{c}_freq"] = char_counts.get(c, 0) / total_chars if total_chars else 0

        all_char_counts = Counter(text)
        for c in self.SPECIAL_CHARS:
            features[f"special_{c}_freq"] = all_char_counts.get(c, 0) / total_chars if total_chars else 0

        try:
            words_list = word_tokenize(text)
        except Exception:
            words_list = text.split()
        words_alpha = [w for w in words_list if w.isalpha()]
        n_words = len(words_alpha)

        if n_words > 0:
            word_lengths = [len(w) for w in words_alpha]
            features["chars_per_word"] = np.mean(word_lengths)
            for i in range(1, 11):
                features[f"word_len_{i}_freq"] = sum(1 for length in word_lengths if length == i) / n_words
            features["word_len_10plus_freq"] = sum(1 for length in word_lengths if length >= 10) / n_words
        else:
            features["chars_per_word"] = 0
            for i in range(1, 11):
                features[f"word_len_{i}_freq"] = 0
            features["word_len_10plus_freq"] = 0

        all_words = text.split()
        n_all_words = len(all_words)
        digit_count = sum(1 for c in text if c.isdigit())
        features["digit_freq"] = digit_count / total_chars if total_chars else 0
        features["two_digit_freq"] = len(re.findall(r"\b\d{2}\b", text)) / n_all_words if n_all_words else 0
        features["three_digit_freq"] = len(re.findall(r"\b\d{3}\b", text)) / n_all_words if n_all_words else 0

        words_lower = [w.lower() for w in words_alpha]
        func_count = sum(1 for w in words_lower if w in self.FUNCTION_WORDS)
        features["function_words_freq"] = func_count / n_words if n_words else 0

        for p in self.PUNCTUATIONS:
            features[f"has_punct_{p}"] = 1 if p in text else 0

        if n_words > 0:
            try:
                pos_tags = pos_tag(words_alpha)
                tags = [tag for _, tag in pos_tags]
                pos_counts = Counter(tags)
                n_tags = len(tags)
                for pos in self.COMMON_POS:
                    features[f"pos_{pos}_freq"] = pos_counts.get(pos, 0) / n_tags
            except Exception:
                for pos in self.COMMON_POS:
                    features[f"pos_{pos}_freq"] = 0
        else:
            for pos in self.COMMON_POS:
                features[f"pos_{pos}_freq"] = 0

        long_words = [w for w in words_lower if len(w) > 2]
        if long_words:
            misspelled = sum(1 for w in long_words if w not in self.english_words)
            features["misspelling_freq"] = misspelled / len(long_words)
        else:
            features["misspelling_freq"] = 0

        if self.use_4grams and self.vocab_4grams:
            text_no_space = text.replace(" ", "")
            ngram_counts = Counter([text_no_space[i:i + 4] for i in range(len(text_no_space) - 3)])
            total_ngrams = sum(ngram_counts.values())
            for ng in self.vocab_4grams:
                features[f"4gram_{ng}"] = ngram_counts.get(ng, 0) / total_ngrams if total_ngrams else 0

        return features

    def build_4gram_vocabulary(self, blog_texts, tweet_texts):
        print(f"Building 4-gram vocabulary (top {self.top_4grams})...")
        all_ngrams = Counter()

        for text in blog_texts + tweet_texts:
            if pd.isna(text) or not str(text).strip():
                continue
            text_no_space = str(text).replace(" ", "")
            ngrams = [text_no_space[i:i + 4] for i in range(len(text_no_space) - 3)]
            all_ngrams.update(ngrams)

        most_common = all_ngrams.most_common(self.top_4grams)
        self.vocab_4grams = [ng for ng, _ in most_common]
        self.feature_names = self._build_fixed_feature_names()
        print(f"  Built vocabulary with {len(self.vocab_4grams)} 4-grams")

    def group_tweets(self, df, group_size=None):
        group_size = group_size or self.tweet_group_size
        grouped = []
        for author in df[self.author_column].unique():
            author_df = df[df[self.author_column] == author]
            raw_tweets = author_df[self.text_column].tolist()
            tweets = [self.preprocess_tweet(t) for t in raw_tweets]
            tweets = [t for t in tweets if t.strip()]

            if not tweets:
                continue

            if self.use_semantic_clustering and len(tweets) >= max(4, group_size):
                cluster_groups = self._semantic_cluster_tweets(tweets, group_size)
                for cluster_id in sorted(cluster_groups.keys()):
                    self._append_tweet_groups(grouped, author, cluster_groups[cluster_id], group_size, cluster_id)
            else:
                self._append_tweet_groups(grouped, author, tweets, group_size, None)
        return grouped

    def _append_tweet_groups(self, grouped, author, tweets, group_size, cluster_id):
        for i in range(0, len(tweets), group_size):
            tweet_group = tweets[i:i + group_size]
            if len(tweet_group) >= group_size * 0.5:
                grouped.append({
                    "author": author,
                    "tweets": tweet_group,
                    "cluster_id": int(cluster_id) if cluster_id is not None else None,
                })

    def _semantic_cluster_tweets(self, tweets, group_size):
        min_group_size = max(2, int(np.ceil(group_size * 0.5)))
        if self.semantic_clusterer is None:
            self.semantic_clusterer = SemanticTweetClusterer(
                model_name=self.semantic_model_name,
                random_state=self.random_state,
                min_group_size=min_group_size,
            )
        return self.semantic_clusterer.cluster(tweets, self.semantic_cluster_k)

    def process_texts(self, texts, labels, text_type="text"):
        print(f"Extracting features from {len(texts)} {text_type} texts...")
        features_list = []
        valid_labels = []

        for idx, (text, label) in enumerate(zip(texts, labels)):
            feats = self.extract_fixed_features(text)
            if feats:
                features_list.append(feats)
                valid_labels.append(label)
            if (idx + 1) % 50 == 0:
                print(f"  Processed {idx + 1}/{len(texts)}...")

        features_df = pd.DataFrame(features_list)
        if self.feature_names:
            for col in self.feature_names:
                if col not in features_df.columns:
                    features_df[col] = 0
            features_df = features_df[self.feature_names]

        print(f"Extracted {len(features_df)} instances, {len(features_df.columns)} features")
        return features_df, pd.Series(valid_labels)

