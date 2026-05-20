import numpy as np
from sklearn.cluster import KMeans
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import TfidfVectorizer

try:
    from sentence_transformers import SentenceTransformer
    SENTENCE_TRANSFORMERS_AVAILABLE = True
except ImportError:
    SentenceTransformer = None
    SENTENCE_TRANSFORMERS_AVAILABLE = False


class SemanticTweetClusterer:
    """Cluster tweet texts with SentenceTransformer embeddings and a TF-IDF fallback."""

    def __init__(
        self,
        model_name="paraphrase-multilingual-MiniLM-L12-v2",
        random_state=42,
        min_group_size=5,
    ):
        self.model_name = model_name
        self.random_state = random_state
        self.min_group_size = min_group_size
        self.semantic_model = None

    def cluster(self, tweets, n_clusters):
        """Return a mapping of cluster id to tweets."""
        if len(tweets) < 2:
            return {0: tweets}

        max_reasonable_clusters = max(1, len(tweets) // self.min_group_size)
        effective_clusters = min(n_clusters, max_reasonable_clusters, len(tweets))
        if effective_clusters < 2:
            return {0: tweets}

        embeddings = self._build_embeddings(tweets)
        if embeddings is None or len(embeddings) != len(tweets):
            return {0: tweets}

        try:
            kmeans = KMeans(
                n_clusters=effective_clusters,
                random_state=self.random_state,
                n_init=10,
            )
            cluster_labels = kmeans.fit_predict(embeddings)
        except Exception as exc:
            print(f"[WARNING] Semantic clustering failed, using sequential grouping: {exc}")
            return {0: tweets}

        cluster_groups = {}
        for idx, label in enumerate(cluster_labels):
            cluster_groups.setdefault(label, []).append(tweets[idx])
        return cluster_groups

    def _build_embeddings(self, tweets):
        if SENTENCE_TRANSFORMERS_AVAILABLE:
            try:
                if self.semantic_model is None:
                    self.semantic_model = SentenceTransformer(self.model_name)
                return self.semantic_model.encode(
                    tweets,
                    show_progress_bar=False,
                    convert_to_numpy=True,
                )
            except Exception as exc:
                print(f"[WARNING] sentence-transformers embedding failed: {exc}")

        try:
            vectorizer = TfidfVectorizer(
                ngram_range=(1, 2),
                max_features=5000,
                min_df=1,
            )
            tfidf = vectorizer.fit_transform(tweets)

            max_components = min(100, tfidf.shape[1] - 1, tfidf.shape[0] - 1)
            if max_components >= 2:
                svd = TruncatedSVD(
                    n_components=max_components,
                    random_state=self.random_state,
                )
                return svd.fit_transform(tfidf)

            return tfidf.toarray()
        except Exception as exc:
            print(f"[WARNING] TF-IDF fallback embedding failed: {exc}")
            return None

