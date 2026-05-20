"""
Adversarial Domain Adaptation for Author Identification
Adapts from Blog (source) to Tweet (target) domains using DANN approach
Unfortunately it simply does not work
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
import pickle
import warnings
warnings.filterwarnings('ignore')

from utils.feature_selection import select_features_mi
from utils.paths import ensure_parent_dir, project_path, results_path
from utils.stylometric_features import UnifiedFeatureExtractor

# PyTorch imports
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, TensorDataset
import torch.nn.functional as F

# ============================================================================
# CONFIGURATION
# ============================================================================

class Config:
    """Configuration for adversarial domain adaptation"""
    
    # Input data files
    BLOG_CSV = str(project_path('training_data', 'selected_blogs.csv'))
    TWEET_CSV = str(project_path('training_data', 'selected_tweets.csv'))
    
    # Column names
    TEXT_COLUMN = 'Text'
    AUTHOR_COLUMN = 'Author Name'
    BLOG_TITLE_COLUMN = 'Title'
    
    # Data parameters
    USE_BLOG_TITLE = True
    TWEET_GROUP_SIZE = 10
    TEST_SIZE = 0.2
    RANDOM_STATE = 42
    
    # Feature extraction
    USE_4GRAMS = True
    TOP_4GRAMS = 1000
    USE_FEATURE_SELECTION = False  
    FEATURE_SELECTION_RATIO = 0.05  
    
    # Neural network architecture
    HIDDEN_DIM = 512  # Feature extractor hidden dimension
    FEATURE_DIM = 256  # Domain-invariant feature dimension
    
    # Training parameters
    BATCH_SIZE = 16  
    EPOCHS = 100
    LEARNING_RATE = 0.001  # Slightly lower initial LR
    USE_LR_SCHEDULER = True  # Add learning rate decay
    
    # Adversarial training
    LAMBDA_DOMAIN = 1.0 
    LAMBDA_SCHEDULE = 'progressive'  # 'fixed' or 'progressive'
    DOMAIN_LABEL_SMOOTHING = 0.2  
    USE_ENTROPY_LOSS = False 
    LAMBDA_ENTROPY = 0.1  # Weight for entropy regularization
    
    # Device
    DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    # Output files
    MODEL_FILE = str(results_path('adversarial', 'dann_model.pth'))
    RESULTS_FILE = str(results_path('adversarial', 'dann_results.txt'))
    TRAINING_HISTORY_IMG = str(results_path('adversarial', 'dann_training_history.png'))


# ============================================================================
# GRADIENT REVERSAL LAYER
# ============================================================================

class GradientReversalFunction(torch.autograd.Function):
    """
    Gradient Reversal Layer from DANN paper
    Forward: identity function
    Backward: multiply gradient by -lambda
    """
    @staticmethod
    def forward(ctx, x, lambda_):
        ctx.lambda_ = lambda_
        return x.view_as(x)
    
    @staticmethod
    def backward(ctx, grad_output):
        output = grad_output.neg() * ctx.lambda_
        return output, None


class GradientReversalLayer(nn.Module):
    def __init__(self):
        super(GradientReversalLayer, self).__init__()
    
    def forward(self, x, lambda_=1.0):
        return GradientReversalFunction.apply(x, lambda_)


# ============================================================================
# DOMAIN-ADVERSARIAL NEURAL NETWORK 
# ============================================================================

class DANN(nn.Module):
    """
    Domain-Adversarial Neural Network for author identification
    
    Architecture:
    - Feature Extractor: Maps input features to domain-invariant representation
    - Author Classifier: Predicts author from features (main task)
    - Domain Classifier: Predicts domain (blog vs tweet) - gets reversed gradients
    """
    
    def __init__(self, input_dim, num_authors, hidden_dim=256, feature_dim=128):
        super(DANN, self).__init__()
        
        # Feature Extractor (shared between both tasks)
        self.feature_extractor = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.2),
            
            nn.Linear(hidden_dim, feature_dim),
            nn.BatchNorm1d(feature_dim),
            nn.ReLU(),
            nn.Dropout(0.2)
        )
        
        # Author Classifier (main task)
        self.author_classifier = nn.Sequential(
            nn.Linear(feature_dim, 128),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(128, num_authors)
        )
        
        # Domain Classifier (adversarial task) - Weakened to reduce domain loss
        self.domain_classifier = nn.Sequential(
            nn.Linear(feature_dim, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Dropout(0.3),  # Increased from 0.5 to 0.7 to weaken domain classifier
            nn.Linear(64, 2)  # Binary: 0=blog, 1=tweet
        )
        
        # Gradient reversal layer
        self.grl = GradientReversalLayer()
    
    def forward(self, x, lambda_=1.0, return_features=False):
        # Extract features
        features = self.feature_extractor(x)
        
        # Author prediction (main task)
        author_output = self.author_classifier(features)
        
        # Domain prediction (adversarial task - with gradient reversal)
        reversed_features = self.grl(features, lambda_)
        domain_output = self.domain_classifier(reversed_features)
        
        if return_features:
            return author_output, domain_output, features
        return author_output, domain_output


# ============================================================================
# TRAINING LOOP
# ============================================================================

def train_dann(model, train_loader_source, train_loader_target, 
               num_epochs, optimizer, device, lambda_schedule='progressive'):
    """
    Train DANN model
    
    Args:
        model: DANN model
        train_loader_source: DataLoader for source domain (blogs with labels)
        train_loader_target: DataLoader for target domain (tweets, may be unlabeled)
        num_epochs: Number of training epochs
        optimizer: Optimizer
        device: torch device
        lambda_schedule: 'fixed' or 'progressive' (gradually increase lambda)
    """
    
    model.to(device)
    criterion_author = nn.CrossEntropyLoss()
    criterion_domain = nn.CrossEntropyLoss(label_smoothing=Config.DOMAIN_LABEL_SMOOTHING)
    
    # Add learning rate scheduler
    if Config.USE_LR_SCHEDULER:
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode='min', factor=0.5, patience=5
        )
        print("  Using ReduceLROnPlateau scheduler (factor=0.5, patience=5)")
    
    history = {
        'author_loss': [],
        'domain_loss': [],
        'total_loss': [],
        'domain_acc': [],
        'lambda': [],
        'learning_rate': []
    }
    
    for epoch in range(num_epochs):
        model.train()
        
        # Calculate lambda (adversarial weight) based on schedule
        if lambda_schedule == 'progressive':
            p = epoch / num_epochs
            # More aggressive schedule: reaches max lambda faster
            lambda_ = Config.LAMBDA_DOMAIN * (2. / (1. + np.exp(-10 * p)) - 1)
        else:
            lambda_ = Config.LAMBDA_DOMAIN
        
        epoch_author_loss = 0
        epoch_domain_loss = 0
        epoch_total_loss = 0
        correct_domain = 0
        total_domain = 0
        
        # Iterate through both source and target data
        source_iter = iter(train_loader_source)
        target_iter = iter(train_loader_target)
        
        n_batches = min(len(train_loader_source), len(train_loader_target))
        
        for batch_idx in range(n_batches):
            # Get source batch (blogs with labels)
            try:
                source_data, source_labels = next(source_iter)
            except StopIteration:
                source_iter = iter(train_loader_source)
                source_data, source_labels = next(source_iter)
            
            # Get target batch (tweets)
            try:
                target_data, target_labels = next(target_iter)
            except StopIteration:
                target_iter = iter(train_loader_target)
                target_data, target_labels = next(target_iter)
            
            source_data = source_data.to(device)
            source_labels = source_labels.to(device)
            target_data = target_data.to(device)
            target_labels = target_labels.to(device)
            
            # Create domain labels (0=source/blog, 1=target/tweet)
            domain_labels_source = torch.zeros(source_data.size(0), dtype=torch.long).to(device)
            domain_labels_target = torch.ones(target_data.size(0), dtype=torch.long).to(device)
            
            optimizer.zero_grad()
            
            # Forward pass for source domain
            author_output_source, domain_output_source = model(source_data, lambda_)
            
            # Forward pass for target domain
            _, domain_output_target = model(target_data, lambda_)
            
            # Calculate losses
            # Author loss (only on source domain with labels)
            loss_author = criterion_author(author_output_source, source_labels)
            
            # Domain loss (on both domains)
            loss_domain_source = criterion_domain(domain_output_source, domain_labels_source)
            loss_domain_target = criterion_domain(domain_output_target, domain_labels_target)
            loss_domain = loss_domain_source + loss_domain_target
            
            # Entropy loss (encourage maximum uncertainty in domain predictions)
            if Config.USE_ENTROPY_LOSS:
                # Calculate entropy: -sum(p * log(p))
                # Maximum entropy = log(2) ≈ 0.69 for binary classification
                domain_probs_source = F.softmax(domain_output_source, dim=1)
                domain_probs_target = F.softmax(domain_output_target, dim=1)
                
                entropy_source = -(domain_probs_source * torch.log(domain_probs_source + 1e-8)).sum(dim=1).mean()
                entropy_target = -(domain_probs_target * torch.log(domain_probs_target + 1e-8)).sum(dim=1).mean()
                
                loss_entropy = -(entropy_source + entropy_target)
                
                # Total loss
                total_loss = loss_author + lambda_ * loss_domain + Config.LAMBDA_ENTROPY * loss_entropy
            else:
                # Total loss (original)
                total_loss = loss_author + lambda_ * loss_domain
            
            # Backward and optimize
            total_loss.backward()
            
            # Gradient clipping for stability
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            
            optimizer.step()
            
            # Track metrics
            epoch_author_loss += loss_author.item()
            epoch_domain_loss += loss_domain.item()
            epoch_total_loss += total_loss.item()
            
            # Domain accuracy
            domain_preds_source = torch.argmax(domain_output_source, dim=1)
            domain_preds_target = torch.argmax(domain_output_target, dim=1)
            correct_domain += (domain_preds_source == domain_labels_source).sum().item()
            correct_domain += (domain_preds_target == domain_labels_target).sum().item()
            total_domain += source_data.size(0) + target_data.size(0)
        
        # Average losses
        avg_author_loss = epoch_author_loss / n_batches
        avg_domain_loss = epoch_domain_loss / n_batches
        avg_total_loss = epoch_total_loss / n_batches
        domain_acc = correct_domain / total_domain
        
        history['author_loss'].append(avg_author_loss)
        history['domain_loss'].append(avg_domain_loss)
        history['total_loss'].append(avg_total_loss)
        history['domain_acc'].append(domain_acc)
        history['lambda'].append(lambda_)
        history['learning_rate'].append(optimizer.param_groups[0]['lr'])
        
        # Step learning rate scheduler
        if Config.USE_LR_SCHEDULER:
            old_lr = optimizer.param_groups[0]['lr']
            scheduler.step(avg_author_loss)
            new_lr = optimizer.param_groups[0]['lr']
            if old_lr != new_lr:
                print(f"\n Learning rate reduced: {old_lr:.6f} to {new_lr:.6f}")
        
        if (epoch + 1) % 5 == 0:
            print(f"Epoch [{epoch+1}/{num_epochs}]")
            print(f"  Author Loss: {avg_author_loss:.4f}")
            print(f"  Domain Loss: {avg_domain_loss:.4f}")
            print(f"  Domain Acc: {domain_acc:.4f} (target: ~0.5 for confusion)")
            print(f"  Lambda: {lambda_:.4f}")
    
    return history


def evaluate_dann(model, test_loader, device, label_encoder):
    """Evaluate DANN on test set"""
    model.eval()
    all_preds = []
    all_labels = []
    
    with torch.no_grad():
        for data, labels in test_loader:
            data = data.to(device)
            author_output, _ = model(data, lambda_=0)  # No gradient reversal during eval
            preds = torch.argmax(author_output, dim=1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.numpy())
    
    accuracy = accuracy_score(all_labels, all_preds)
    
    # Decode labels for report
    all_labels_decoded = label_encoder.inverse_transform(all_labels)
    all_preds_decoded = label_encoder.inverse_transform(all_preds)
    
    print("\n" + "="*70)
    print("EVALUATION RESULTS")
    print("="*70)
    print(f"Test Accuracy: {accuracy*100:.2f}%")
    print("\nClassification Report:")
    print(classification_report(all_labels_decoded, all_preds_decoded))
    
    return {
        'accuracy': accuracy,
        'predictions': all_preds_decoded,
        'true_labels': all_labels_decoded
    }


# ============================================================================
# MAIN PIPELINE
# ============================================================================

def main():
    print("="*70)
    print("ADVERSARIAL DOMAIN ADAPTATION FOR AUTHOR IDENTIFICATION")
    print("="*70)
    print(f"Device: {Config.DEVICE}")
    
    # [1] Load data
    print("\n[1/8] Loading data...")
    try:
        blog_df = pd.read_csv(Config.BLOG_CSV)
        print(f"  Loaded {len(blog_df)} blog entries")
    except Exception as e:
        print(f"  [ERROR] Could not load blog data: {e}")
        return
    
    try:
        tweet_df = pd.read_csv(Config.TWEET_CSV)
        print(f"  Loaded {len(tweet_df)} tweets")
    except Exception as e:
        print(f"  [ERROR] Could not load tweet data: {e}")
        return
    
    # [2] Initialize feature extractor
    print("\n[2/8] Initializing feature extractor...")
    extractor = UnifiedFeatureExtractor(
        top_4grams=Config.TOP_4GRAMS,
        use_4grams=Config.USE_4GRAMS,
        text_column=Config.TEXT_COLUMN,
        author_column=Config.AUTHOR_COLUMN,
        random_state=Config.RANDOM_STATE,
        tweet_group_size=Config.TWEET_GROUP_SIZE,
    )
    
    # [3] Preprocess blogs
    print("\n[3/8] Preprocessing blogs...")
    blog_texts = []
    blog_labels = []
    for _, row in blog_df.iterrows():
        title = row.get(Config.BLOG_TITLE_COLUMN, "") if Config.USE_BLOG_TITLE else ""
        text = extractor.preprocess_blog(row[Config.TEXT_COLUMN], title)
        if text.strip():
            blog_texts.append(text)
            blog_labels.append(row[Config.AUTHOR_COLUMN])
    print(f"  Processed {len(blog_texts)} blogs from {len(set(blog_labels))} authors")
    
    # [4] Preprocess tweets (group by author)
    print("\n[4/8] Preprocessing tweets...")
    grouped_tweets = []
    for author in tweet_df[Config.AUTHOR_COLUMN].unique():
        author_tweets = tweet_df[tweet_df[Config.AUTHOR_COLUMN] == author][Config.TEXT_COLUMN].tolist()
        for i in range(0, len(author_tweets), Config.TWEET_GROUP_SIZE):
            group = author_tweets[i:i + Config.TWEET_GROUP_SIZE]
            if len(group) >= Config.TWEET_GROUP_SIZE:
                grouped_tweets.append({'tweets': group, 'author': author})
    
    tweet_texts = []
    tweet_labels = []
    for group in grouped_tweets:
        combined = ' '.join([extractor.preprocess_tweet(t) for t in group['tweets']])
        if combined.strip():
            tweet_texts.append(combined)
            tweet_labels.append(group['author'])
    print(f"  Processed {len(tweet_texts)} tweet groups from {len(set(tweet_labels))} authors")
    
    # [5] Split tweet data
    print("\n[5/8] Splitting tweet data...")
    tweet_train_texts, tweet_test_texts, tweet_train_labels, tweet_test_labels = train_test_split(
        tweet_texts, tweet_labels,
        test_size=Config.TEST_SIZE,
        random_state=Config.RANDOM_STATE,
        stratify=tweet_labels
    )
    print(f"  Tweet train: {len(tweet_train_texts)}")
    print(f"  Tweet test: {len(tweet_test_texts)}")
    
    # [6] Build vocabulary
    print("\n[6/8] Building feature vocabulary...")
    if Config.USE_4GRAMS:
        extractor.build_4gram_vocabulary(blog_texts, tweet_train_texts)
    else:
        extractor.vocab_4grams = []
        extractor.feature_names = extractor._build_fixed_feature_names()
    print(f"  Total features: {len(extractor.feature_names)}")
    
    # [7] Extract features
    print("\n[7/8] Extracting features...")
    blog_features, blog_labels_series = extractor.process_texts(blog_texts, blog_labels, 'blogs')
    tweet_train_features, tweet_train_labels_series = extractor.process_texts(
        tweet_train_texts, tweet_train_labels, 'tweet-train'
    )
    tweet_test_features, tweet_test_labels_series = extractor.process_texts(
        tweet_test_texts, tweet_test_labels, 'tweet-test'
    )
    
    print(f"\n  Blog features: {blog_features.shape}")
    print(f"  Tweet train features: {tweet_train_features.shape}")
    print(f"  Tweet test features: {tweet_test_features.shape}")
    
    # [8] Feature selection using mutual information
    if Config.USE_FEATURE_SELECTION:
        print("\n[8/9] Feature selection using mutual information...")
        print("  Combining blog + tweet train for feature selection...")
        
        # Combine blog and tweet train for feature selection
        combined_train_features = pd.concat([blog_features, tweet_train_features], ignore_index=True)
        combined_train_labels = pd.concat([blog_labels_series, tweet_train_labels_series], ignore_index=True)
        
        # Select features
        selected_features, mi_scores = select_features_mi(
            combined_train_features,
            combined_train_labels,
            ratio=Config.FEATURE_SELECTION_RATIO,
            random_state=Config.RANDOM_STATE
        )
        
        # Apply selection to all datasets
        blog_features = blog_features[selected_features]
        tweet_train_features = tweet_train_features[selected_features]
        tweet_test_features = tweet_test_features[selected_features]
        
        print(f"\n  After feature selection:")
        print(f"    Blog features: {blog_features.shape}")
        print(f"    Tweet train features: {tweet_train_features.shape}")
        print(f"    Tweet test features: {tweet_test_features.shape}")
    else:
        print("\n[8/9] Skipping feature selection (disabled)")
        selected_features = list(blog_features.columns)
    
    # [9] Prepare PyTorch datasets
    print("\n[9/9] Preparing neural network...")
    
    # Encode labels
    label_encoder = LabelEncoder()
    all_labels = pd.concat([blog_labels_series, tweet_train_labels_series, tweet_test_labels_series])
    label_encoder.fit(all_labels)
    
    blog_labels_encoded = label_encoder.transform(blog_labels_series)
    tweet_train_labels_encoded = label_encoder.transform(tweet_train_labels_series)
    tweet_test_labels_encoded = label_encoder.transform(tweet_test_labels_series)
    
    num_authors = len(label_encoder.classes_)
    print(f"  Number of authors: {num_authors}")
    
    # Scale features
    scaler = StandardScaler()
    blog_features_scaled = scaler.fit_transform(blog_features)
    tweet_train_features_scaled = scaler.transform(tweet_train_features)
    tweet_test_features_scaled = scaler.transform(tweet_test_features)
    
    # Create PyTorch datasets
    blog_dataset = TensorDataset(
        torch.FloatTensor(blog_features_scaled),
        torch.LongTensor(blog_labels_encoded)
    )
    tweet_train_dataset = TensorDataset(
        torch.FloatTensor(tweet_train_features_scaled),
        torch.LongTensor(tweet_train_labels_encoded)
    )
    tweet_test_dataset = TensorDataset(
        torch.FloatTensor(tweet_test_features_scaled),
        torch.LongTensor(tweet_test_labels_encoded)
    )
    
    # Create data loaders
    blog_loader = DataLoader(blog_dataset, batch_size=Config.BATCH_SIZE, shuffle=True)
    tweet_train_loader = DataLoader(tweet_train_dataset, batch_size=Config.BATCH_SIZE, shuffle=True)
    tweet_test_loader = DataLoader(tweet_test_dataset, batch_size=Config.BATCH_SIZE, shuffle=False)
    
    # Initialize model
    input_dim = blog_features.shape[1]
    model = DANN(
        input_dim=input_dim,
        num_authors=num_authors,
        hidden_dim=Config.HIDDEN_DIM,
        feature_dim=Config.FEATURE_DIM
    )
    
    print(f"  Model architecture:")
    print(f"    Input dim: {input_dim}")
    print(f"    Hidden dim: {Config.HIDDEN_DIM}")
    print(f"    Feature dim: {Config.FEATURE_DIM}")
    print(f"    Output (authors): {num_authors}")
    
    # Optimizer
    optimizer = optim.Adam(model.parameters(), lr=Config.LEARNING_RATE)
    
    # Train model
    print("\n" + "="*70)
    print("TRAINING DANN MODEL")
    print("="*70)
    print(f"Source domain: Blogs (labeled)")
    print(f"Target domain: Tweets (labeled for evaluation)")
    print(f"Epochs: {Config.EPOCHS}")
    print(f"Lambda schedule: {Config.LAMBDA_SCHEDULE}")
    
    history = train_dann(
        model=model,
        train_loader_source=blog_loader,
        train_loader_target=tweet_train_loader,
        num_epochs=Config.EPOCHS,
        optimizer=optimizer,
        device=Config.DEVICE,
        lambda_schedule=Config.LAMBDA_SCHEDULE
    )
    
    # Evaluate on test set
    results = evaluate_dann(model, tweet_test_loader, Config.DEVICE, label_encoder)
    
    # Plot training history
    plot_training_history(history)
    
    # Save model
    ensure_parent_dir(Config.MODEL_FILE)
    torch.save({
        'model_state_dict': model.state_dict(),
        'label_encoder': label_encoder,
        'scaler': scaler,
        'feature_names': extractor.feature_names,
        'config': {
            'input_dim': input_dim,
            'num_authors': num_authors,
            'hidden_dim': Config.HIDDEN_DIM,
            'feature_dim': Config.FEATURE_DIM
        }
    }, Config.MODEL_FILE)
    
    print(f"\n[SUCCESS] Model saved to {Config.MODEL_FILE}")
    
    return model, results, history


def plot_training_history(history):
    """Plot training metrics"""
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    
    # Author loss
    axes[0, 0].plot(history['author_loss'])
    axes[0, 0].set_title('Author Classification Loss')
    axes[0, 0].set_xlabel('Epoch')
    axes[0, 0].set_ylabel('Loss')
    axes[0, 0].grid(True)
    
    # Domain loss
    axes[0, 1].plot(history['domain_loss'])
    axes[0, 1].set_title('Domain Classification Loss')
    axes[0, 1].set_xlabel('Epoch')
    axes[0, 1].set_ylabel('Loss')
    axes[0, 1].grid(True)
    
    # Domain accuracy (should approach 0.5 for good adaptation)
    axes[1, 0].plot(history['domain_acc'])
    axes[1, 0].axhline(y=0.5, color='r', linestyle='--', label='Target (0.5)')
    axes[1, 0].set_title('Domain Classifier Accuracy')
    axes[1, 0].set_xlabel('Epoch')
    axes[1, 0].set_ylabel('Accuracy')
    axes[1, 0].legend()
    axes[1, 0].grid(True)
    
    # Lambda schedule
    axes[1, 1].plot(history['lambda'])
    axes[1, 1].set_title('Adversarial Weight (Lambda)')
    axes[1, 1].set_xlabel('Epoch')
    axes[1, 1].set_ylabel('Lambda')
    axes[1, 1].grid(True)
    
    plt.tight_layout()
    ensure_parent_dir(Config.TRAINING_HISTORY_IMG)
    plt.savefig(Config.TRAINING_HISTORY_IMG, dpi=300, bbox_inches='tight')
    print(f"\n[PLOT] Training history saved to {Config.TRAINING_HISTORY_IMG}")


if __name__ == "__main__":
    model, results, history = main()

