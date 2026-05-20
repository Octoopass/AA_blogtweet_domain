"""
Author Data Selection Script for Training
Randomly selects X authors and extracts their data from tweet and blog datasets.
"""

import pandas as pd
import random
import argparse
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = PROJECT_ROOT / 'data'


# ============================================================================
# CONFIGURATION 
# ============================================================================

CONFIG = {
    # Number of authors to randomly select
    'num_authors': 30,
    
    # Maximum word count for tweet data per author
    'tweet_word_limit': 3000,
    
    # Maximum word count for blog data per author (None for no limit)
    'blog_word_limit': 3000, 
    
    # Input file paths
    'tweet_file': str(DATA_DIR / 'Tweet30.csv'),
    'blog_file': str(DATA_DIR / 'Blog30.csv'),
    
    # Output directory for results
    'output_dir': str(PROJECT_ROOT / 'training_data'),
    
    # Random seed for reproducibility (None for random selection each time)
    'seed': 42, 
}

# ============================================================================
# END CONFIGURATION SECTION
# ============================================================================


def count_words(text):
    """Count words in a text string."""
    if pd.isna(text):
        return 0
    return len(str(text).split())


def limit_text_by_words(texts, word_limit=5000):
    """
    Limit a list of texts to a maximum word count.
    Returns texts that fit within the word limit.
    Ensures at least one text is included even if it exceeds the limit.
    """
    if not texts:  # If no texts available, return empty
        return [], 0
    
    limited_texts = []
    total_words = 0
    
    for text in texts:
        text_words = count_words(text)
        
        # If this is the first item, always include it (even if it exceeds limit)
        if len(limited_texts) == 0:
            limited_texts.append(text)
            total_words += text_words
            # If first item already exceeds limit, stop here
            if total_words >= word_limit:
                break
        elif total_words + text_words <= word_limit:
            limited_texts.append(text)
            total_words += text_words
        else:
            # Calculate remaining words we can include
            remaining_words = word_limit - total_words
            if remaining_words > 0:
                # Add partial text
                words = str(text).split()
                limited_texts.append(' '.join(words[:remaining_words]))
                total_words += remaining_words
            break
    
    return limited_texts, total_words


def select_random_authors(tweet_file, blog_file, num_authors, tweet_word_limit=5000, 
                          blog_word_limit=None, output_dir='training_data', seed=None):
    """
    Select random authors and extract their data for training.
    
    Parameters:
    -----------
    tweet_file : str
        Path to the tweet CSV file
    blog_file : str
        Path to the blog CSV file
    num_authors : int
        Number of authors to randomly select
    tweet_word_limit : int
        Maximum word count for tweet data per author (default: 5000)
    blog_word_limit : int, optional
        Maximum word count for blog data per author (default: None for no limit)
    output_dir : str
        Directory to save output files (default: 'training_data')
    seed : int, optional
        Random seed for reproducibility
    
    Returns:
    --------
    dict : Dictionary containing selected authors and statistics
    """
    
    # Set random seed if provided
    if seed is not None:
        random.seed(seed)
    
    # Read the datasets
    print(f"Reading {tweet_file}...")
    tweets_df = pd.read_csv(tweet_file, encoding='utf-8-sig')
    
    print(f"Reading {blog_file}...")
    blogs_df = pd.read_csv(blog_file, encoding='utf-8-sig')
    
    # Get unique authors from both datasets
    tweet_authors = set(tweets_df['Author Name'].unique())
    blog_authors = set(blogs_df['Author Name'].unique())
    all_authors = tweet_authors.union(blog_authors)
    
    print(f"\nFound {len(all_authors)} unique authors across both datasets")
    print(f"  - Tweet authors: {len(tweet_authors)}")
    print(f"  - Blog authors: {len(blog_authors)}")
    print(f"  - Authors in both: {len(tweet_authors.intersection(blog_authors))}")
    
    # Check if we have enough authors
    if num_authors > len(all_authors):
        print(f"\nWarning: Requested {num_authors} authors but only {len(all_authors)} available.")
        print(f"Selecting all {len(all_authors)} authors instead.")
        num_authors = len(all_authors)
    
    # Randomly select authors
    selected_authors = random.sample(sorted(all_authors), num_authors)
    print(f"\nRandomly selected {num_authors} authors:")
    for i, author in enumerate(selected_authors, 1):
        print(f"  {i}. {author}")
    
    # Create output directory
    output_path = Path(output_dir)
    output_path.mkdir(exist_ok=True)
    
    # Prepare data for each selected author
    results = {
        'selected_authors': selected_authors,
        'author_stats': []
    }
    
    all_tweets = []
    all_blogs = []
    
    for author in selected_authors:
        author_stats = {'author': author}
        
        # Extract tweets for this author
        author_tweets = tweets_df[tweets_df['Author Name'] == author]
        tweet_texts = author_tweets['Text'].tolist()
        
        # Limit tweet texts by word count
        limited_tweets, total_tweet_words = limit_text_by_words(tweet_texts, tweet_word_limit)
        author_stats['original_tweet_count'] = len(tweet_texts)
        author_stats['included_tweet_count'] = len(limited_tweets)
        author_stats['total_tweet_words'] = total_tweet_words
        
        # Extract blogs for this author
        author_blogs = blogs_df[blogs_df['Author Name'] == author]
        blog_texts = author_blogs['Text'].tolist()
        blog_titles = author_blogs['Title'].tolist()
        
        # Limit blog texts by word count if blog_word_limit is set
        if blog_word_limit is not None:
            limited_blogs, total_blog_words = limit_text_by_words(blog_texts, blog_word_limit)
            author_stats['original_blog_count'] = len(blog_texts)
            author_stats['included_blog_count'] = len(limited_blogs)
            author_stats['total_blog_words'] = total_blog_words
        else:
            limited_blogs = blog_texts
            author_stats['original_blog_count'] = len(blog_texts)
            author_stats['included_blog_count'] = len(blog_texts)
            total_blog_words = sum(count_words(text) for text in blog_texts)
            author_stats['total_blog_words'] = total_blog_words
        
        # Add to combined datasets
        for tweet in limited_tweets:
            all_tweets.append({'Author Name': author, 'Text': tweet})
        
        for i, blog_text in enumerate(limited_blogs):
            # Get the corresponding title if available, otherwise use empty string
            blog_title = blog_titles[i] if i < len(blog_titles) else ''
            all_blogs.append({
                'Author Name': author,
                'Title': blog_title,
                'Text': blog_text
            })
        
        results['author_stats'].append(author_stats)
    
    # Create output DataFrames
    selected_tweets_df = pd.DataFrame(all_tweets)
    selected_blogs_df = pd.DataFrame(all_blogs)
    
    # Save to CSV files
    tweets_output = output_path / 'selected_tweets.csv'
    blogs_output = output_path / 'selected_blogs.csv'
    stats_output = output_path / 'selection_statistics.txt'
    
    selected_tweets_df.to_csv(tweets_output, index=False, encoding='utf-8-sig')
    selected_blogs_df.to_csv(blogs_output, index=False, encoding='utf-8-sig')
    
    print(f"\n{'='*60}")
    print("SELECTION COMPLETE")
    print(f"{'='*60}")
    
    # Print and save statistics
    with open(stats_output, 'w', encoding='utf-8') as f:
        stats_lines = []
        stats_lines.append(f"Random Author Selection Statistics")
        stats_lines.append(f"{'='*60}")
        stats_lines.append(f"Total authors selected: {num_authors}")
        stats_lines.append(f"Tweet word limit per author: {tweet_word_limit}")
        if blog_word_limit is not None:
            stats_lines.append(f"Blog word limit per author: {blog_word_limit}")
        else:
            stats_lines.append(f"Blog word limit per author: None (no limit)")
        if seed is not None:
            stats_lines.append(f"Random seed: {seed}")
        stats_lines.append(f"\nPer-Author Statistics:")
        stats_lines.append(f"{'-'*60}")
        
        for stat in results['author_stats']:
            stats_lines.append(f"\nAuthor: {stat['author']}")
            stats_lines.append(f"  Tweets: {stat['included_tweet_count']}/{stat['original_tweet_count']} " +
                             f"({stat['total_tweet_words']} words)")
            stats_lines.append(f"  Blogs: {stat['included_blog_count']}/{stat['original_blog_count']} " +
                             f"({stat['total_blog_words']} words)")
        
        total_tweet_words = sum(s['total_tweet_words'] for s in results['author_stats'])
        total_blog_words = sum(s['total_blog_words'] for s in results['author_stats'])
        total_tweets = sum(s['included_tweet_count'] for s in results['author_stats'])
        total_blogs = sum(s['included_blog_count'] for s in results['author_stats'])
        
        stats_lines.append(f"\n{'='*60}")
        stats_lines.append(f"TOTAL STATISTICS:")
        stats_lines.append(f"  Total tweets included: {total_tweets}")
        stats_lines.append(f"  Total tweet words: {total_tweet_words}")
        stats_lines.append(f"  Total blogs included: {total_blogs}")
        stats_lines.append(f"  Total blog words: {total_blog_words}")
        stats_lines.append(f"  Combined word count: {total_tweet_words + total_blog_words}")
        
        stats_lines.append(f"\n{'='*60}")
        stats_lines.append(f"OUTPUT FILES:")
        stats_lines.append(f"  Tweets: {tweets_output}")
        stats_lines.append(f"  Blogs: {blogs_output}")
        stats_lines.append(f"  Statistics: {stats_output}")
        
        stats_text = '\n'.join(stats_lines)
        f.write(stats_text)
        print(f"\n{stats_text}")
    
    return results


def main():
    parser = argparse.ArgumentParser(
        description='Randomly select authors and prepare their data for training',
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    
    parser.add_argument('--num-authors', '-n', type=int, default=CONFIG['num_authors'],
                       help=f"Number of authors to randomly select (default: {CONFIG['num_authors']})")
    parser.add_argument('--tweet-file', type=str, default=CONFIG['tweet_file'],
                       help=f"Path to tweet CSV file (default: {CONFIG['tweet_file']})")
    parser.add_argument('--blog-file', type=str, default=CONFIG['blog_file'],
                       help=f"Path to blog CSV file (default: {CONFIG['blog_file']})")
    parser.add_argument('--tweet-word-limit', type=int, default=CONFIG['tweet_word_limit'],
                       help=f"Maximum word count for tweet data per author (default: {CONFIG['tweet_word_limit']})")
    parser.add_argument('--blog-word-limit', type=int, default=CONFIG['blog_word_limit'],
                       help=f"Maximum word count for blog data per author (default: {CONFIG['blog_word_limit']}, 0 = no limit)")
    parser.add_argument('--output-dir', type=str, default=CONFIG['output_dir'],
                       help=f"Directory to save output files (default: {CONFIG['output_dir']})")
    parser.add_argument('--seed', type=int, default=CONFIG['seed'],
                       help=f"Random seed for reproducibility (default: {CONFIG['seed']})")
    
    args = parser.parse_args()
    
    # Handle blog_word_limit: treat 0 or negative values as None (no limit)
    blog_limit = args.blog_word_limit if args.blog_word_limit and args.blog_word_limit > 0 else None
    
    print("\n" + "="*60)
    print("AUTHOR DATA SELECTION SCRIPT")
    print("="*60)
    print(f"Configuration:")
    print(f"  Authors to select: {args.num_authors}")
    print(f"  Tweet word limit: {args.tweet_word_limit}")
    print(f"  Blog word limit: {blog_limit if blog_limit else 'None (no limit)'}")
    print(f"  Random seed: {args.seed if args.seed else 'Random (not set)'}")
    print(f"  Output directory: {args.output_dir}")
    print("="*60 + "\n")
    
    select_random_authors(
        tweet_file=args.tweet_file,
        blog_file=args.blog_file,
        num_authors=args.num_authors,
        tweet_word_limit=args.tweet_word_limit,
        blog_word_limit=blog_limit,
        output_dir=args.output_dir,
        seed=args.seed
    )


if __name__ == '__main__':
    main()
