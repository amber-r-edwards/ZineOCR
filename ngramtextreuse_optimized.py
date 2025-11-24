"""
N-grams and Jaccard Similarity Text Reuse Analysis - OPTIMIZED
===============================================================

Optimizations:
- Parallel processing with multiprocessing
- Smart pre-filtering (length-based, publication-based, date-based)
- Checkpointing for resumability
- Memory-efficient processing
- Progress tracking with ETA
- Command-line arguments for flexibility

"""

import pandas as pd # type: ignore
import numpy as np
from datetime import datetime
import re
from pathlib import Path
import json
import os
import argparse
import pickle
import time
from typing import Set, Dict, List, Tuple, Optional
from multiprocessing import Pool, cpu_count
from functools import partial
import sys

# ============================================================================
# CONFIGURATION CLASS
# ============================================================================

class Config:
    """Configuration for text reuse analysis"""
    def __init__(self, args):
        self.input_dir = args.input_dir
        self.output_dir = args.output_dir
        self.metadata_file = args.metadata
        
        # Analysis parameters
        self.ngram_size = args.ngram_size
        self.shingle_size = args.shingle_size
        self.threshold = args.threshold
        self.use_windows = args.use_windows
        self.window_size = args.window_size
        self.overlap = args.overlap
        
        # Filtering parameters
        self.same_pub = args.same_pub
        self.min_shared_words = args.min_shared_words
        
        # Performance parameters
        self.workers = args.workers
        self.checkpoint_interval = args.checkpoint_interval
        self.batch_size = args.batch_size
        
        # Output options
        self.verbose = args.verbose
        self.save_matrix = args.save_matrix

# ============================================================================
# TEXT PREPROCESSING AND N-GRAM FUNCTIONS
# ============================================================================

def clean_text(text: str) -> str:
    """Clean and normalize text for n-gram analysis."""
    if pd.isna(text) or not text:
        return ""
    
    text = str(text).lower().strip()
    text = re.sub(r'[^\w\s]', ' ', text)
    text = re.sub(r'\d+', '', text)
    text = re.sub(r'\s+', ' ', text)
    
    return text.strip()

def create_ngrams(text: str, n: int = 4) -> Set[str]:
    """Create word n-grams from text."""
    words = text.split()
    if len(words) < n:
        return set()
    
    ngrams = set()
    for i in range(len(words) - n + 1):
        ngram = ' '.join(words[i:i+n])
        ngrams.add(ngram)
    
    return ngrams

def create_shingles(text: str, k: int = 5) -> Set[str]:
    """Create character k-shingles from text."""
    if len(text) < k:
        return {text} if text else set()
    
    shingles = set()
    for i in range(len(text) - k + 1):
        shingle = text[i:i+k]
        shingles.add(shingle)
    
    return shingles

def jaccard_similarity(set1: Set[str], set2: Set[str]) -> float:
    """Calculate Jaccard similarity between two sets."""
    if not set1 and not set2:
        return 1.0
    if not set1 or not set2:
        return 0.0
    
    intersection = len(set1.intersection(set2))
    union = len(set1.union(set2))
    
    return intersection / union if union > 0 else 0.0

# ============================================================================
# PRE-FILTERING FUNCTIONS
# ============================================================================

def should_compare_pair(row1: pd.Series, row2: pd.Series, config: Config) -> bool:
    """
    Pre-filtering: decide if pair should be compared.
    Returns True if pair should be compared, False to skip.
    """
    # Filter 1: Same publication filter
    if not config.same_pub and row1['publication'] == row2['publication']:
        return False
    
    # Filter 2: Minimum length filter (skip very short documents)
    # This is kept minimal - only skip truly empty or tiny documents
    len1 = row1['word_count'] if 'word_count' in row1 else len(row1['text_clean'].split())
    len2 = row2['word_count'] if 'word_count' in row2 else len(row2['text_clean'].split())
    
    # Only skip if either document is extremely short (< 10 words)
    if len1 < 10 or len2 < 10:
        return False
    
    return True

# ============================================================================
# PARALLEL COMPARISON FUNCTIONS
# ============================================================================

def find_shared_content(text1: str, text2: str, ngram_size: int = 4) -> Tuple[str, str, str]:
    """Find the longest shared n-gram sequence and extract context."""
    clean1 = clean_text(text1)
    clean2 = clean_text(text2)
    
    words1 = clean1.split()
    words2 = clean2.split()
    
    if len(words1) < ngram_size or len(words2) < ngram_size:
        return "", "", ""
    
    # Find all shared n-grams and their positions
    ngrams1 = {}
    for i in range(len(words1) - ngram_size + 1):
        ngram = ' '.join(words1[i:i+ngram_size])
        if ngram not in ngrams1:
            ngrams1[ngram] = []
        ngrams1[ngram].append(i)
    
    shared_sequences = []
    for i in range(len(words2) - ngram_size + 1):
        ngram = ' '.join(words2[i:i+ngram_size])
        if ngram in ngrams1:
            for pos1 in ngrams1[ngram]:
                shared_sequences.append((ngram, pos1, i))
    
    if not shared_sequences:
        return "", "", ""
    
    # Find the longest contiguous shared sequence
    best_sequence = ""
    best_pos1 = 0
    best_pos2 = 0
    best_length = 0
    
    for ngram, pos1, pos2 in shared_sequences:
        current_length = ngram_size
        extend_pos1 = pos1 + ngram_size
        extend_pos2 = pos2 + ngram_size
        
        while (extend_pos1 < len(words1) and 
               extend_pos2 < len(words2) and 
               words1[extend_pos1] == words2[extend_pos2]):
            current_length += 1
            extend_pos1 += 1
            extend_pos2 += 1
        
        if current_length > best_length:
            best_length = current_length
            best_pos1 = pos1
            best_pos2 = pos2
            best_sequence = ' '.join(words1[pos1:pos1 + current_length])
    
    if not best_sequence:
        ngram, pos1, pos2 = shared_sequences[0]
        best_sequence = ngram
        best_pos1 = pos1
        best_pos2 = pos2
        best_length = ngram_size
    
    # Extract context (50 words before and after)
    context_size = 50
    
    source_start = max(0, best_pos1 - context_size)
    source_end = min(len(words1), best_pos1 + best_length + context_size)
    source_context = ' '.join(words1[source_start:source_end])
    
    target_start = max(0, best_pos2 - context_size)
    target_end = min(len(words2), best_pos2 + best_length + context_size)
    target_context = ' '.join(words2[target_start:target_end])
    
    return best_sequence, source_context, target_context

def compare_pair(pair_data: Tuple, config: Config) -> Optional[Dict]:
    """
    Compare a single pair of documents/windows.
    This function is called in parallel by worker processes.
    """
    idx1, idx2, row1, row2 = pair_data
    
    # Pre-filter check
    if not should_compare_pair(row1, row2, config):
        return None
    
    # Extract text
    text1 = row1['text'] if 'text' in row1 else row1['text_clean']
    text2 = row2['text'] if 'text' in row2 else row2['text_clean']
    
    clean1 = clean_text(text1)
    clean2 = clean_text(text2)
    
    # Skip if too short
    if len(clean1.split()) < config.ngram_size or len(clean2.split()) < config.ngram_size:
        return None
    
    # Create n-grams and shingles
    ngrams1 = create_ngrams(clean1, n=config.ngram_size)
    ngrams2 = create_ngrams(clean2, n=config.ngram_size)
    shingles1 = create_shingles(clean1, k=config.shingle_size)
    shingles2 = create_shingles(clean2, k=config.shingle_size)
    
    # Calculate similarities
    ngram_sim = jaccard_similarity(ngrams1, ngrams2)
    shingle_sim = jaccard_similarity(shingles1, shingles2)
    combined_sim = 0.4 * ngram_sim + 0.6 * shingle_sim
    
    # Check threshold
    if combined_sim < config.threshold:
        return None
    
    # Find shared content
    shared_content, source_context, target_context = find_shared_content(text1, text2, config.ngram_size)
    
    # Additional filter: minimum shared words
    if config.min_shared_words > 0:
        shared_word_count = len(shared_content.split())
        if shared_word_count < config.min_shared_words:
            return None
    
    # Build result dictionary
    result = {
        'source_page_id': row1['page_id'],
        'target_page_id': row2['page_id'],
        'source_publication': row1['publication'],
        'target_publication': row2['publication'],
        'ngram_similarity': ngram_sim,
        'shingle_similarity': shingle_sim,
        'combined_similarity': combined_sim,
        'shared_content': shared_content,
        'source_context': source_context,
        'target_context': target_context
    }
    
    # Add date information if available
    if 'date' in row1:
        result['source_date'] = row1['date']
    if 'date' in row2:
        result['target_date'] = row2['date']
    
    # Add window information if using windows
    if config.use_windows and 'window_id' in row1:
        result.update({
            'source_window_id': row1['window_id'],
            'target_window_id': row2['window_id'],
            'source_combined_id': row1.get('combined_id', ''),
            'target_combined_id': row2.get('combined_id', ''),
            'source_start_word': row1.get('start_word', 0),
            'source_end_word': row1.get('end_word', 0),
            'target_start_word': row2.get('start_word', 0),
            'target_end_word': row2.get('end_word', 0)
        })
    
    return result

# ============================================================================
# WINDOWING FUNCTIONS
# ============================================================================

def create_text_windows(text: str, window_size: int = 200, overlap: int = 50) -> List[Dict]:
    """Split text into overlapping windows."""
    words = clean_text(text).split()
    
    if len(words) <= window_size:
        return [{
            'window_id': 0,
            'start_word': 0,
            'end_word': len(words),
            'text': ' '.join(words),
            'total_page_words': len(words)
        }]
    
    windows = []
    step = window_size - overlap
    window_id = 0
    
    for start_pos in range(0, len(words), step):
        end_pos = min(start_pos + window_size, len(words))
        
        if end_pos - start_pos < window_size // 2:
            break
        
        window_text = ' '.join(words[start_pos:end_pos])
        windows.append({
            'window_id': window_id,
            'start_word': start_pos,
            'end_word': end_pos,
            'text': window_text,
            'total_page_words': len(words)
        })
        
        window_id += 1
        
        if end_pos >= len(words):
            break
    
    return windows

def prepare_windowed_data(metadata: pd.DataFrame, window_size: int = 200, overlap: int = 50) -> pd.DataFrame:
    """Prepare windowed data for analysis."""
    all_windows = []
    
    for idx, row in metadata.iterrows():
        windows = create_text_windows(row['text_clean'], window_size, overlap)
        
        for window in windows:
            window_row = {
                'page_id': row['page_id'],
                'publication': row['publication'],
                'window_id': window['window_id'],
                'start_word': window['start_word'],
                'end_word': window['end_word'],
                'text': window['text'],
                'text_clean': window['text'],
                'total_page_words': window['total_page_words'],
                'combined_id': f"{row['page_id']}_w{window['window_id']}"
            }
            
            if 'date' in row:
                window_row['date'] = row['date']
            
            all_windows.append(window_row)
    
    return pd.DataFrame(all_windows)

# ============================================================================
# MAIN COMPARISON ENGINE WITH PARALLELIZATION
# ============================================================================

def run_parallel_comparison(metadata: pd.DataFrame, config: Config) -> pd.DataFrame:
    """
    Run parallelized pairwise comparison with progress tracking and checkpointing.
    """
    print("\n" + "="*70)
    print("PARALLEL TEXT REUSE DETECTION")
    print("="*70)
    
    # Prepare data
    if config.use_windows:
        print(f"Creating text windows (size={config.window_size}, overlap={config.overlap})...")
        working_data = prepare_windowed_data(metadata, config.window_size, config.overlap)
        print(f"Created {len(working_data)} windows from {len(metadata)} pages")
    else:
        working_data = metadata.copy()
        if 'word_count' not in working_data.columns:
            working_data['word_count'] = working_data['text_clean'].apply(lambda x: len(x.split()))
    
    # Generate all pairs
    n = len(working_data)
    total_pairs = (n * (n - 1)) // 2
    print(f"\nTotal segments to analyze: {n}")
    print(f"Potential pairwise comparisons: {total_pairs:,}")
    print(f"Using {config.workers} parallel workers")
    
    # Create pairs as generator to save memory
    def generate_pairs():
        for i in range(len(working_data)):
            for j in range(i + 1, len(working_data)):
                yield (i, j, working_data.iloc[i], working_data.iloc[j])
    
    # Check for existing checkpoint
    checkpoint_file = os.path.join(config.output_dir, 'checkpoint.pkl')
    if os.path.exists(checkpoint_file):
        print(f"\nFound existing checkpoint file: {checkpoint_file}")
        try:
            with open(checkpoint_file, 'rb') as f:
                checkpoint_data = pickle.load(f)
            results = checkpoint_data['results']
            start_pair = checkpoint_data['completed_pairs']
            print(f"✓ Automatically resuming from pair {start_pair:,}/{total_pairs:,}")
            print(f"  ({len(results):,} matches already found)")
        except Exception as e:
            print(f"Warning: Could not load checkpoint: {e}")
            print("Starting from beginning...")
            results = []
            start_pair = 0
    else:
        print("\nNo checkpoint found - starting fresh analysis")
        results = []
        start_pair = 0
    
    # Process pairs in batches with parallelization
    start_time = time.time()
    processed = start_pair
    
    # Create partial function with config
    compare_func = partial(compare_pair, config=config)
    
    # Process in batches
    batch_pairs = []
    pair_gen = generate_pairs()
    
    # Skip to start_pair if resuming
    for _ in range(start_pair):
        next(pair_gen)
    
    with Pool(processes=config.workers) as pool:
        for pair_data in pair_gen:
            batch_pairs.append(pair_data)
            
            # Process batch when it reaches batch_size
            if len(batch_pairs) >= config.batch_size:
                batch_results = pool.map(compare_func, batch_pairs)
                
                # Filter out None results and add to results list
                batch_results = [r for r in batch_results if r is not None]
                results.extend(batch_results)
                
                processed += len(batch_pairs)
                batch_pairs = []
                
                # Progress update
                if config.verbose and processed % 1000 == 0:
                    elapsed = time.time() - start_time
                    rate = processed / elapsed if elapsed > 0 else 0
                    remaining = total_pairs - processed
                    eta_seconds = remaining / rate if rate > 0 else 0
                    eta_hours = eta_seconds / 3600
                    
                    pct = (processed / total_pairs) * 100
                    print(f"Progress: {processed:,}/{total_pairs:,} ({pct:.1f}%) - "
                          f"Matches: {len(results):,} - "
                          f"Rate: {rate:.1f} pairs/sec - "
                          f"ETA: {eta_hours:.1f}h")
                
                # Checkpoint
                if config.checkpoint_interval > 0 and processed % config.checkpoint_interval == 0:
                    checkpoint_data = {
                        'results': results,
                        'completed_pairs': processed,
                        'timestamp': datetime.now().isoformat()
                    }
                    with open(checkpoint_file, 'wb') as f:
                        pickle.dump(checkpoint_data, f)
                    if config.verbose:
                        print(f"  Checkpoint saved ({len(results):,} matches so far)")
        
        # Process remaining pairs
        if batch_pairs:
            batch_results = pool.map(compare_func, batch_pairs)
            batch_results = [r for r in batch_results if r is not None]
            results.extend(batch_results)
            processed += len(batch_pairs)
    
    # Final statistics
    elapsed = time.time() - start_time
    print(f"\n{'='*70}")
    print(f"Comparison complete!")
    print(f"Total comparisons: {processed:,}")
    print(f"Matches found: {len(results):,}")
    print(f"Total time: {elapsed/3600:.2f} hours")
    print(f"Average rate: {processed/elapsed:.1f} pairs/sec")
    print(f"{'='*70}")
    
    # Remove checkpoint file if completed successfully
    if os.path.exists(checkpoint_file):
        os.remove(checkpoint_file)
        print("Checkpoint file removed")
    
    return pd.DataFrame(results)

# ============================================================================
# POST-PROCESSING FUNCTIONS
# ============================================================================

def classify_reuse_type(row):
    """Classify the type of text reuse based on similarity scores."""
    combined_sim = row['combined_similarity']
    ngram_sim = row['ngram_similarity']
    
    if combined_sim > 0.7 and ngram_sim > 0.6:
        return 'substantial_reuse'
    elif combined_sim > 0.4 and ngram_sim > 0.3:
        return 'moderate_reuse'
    elif combined_sim > 0.2 and ngram_sim > 0.15:
        return 'partial_reuse'
    else:
        return 'minimal_reuse'

def filter_boilerplate(reuse_df: pd.DataFrame, min_occurrences: int = 5) -> pd.DataFrame:
    """Remove boilerplate content that appears frequently."""
    if len(reuse_df) == 0:
        return reuse_df
    
    print("\n" + "="*70)
    print("BOILERPLATE FILTERING")
    print("="*70)
    
    original_count = len(reuse_df)
    
    # Stage 1: Remove very short shared content
    min_words = 10
    short_content = reuse_df['shared_content'].apply(lambda x: len(x.split()) < min_words)
    reuse_df = reuse_df[~short_content].copy()
    print(f"Stage 1: Removed {short_content.sum()} matches with <{min_words} words")
    
    # Stage 2: Identify repeated content
    content_counts = reuse_df['shared_content'].value_counts()
    frequent_content = content_counts[content_counts >= min_occurrences].index
    is_boilerplate = reuse_df['shared_content'].isin(frequent_content)
    
    print(f"Stage 2: Identified {len(frequent_content)} pieces appearing {min_occurrences}+ times")
    print(f"  Removing {is_boilerplate.sum()} boilerplate matches")
    
    reuse_df = reuse_df[~is_boilerplate].copy()
    
    print(f"\nFiltering summary:")
    print(f"  Started with: {original_count}")
    print(f"  Final count: {len(reuse_df)}")
    print("="*70)
    
    return reuse_df

# ============================================================================
# DATA LOADING
# ============================================================================

def load_and_prepare_metadata(config: Config) -> pd.DataFrame:
    """Load metadata and text files."""
    print("\n" + "="*70)
    print("LOADING DATA")
    print("="*70)
    
    # Load metadata
    metadata = pd.read_csv(config.metadata_file)
    print(f"Loaded metadata: {len(metadata)} entries")
    
    # Load text files
    text_dir = Path(config.input_dir)
    
    texts = []
    for idx, row in metadata.iterrows():
        page_id = row['page_id']
        text_file = text_dir / f"{page_id}.txt"
        
        if text_file.exists():
            with open(text_file, 'r', encoding='utf-8') as f:
                text = f.read()
            texts.append(text)
        else:
            texts.append("")
            if config.verbose:
                print(f"Warning: Text file not found for {page_id}")
    
    metadata['text_clean'] = [clean_text(t) for t in texts]
    metadata['word_count'] = metadata['text_clean'].apply(lambda x: len(x.split()))
    
    # Filter out empty texts
    metadata = metadata[metadata['word_count'] > 0].reset_index(drop=True)
    
    print(f"Loaded {len(metadata)} documents with text")
    print(f"Total words: {metadata['word_count'].sum():,}")
    print(f"Average words per document: {metadata['word_count'].mean():.1f}")
    print("="*70)
    
    return metadata

# ============================================================================
# MAIN EXECUTION
# ============================================================================

def main():
    """Main execution function."""
    parser = argparse.ArgumentParser(
        description='Optimized N-gram text reuse analysis with parallelization'
    )
    
    # Required arguments
    parser.add_argument('--input_dir', required=True, help='Directory containing text files')
    parser.add_argument('--output_dir', required=True, help='Output directory for results')
    parser.add_argument('--metadata', required=True, help='Path to metadata CSV file')
    
    # Analysis parameters
    parser.add_argument('--ngram_size', type=int, default=4, help='N-gram size (default: 4)')
    parser.add_argument('--shingle_size', type=int, default=5, help='Character shingle size (default: 5)')
    parser.add_argument('--threshold', type=float, default=0.12, help='Similarity threshold (default: 0.12)')
    parser.add_argument('--use_windows', action='store_true', help='Use windowed analysis')
    parser.add_argument('--window_size', type=int, default=200, help='Window size in words (default: 200)')
    parser.add_argument('--overlap', type=int, default=50, help='Window overlap in words (default: 50)')
    
    # Filtering parameters
    parser.add_argument('--same_pub', action='store_true', help='Compare documents within same publication')
    parser.add_argument('--min_shared_words', type=int, default=10, help='Minimum shared words in match (default: 10)')
    
    # These filters are available but not recommended for this project
    # parser.add_argument('--max_date_diff', type=int, default=None, help='Max days between documents')
    # parser.add_argument('--min_length', type=int, default=10, help='Minimum document length')
    # parser.add_argument('--max_length_diff_ratio', type=float, default=999, help='Max length ratio')
    
    # Performance parameters
    parser.add_argument('--workers', type=int, default=None, help='Number of parallel workers (default: auto)')
    parser.add_argument('--checkpoint_interval', type=int, default=5000, help='Save checkpoint every N pairs (default: 5000, 0 to disable)')
    parser.add_argument('--batch_size', type=int, default=100, help='Process pairs in batches of N (default: 100)')
    
    # Output options
    parser.add_argument('--verbose', action='store_true', help='Verbose output')
    parser.add_argument('--save_matrix', action='store_true', help='Save full similarity matrix')
    
    args = parser.parse_args()
    
    # Set workers to CPU count if not specified
    if args.workers is None:
        args.workers = max(1, cpu_count() - 1)
    
    # Create config
    config = Config(args)
    
    # Create output directory
    os.makedirs(config.output_dir, exist_ok=True)
    
    print("="*70)
    print("N-GRAMS AND JACCARD SIMILARITY TEXT REUSE ANALYSIS - OPTIMIZED")
    print("="*70)
    print(f"Configuration:")
    print(f"  Input: {config.input_dir}")
    print(f"  Output: {config.output_dir}")
    print(f"  Metadata: {config.metadata_file}")
    print(f"  N-gram size: {config.ngram_size}")
    print(f"  Threshold: {config.threshold}")
    print(f"  Workers: {config.workers}")
    print(f"  Use windows: {config.use_windows}")
    if config.use_windows:
        print(f"  Window size: {config.window_size} words")
        print(f"  Overlap: {config.overlap} words")
    print("="*70)
    
    # Load data
    metadata = load_and_prepare_metadata(config)
    
    # Run analysis
    reuse_results = run_parallel_comparison(metadata, config)
    
    if len(reuse_results) == 0:
        print("\nNo text reuse found above threshold.")
        return
    
    # Classify reuse types
    reuse_results['reuse_type'] = reuse_results.apply(classify_reuse_type, axis=1)
    
    # Filter boilerplate
    reuse_filtered = filter_boilerplate(reuse_results)
    
    # Save results
    suffix = "windowed" if config.use_windows else "fullpage"
    results_file = os.path.join(config.output_dir, f'text_reuse_ngrams_{suffix}.csv')
    filtered_file = os.path.join(config.output_dir, f'text_reuse_ngrams_{suffix}_filtered.csv')
    
    reuse_results.to_csv(results_file, index=False)
    reuse_filtered.to_csv(filtered_file, index=False)
    
    # Summary statistics
    print("\n" + "="*70)
    print("ANALYSIS RESULTS")
    print("="*70)
    print(f"Total matches found: {len(reuse_results)}")
    print(f"After filtering: {len(reuse_filtered)}")
    
    if len(reuse_filtered) > 0:
        print("\nReuse type distribution:")
        print(reuse_filtered['reuse_type'].value_counts())
        
        print(f"\nSimilarity statistics:")
        print(f"  Average combined similarity: {reuse_filtered['combined_similarity'].mean():.3f}")
        print(f"  Max combined similarity: {reuse_filtered['combined_similarity'].max():.3f}")
        print(f"  Min combined similarity: {reuse_filtered['combined_similarity'].min():.3f}")
        
        avg_shared_length = reuse_filtered['shared_content'].apply(lambda x: len(x.split())).mean()
        print(f"  Average shared content length: {avg_shared_length:.1f} words")
        
        if config.use_windows:
            unique_page_pairs = reuse_filtered[['source_page_id', 'target_page_id']].drop_duplicates()
            print(f"  Unique page pairs with reuse: {len(unique_page_pairs)}")
    
    print(f"\n{'='*70}")
    print("✅ Analysis complete!")
    print(f"{'='*70}")
    print(f"Generated files in {config.output_dir}/:")
    print(f"  - {os.path.basename(results_file)}")
    print(f"  - {os.path.basename(filtered_file)}")
    print("="*70)

if __name__ == "__main__":
    main()
