#!/bin/bash
#SBATCH --job-name=ngram_opt_340
#SBATCH --output=logs/ngram_opt_%j.out
#SBATCH --error=logs/ngram_opt_%j.err
#SBATCH --time=16:00:00              # Reduced from 24h - optimizations should speed this up
#SBATCH --mem=24GB                   # 24GB RAM
#SBATCH --cpus-per-task=8           # 8 cores for parallelization
#SBATCH --partition=standard         # Adjust to your cluster's partition name
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=your_email@university.edu

# ============================================================================
# OPTIMIZED NGRAM TEXT REUSE ANALYSIS - 288 ISSUES
# ============================================================================
# 
# Key optimizations:
# - Parallel processing with 8 cores
# - Checkpointing allows resuming if interrupted
# - Progress tracking with ETA
# - Memory-efficient batch processing
# - Compares ALL document pairs (no length/date filtering per user request)
#
# ============================================================================

echo "=========================================="
echo "OPTIMIZED NGRAM TEXT REUSE - 288 ISSUES"
echo "=========================================="
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURM_NODELIST"
echo "CPUs: $SLURM_CPUS_PER_TASK"
echo "Memory: $SLURM_MEM_PER_NODE"
echo "Start time: $(date)"
echo "=========================================="

# Create directories
mkdir -p logs
mkdir -p results/288_issues_optimized

# Load modules (adjust to your cluster)
module purge
module load python/3.10

# Activate virtual environment
source /path/to/your/venv/bin/activate
# OR: conda activate your_env_name

# Set environment variables
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
export PYTHONUNBUFFERED=1

# Navigate to working directory
cd $SLURM_SUBMIT_DIR

# Verify input data
echo ""
echo "Checking input data..."
INPUT_DIR="./data/text_files"
METADATA_FILE="./metadata.csv"

NUM_FILES=$(find $INPUT_DIR -name "*.txt" | wc -l)
echo "Found $NUM_FILES text files in $INPUT_DIR"

if [ ! -f "$METADATA_FILE" ]; then
    echo "ERROR: Metadata file not found: $METADATA_FILE"
    exit 1
fi

if [ $NUM_FILES -eq 0 ]; then
    echo "ERROR: No text files found in $INPUT_DIR"
    exit 1
fi

# Run optimized analysis
echo ""
echo "Starting OPTIMIZED ngram analysis..."
echo "Optimizations enabled:"
echo "  ✓ Parallel processing (8 workers)"
echo "  ✓ Checkpointing every 5000 comparisons"
echo "  ✓ Batch processing (100 pairs/batch)"
echo "  ✓ Progress tracking with ETA"
echo "  ✓ All document pairs compared (no length/date filters)"
echo ""

# BASIC CONFIGURATION (fastest, good for initial exploration)
# Uncomment this for a quick first run with higher threshold:
# python3 ngramtextreuse_optimized.py \
#     --input_dir $INPUT_DIR \
#     --output_dir results/288_issues_optimized \
#     --metadata $METADATA_FILE \
#     --threshold 0.12 \
#     --min_shared_words 5 \
#     --workers $SLURM_CPUS_PER_TASK \
#     --verbose

# WINDOWED ANALYSIS (more granular, recommended)
python3 ngramtextreuse_optimized.py \
    --input_dir $INPUT_DIR \
    --output_dir results/288_issues_optimized \
    --metadata $METADATA_FILE \
    --ngram_size 4 \
    --shingle_size 5 \
    --threshold 0.12 \
    --use_windows \
    --window_size 200 \
    --overlap 50 \
    --min_shared_words 12 \
    --workers $SLURM_CPUS_PER_TASK \
    --checkpoint_interval 5000 \
    --batch_size 100 \
    --verbose

# CROSS-PUBLICATION ONLY (compare between publications, not within)
# Uncomment this to only find text reuse BETWEEN different publications:
# python3 ngramtextreuse_optimized.py \
#     --input_dir $INPUT_DIR \
#     --output_dir results/288_issues_optimized \
#     --metadata $METADATA_FILE \
#     --threshold 0.12 \
#     --min_shared_words 12 \
#     --workers $SLURM_CPUS_PER_TASK \
#     --verbose
# Note: removed --same_pub flag to only compare between publications

EXIT_CODE=$?

echo ""
echo "=========================================="
echo "Job completed with exit code: $EXIT_CODE"
echo "End time: $(date)"
echo "=========================================="

# Calculate runtime
ELAPSED_TIME=$SECONDS
HOURS=$((ELAPSED_TIME / 3600))
MINUTES=$(((ELAPSED_TIME % 3600) / 60))
echo "Total runtime: ${HOURS}h ${MINUTES}m"

# Check outputs
if [ $EXIT_CODE -eq 0 ]; then
    echo ""
    echo "Output files:"
    ls -lh results/340_issues_optimized/
    
    # Count results
    RESULTS_FILE="results/340_issues_optimized/text_reuse_ngrams_windowed_filtered.csv"
    if [ -f "$RESULTS_FILE" ]; then
        NUM_MATCHES=$(tail -n +2 "$RESULTS_FILE" | wc -l)
        echo ""
        echo "Found $NUM_MATCHES filtered text reuse matches"
    fi
    
    # Show resource usage
    echo ""
    echo "Resource usage summary:"
    sacct -j $SLURM_JOB_ID --format=JobID,JobName,MaxRSS,Elapsed,CPUTime
fi

# Clean up checkpoint if exists
if [ -f "results/288_issues_optimized/checkpoint.pkl" ]; then
    rm results/288_issues_optimized/checkpoint.pkl
    echo "Removed checkpoint file"
fi

exit $EXIT_CODE
