#!/bin/bash

# ============================================================================
# DualCnst Full Pipeline
# ============================================================================
# Usage: bash scripts/run_full_pipeline.sh <dataset> <score> [options]
#
# Examples:
#   bash scripts/run_full_pipeline.sh ImageNet DualCnst
#   bash scripts/run_full_pipeline.sh bird200 NegLabel --skip-label-generation
#   bash scripts/run_full_pipeline.sh ImageNet10 DualCnst --skip-all-checks
# ============================================================================

set -e

# Configuration
DATASET=${1:-ImageNet}
SCORE=${2:-DualCnst}
EXPERIMENT_NAME=""
SKIP_LABEL_GEN=false
SKIP_IMAGE_GEN=false
SKIP_ALL_CHECKS=false

shift 2 2>/dev/null || true
while [[ $# -gt 0 ]]; do
    case $1 in
        --skip-label-generation)
            SKIP_LABEL_GEN=true
            shift
            ;;
        --skip-image-generation)
            SKIP_IMAGE_GEN=true
            shift
            ;;
        --skip-all-checks)
            SKIP_ALL_CHECKS=true
            shift
            ;;
        *)
            if [ -z "$EXPERIMENT_NAME" ] && [[ ! $1 == --* ]]; then
                EXPERIMENT_NAME=$1
                shift
            else
                echo "Unknown parameter: $1"
                exit 1
            fi
            ;;
    esac
done

if [ -z "$EXPERIMENT_NAME" ]; then
    EXPERIMENT_NAME="pipeline_exp"
fi
PROJECT_ROOT=$(cd "$(dirname "$0")/.." && pwd)
OOD_NEG_MINING_DIR="$PROJECT_ROOT/OODNegMining"
GENERATION_SCRIPT="$PROJECT_ROOT/utils/generation_image_SDXLTurbo.py"
EVAL_SCRIPT="$PROJECT_ROOT/scripts/eval_dualcnst.sh"

declare -A DATASET_LABEL_MAPPING=(
    ["ImageNet"]="ImageNet"
    ["ImageNet10"]="ImageNet"
    ["ImageNet20"]="ImageNet"
    ["ImageNet100"]="ImageNet"
    ["bird200"]="CUB"
    ["food101"]="Food"
    ["car196"]="Stanford-Cars"
    ["pet37"]="Oxford-Pet"
    ["CXR"]="CXR"
)

LABEL_SUBDIR=${DATASET_LABEL_MAPPING[$DATASET]:-$DATASET}

declare -A POSITIVE_FILENAME_MAPPING=(
    ["ImageNet"]="ImageNet.txt"
    ["ImageNet10"]="ImageNet10.txt"
    ["ImageNet20"]="ImageNet20.txt"
    ["ImageNet100"]="ImageNet100.txt"
    ["bird200"]="positive_samples_bird200.txt"
    ["food101"]="positive_samples_food101.txt"
    ["car196"]="positive_samples_car196.txt"
    ["pet37"]="positive_samples_pet37.txt"
    ["CXR"]="CXR.txt"
)

declare -A NEGATIVE_FILENAME_MAPPING=(
    ["ImageNet"]="ImageNet_neg.txt"
    ["ImageNet10"]="ImageNet10_neg.txt"
    ["ImageNet20"]="ImageNet20_neg.txt"
    ["ImageNet100"]="ImageNet100_neg.txt"
    ["bird200"]="low_similarity_neg_samples_bird200.txt"
    ["food101"]="low_similarity_neg_samples_food101.txt"
    ["car196"]="low_similarity_neg_samples_car196.txt"
    ["pet37"]="low_similarity_neg_samples_pet37.txt"
    ["CXR"]="CXR_neg.txt"
)

POS_FILENAME=${POSITIVE_FILENAME_MAPPING[$DATASET]}
NEG_FILENAME=${NEGATIVE_FILENAME_MAPPING[$DATASET]}

POS_LABEL_FILE="$OOD_NEG_MINING_DIR/$LABEL_SUBDIR/positive/$POS_FILENAME"
NEG_LABEL_FILE="$OOD_NEG_MINING_DIR/$LABEL_SUBDIR/negative/$NEG_FILENAME"

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
CYAN='\033[0;36m'
NC='\033[0m'

print_step() {
    echo -e "\n${BLUE} [$1/3]${NC} ${CYAN}$2${NC}"
}

print_info() {
    echo -e "  ${GREEN}${NC} $1"
}

print_warning() {
    echo -e "  ${YELLOW}${NC} $1"
}

print_error() {
    echo -e "  ${RED}${NC} $1"
}

# Print configuration
echo ""
echo "═══════════════════════════════════════════════════════════════════════"
echo "  DualCnst Full Pipeline"
echo "═══════════════════════════════════════════════════════════════════════"
echo "  Dataset:     $DATASET"
echo "  Score:       $SCORE"
echo "  Experiment:  $EXPERIMENT_NAME"
echo "═══════════════════════════════════════════════════════════════════════"

# Step 1: OOD Label Selection
print_step 1 "OOD Label Selection"

if [ "$SKIP_LABEL_GEN" = true ]; then
    print_warning "Skipped (--skip-label-generation)"
elif [ "$SKIP_ALL_CHECKS" = false ] && [ -f "$POS_LABEL_FILE" ] && [ -f "$NEG_LABEL_FILE" ]; then
    print_info "Labels already exist ($(wc -l < "$POS_LABEL_FILE") pos, $(wc -l < "$NEG_LABEL_FILE") neg)"
else
    if [ ! -f "$OOD_NEG_MINING_DIR/select_neglabel_cli.py" ]; then
        print_error "select_neglabel_cli.py not found"
        exit 1
    fi

    python "$OOD_NEG_MINING_DIR/select_neglabel_cli.py" --dataset "$DATASET"

    if [ -f "$POS_LABEL_FILE" ] && [ -f "$NEG_LABEL_FILE" ]; then
        print_info "Generated $(wc -l < "$POS_LABEL_FILE") positive and $(wc -l < "$NEG_LABEL_FILE") negative labels"
    else
        print_error "Label generation failed"
        exit 1
    fi
fi

# Step 2: Image Generation
print_step 2 "Image Generation"

if [ "$SKIP_IMAGE_GEN" = true ]; then
    print_warning "Skipped (--skip-image-generation)"
else
    if [ ! -f "$GENERATION_SCRIPT" ]; then
        print_error "Generation script not found: $GENERATION_SCRIPT"
        exit 1
    fi

    if command -v conda &> /dev/null; then
        print_info "Running in 'sd' conda environment..."
        conda run -n sd python "$GENERATION_SCRIPT"
    else
        print_warning "Conda not detected, running directly (may fail)"
        python "$GENERATION_SCRIPT"
    fi

    print_info "Image generation complete"
fi

# Step 3: OOD Detection Evaluation
print_step 3 "OOD Detection Evaluation"

if [ ! -f "$EVAL_SCRIPT" ]; then
    print_error "Evaluation script not found: $EVAL_SCRIPT"
    exit 1
fi

if command -v conda &> /dev/null; then
    print_info "Running in 'ood' conda environment..."
    conda run -n ood bash "$EVAL_SCRIPT" "$EXPERIMENT_NAME" "$DATASET" "$SCORE"
else
    print_warning "Conda not detected, running directly (may fail)"
    bash "$EVAL_SCRIPT" "$EXPERIMENT_NAME" "$DATASET" "$SCORE"
fi

print_info "Evaluation complete"

echo ""
echo "═══════════════════════════════════════════════════════════════════════"
echo -e "  ${GREEN} Pipeline Complete${NC}"
echo "═══════════════════════════════════════════════════════════════════════"
echo "  Results saved to: results/$DATASET/$SCORE/"
echo "═══════════════════════════════════════════════════════════════════════"
echo ""
