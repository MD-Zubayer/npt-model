#!/bin/bash

# NPT Model Environment Setup Script
# Sets up the complete development environment for training

set -e  # Exit on error

echo "=========================================="
echo "NPT Model Environment Setup"
echo "=========================================="

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Check Python version
echo -e "${YELLOW}Checking Python version...${NC}"
python_version=$(python3 --version 2>&1 | awk '{print $2}')
echo "Python version: $python_version"

# Create virtual environment
echo -e "${YELLOW}Creating virtual environment...${NC}"
if [ ! -d "venv" ]; then
    python3 -m venv venv
    echo -e "${GREEN}✓ Virtual environment created${NC}"
else
    echo -e "${YELLOW}Virtual environment already exists${NC}"
fi

# Activate virtual environment
echo -e "${YELLOW}Activating virtual environment...${NC}"
source venv/bin/activate

# Upgrade/install bootstrap tools (best effort)
echo -e "${YELLOW}Upgrading pip...${NC}"
if ! pip install --upgrade pip setuptools wheel; then
    echo -e "${YELLOW}⚠ Network না থাকায় pip/setuptools upgrade skip করা হলো${NC}"
fi

# Install requirements (best effort)
echo -e "${YELLOW}Installing requirements...${NC}"
if ! pip install -r requirements.txt; then
    echo -e "${YELLOW}⚠ requirements install সম্পূর্ণ হয়নি (সম্ভবত network issue)${NC}"
fi

# Install development dependencies (best effort)
echo -e "${YELLOW}Installing development dependencies...${NC}"
if ! pip install -e ".[dev]"; then
    echo -e "${YELLOW}⚠ dev dependencies install skip করা হলো${NC}"
fi

# Create necessary directories
echo -e "${YELLOW}Creating directories...${NC}"
mkdir -p model/checkpoints
mkdir -p outputs
mkdir -p experiments/logs
mkdir -p experiments/wandb
mkdir -p data/raw/{web_scraped,books,docs,conversations}
mkdir -p configs

echo -e "${GREEN}=========================================="
echo "✓ Environment setup completed!"
echo -e "==========================================${NC}"

echo ""
echo "Next steps:"
echo "1. Activate environment: source venv/bin/activate"
echo "2. Configure data sources in data_pipeline/"
echo "3. Run training: python training/pretraining/train.py"
echo ""

# Optional: Print GPU info if CUDA is available
if command -v nvidia-smi &> /dev/null; then
    echo -e "${YELLOW}GPU Information:${NC}"
    nvidia-smi --query-gpu=index,name,memory.total --format=csv,noheader
else
    echo -e "${YELLOW}GPU not detected. Will use CPU for training.${NC}"
fi
