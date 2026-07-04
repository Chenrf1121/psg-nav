#!/bin/bash
# Download HSSD ObjectNav dataset
# HSSD (Habitat Synthetic Scenes Dataset) is a procedurally generated dataset

set -e

echo "=================================="
echo "HSSD ObjectNav Dataset Downloader"
echo "=================================="
echo ""

# Create target directory
TARGET_DIR="data/HSSD/datasets/objectnav/hssd"
mkdir -p "$TARGET_DIR"

echo "Target directory: $TARGET_DIR"
echo ""

# HSSD dataset information
VERSION="v1"
DOWNLOAD_URL="https://dl.fbaipublicfiles.com/habitat/data/datasets/objectnav/hssd/v1/objectnav_hssd_v1.zip"

echo "Downloading HSSD ObjectNav $VERSION..."
echo "URL: $DOWNLOAD_URL"
echo ""
echo "Note: If the URL doesn't work, HSSD dataset might need to be downloaded through:"
echo "  1. Habitat's official repository"
echo "  2. Using habitat_sim.utils.datasets_download"
echo ""

# Download the dataset
cd "$TARGET_DIR"
wget -c "$DOWNLOAD_URL" || {
    echo ""
    echo "=================================="
    echo "Direct download failed!"
    echo "=================================="
    echo ""
    echo "Alternative download methods:"
    echo ""
    echo "Method 1: Using habitat-sim datasets downloader"
    echo "  python -m habitat_sim.utils.datasets_download --username <api-token-id> --password <api-token-secret> --uids hssd-hab"
    echo ""
    echo "Method 2: Download from Habitat Challenge data"
    echo "  If you're participating in Habitat Challenge, HSSD might be available through challenge data"
    echo ""
    echo "Method 3: Manual download"
    echo "  Visit: https://aihabitat.org/datasets/hssd/"
    echo "  Or check: https://github.com/facebookresearch/habitat-sim/blob/main/DATASETS.md"
    echo ""
    exit 1
}

# Extract the dataset
ZIPFILE="objectnav_hssd_${VERSION}.zip"
echo ""
echo "Extracting $ZIPFILE..."
unzip -o "$ZIPFILE"

# Clean up
echo ""
echo "Cleaning up..."
rm "$ZIPFILE"

echo ""
echo "=================================="
echo "Download completed successfully!"
echo "=================================="
echo ""
echo "Dataset location: $TARGET_DIR/$VERSION"
echo ""
echo "Note: You also need to download the HSSD scene dataset."
echo "HSSD scenes can be downloaded using:"
echo "  python -m habitat_sim.utils.datasets_download --username <api-token-id> --password <api-token-secret> --uids hssd-hab"
echo ""
echo "For HSSD dataset access, visit: https://aihabitat.org/datasets/hssd/"
echo ""
