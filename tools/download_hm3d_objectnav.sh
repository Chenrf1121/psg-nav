#!/bin/bash
# Download HM3D ObjectNav dataset
# Based on habitat-lab DATASETS.md documentation

set -e

echo "=================================="
echo "HM3D ObjectNav Dataset Downloader"
echo "=================================="
echo ""

# Create target directory
TARGET_DIR="data/HM3D/datasets/objectnav/hm3d"
mkdir -p "$TARGET_DIR"

echo "Target directory: $TARGET_DIR"
echo ""

# Option to choose version
echo "Choose HM3D ObjectNav version to download:"
echo "  1) v1 (154 MB) - Uses HM3DSem-v0.1"
echo "  2) v2 (245 MB) - Uses HM3DSem-v0.2 [Recommended]"
echo ""
read -p "Enter choice (1 or 2): " VERSION_CHOICE

if [ "$VERSION_CHOICE" == "1" ]; then
    VERSION="v1"
    DOWNLOAD_URL="https://dl.fbaipublicfiles.com/habitat/data/datasets/objectnav/hm3d/v1/objectnav_hm3d_v1.zip"
    SIZE="154 MB"
    echo "Note: v1 uses HM3DSem-v0.1 (120 scenes)"
elif [ "$VERSION_CHOICE" == "2" ]; then
    VERSION="v2"
    DOWNLOAD_URL="https://dl.fbaipublicfiles.com/habitat/data/datasets/objectnav/hm3d/v2/objectnav_hm3d_v2.zip"
    SIZE="245 MB"
    echo "Note: v2 uses HM3DSem-v0.2 (216 scenes) - Recommended for 2023+ challenges"
else
    echo "Invalid choice. Exiting."
    exit 1
fi

echo ""
echo "Downloading HM3D ObjectNav $VERSION ($SIZE)..."
echo "URL: $DOWNLOAD_URL"
echo ""

# Download the dataset
cd "$TARGET_DIR"
wget -c "$DOWNLOAD_URL"

# Extract the dataset
ZIPFILE="objectnav_hm3d_${VERSION}.zip"
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
echo "Note: Make sure you also have the HM3D scene dataset downloaded."
echo "If not, you can download it using:"
echo "  python -m habitat_sim.utils.datasets_download --username <api-token-id> --password <api-token-secret> --uids hm3d_minival_v0.2"
echo ""
echo "For HM3D scene dataset access, register at: https://matterport.com/habitat-matterport-3d-research-dataset"
echo ""
