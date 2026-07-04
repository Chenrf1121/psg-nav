#!/bin/bash

# HM3D 数据集下载脚本
# 用于 Habitat 2022 ObjectNav Challenge

set -e

echo "=========================================="
echo "HM3D Dataset Download Script"
echo "=========================================="
echo ""

# 设置颜色
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# 检查参数
DOWNLOAD_SCENES=true
DOWNLOAD_EPISODES=true

while [[ $# -gt 0 ]]; do
    case $1 in
        --scenes-only)
            DOWNLOAD_EPISODES=false
            shift
            ;;
        --episodes-only)
            DOWNLOAD_SCENES=false
            shift
            ;;
        *)
            echo "Unknown option: $1"
            echo "Usage: $0 [--scenes-only | --episodes-only]"
            exit 1
            ;;
    esac
done

# 创建目录
BASE_DIR="data"
mkdir -p "$BASE_DIR"

# ========================================
# 1. 下载 HM3D 场景数据集
# ========================================
if [ "$DOWNLOAD_SCENES" = true ]; then
    echo -e "${YELLOW}[1/2] 下载 HM3D 场景数据集...${NC}"

    SCENES_DIR="$BASE_DIR/scene_datasets"
    mkdir -p "$SCENES_DIR"
    cd "$SCENES_DIR"

    # 下载验证集场景 (约 1.4 GB)
    SCENE_FILE="hm3d-val-habitat.tar"
    SCENE_URL="https://dl.fbaipublicfiles.com/habitat/data/scene_datasets/hm3d-1.0/$SCENE_FILE"

    if [ -f "$SCENE_FILE" ]; then
        echo -e "${GREEN}✓ 场景文件已存在，跳过下载${NC}"
    else
        echo "  下载 $SCENE_FILE (约 1.4 GB)..."
        wget -c "$SCENE_URL" || {
            echo -e "${RED}✗ 下载失败！${NC}"
            echo "  请手动下载: $SCENE_URL"
            exit 1
        }
    fi

    # 解压
    if [ -d "hm3d" ]; then
        echo -e "${GREEN}✓ 场景已解压${NC}"
    else
        echo "  解压场景文件..."
        tar -xvf "$SCENE_FILE"
        echo -e "${GREEN}✓ 场景解压完成${NC}"
    fi

    # 清理
    read -p "  删除压缩文件以节省空间? (y/n) " -n 1 -r
    echo
    if [[ $REPLY =~ ^[Yy]$ ]]; then
        rm "$SCENE_FILE"
        echo -e "${GREEN}✓ 压缩文件已删除${NC}"
    fi

    cd - > /dev/null
fi

# ========================================
# 2. 下载 ObjectNav Episode 数据集
# ========================================
if [ "$DOWNLOAD_EPISODES" = true ]; then
    echo ""
    echo -e "${YELLOW}[2/2] 下载 ObjectNav Episode 数据集...${NC}"

    EPISODES_DIR="$BASE_DIR/datasets/objectnav/hm3d/v1"
    mkdir -p "$EPISODES_DIR"
    cd "$EPISODES_DIR"

    # 下载 episodes (约 154 MB)
    EPISODE_FILE="objectnav_hm3d_v1.zip"
    EPISODE_URL="https://dl.fbaipublicfiles.com/habitat/data/datasets/objectnav/hm3d/v1/$EPISODE_FILE"

    if [ -f "$EPISODE_FILE" ]; then
        echo -e "${GREEN}✓ Episode 文件已存在，跳过下载${NC}"
    else
        echo "  下载 $EPISODE_FILE (约 154 MB)..."
        wget -c "$EPISODE_URL" || {
            echo -e "${RED}✗ 下载失败！${NC}"
            echo "  请手动下载: $EPISODE_URL"
            exit 1
        }
    fi

    # 解压
    if [ -d "val" ]; then
        echo -e "${GREEN}✓ Episodes 已解压${NC}"
    else
        echo "  解压 episode 文件..."
        unzip -o "$EPISODE_FILE"
        echo -e "${GREEN}✓ Episodes 解压完成${NC}"
    fi

    # 清理
    read -p "  删除压缩文件以节省空间? (y/n) " -n 1 -r
    echo
    if [[ $REPLY =~ ^[Yy]$ ]]; then
        rm "$EPISODE_FILE"
        echo -e "${GREEN}✓ 压缩文件已删除${NC}"
    fi

    cd - > /dev/null
fi

# ========================================
# 验证下载
# ========================================
echo ""
echo -e "${YELLOW}验证下载...${NC}"

VALIDATION_PASSED=true

# 检查场景目录
if [ "$DOWNLOAD_SCENES" = true ]; then
    if [ -d "$BASE_DIR/scene_datasets/hm3d/val" ]; then
        SCENE_COUNT=$(ls -1 "$BASE_DIR/scene_datasets/hm3d/val" | wc -l)
        echo -e "${GREEN}✓ 场景目录存在 ($SCENE_COUNT 个场景)${NC}"
    else
        echo -e "${RED}✗ 场景目录不存在${NC}"
        VALIDATION_PASSED=false
    fi
fi

# 检查 episode 文件
if [ "$DOWNLOAD_EPISODES" = true ]; then
    if [ -f "$BASE_DIR/datasets/objectnav/hm3d/v1/val/val.json.gz" ]; then
        echo -e "${GREEN}✓ Episode 文件存在${NC}"
    else
        echo -e "${RED}✗ Episode 文件不存在${NC}"
        VALIDATION_PASSED=false
    fi
fi

echo ""
echo "=========================================="
if [ "$VALIDATION_PASSED" = true ]; then
    echo -e "${GREEN}✅ HM3D 数据集下载完成！${NC}"
    echo ""
    echo "下一步："
    echo "1. 修改 PSG_Nav.py:1067 使用 challenge_objectnav2022 配置"
    echo "2. 更新 utils/utils_glip.py 中的类别列表"
    echo "3. 运行: python test_hm3d_setup.py 验证设置"
    echo "4. 运行: python PSG_Nav.py --split_l 0 --split_r 10 测试"
else
    echo -e "${RED}❌ 下载验证失败，请检查错误信息${NC}"
    exit 1
fi
echo "=========================================="
