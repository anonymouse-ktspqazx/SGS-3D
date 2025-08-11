#  SGS-3D: High-Fidelity 3D Instance Segmentation via Reliable Semantic Mask Splitting and Growing

This repository contains the implementation of **SGS-3D**, our novel 3D instance segmentation method submitted for anonymous review. SGS-3D introduces significant improvements in 3D instance generation from 2D masks, with a focus on ScanNet200 dataset.


## 🔧 Installation

### Environment Requirements

The following installation guide assumes `python=3.8`, `pytorch=1.12.1`, `cuda=11.3`. You may adjust according to your system.

### 1. Create Environment

```bash
conda create -n SGS3D python=3.8
conda activate SGS3D
conda install pytorch==1.12.1 torchvision==0.13.1 cudatoolkit=11.3 -c pytorch
```

### 2. Install Core Dependencies

```bash
# Install spconv
pip3 install spconv-cu113==2.1.25

# Install torch_scatter (version must match your python version)
# Visit https://data.pyg.org/whl/torch-1.12.1+cu113.html for the correct version
pip install torch_scatter-2.1.0+pt112cu113-cp38-cp38-linux_x86_64.whl

# Install Detectron2
python -m pip install 'git+https://github.com/facebookresearch/detectron2.git'
```

### 3. Install Foundation Models

```bash
# Install GroundingDINO + SAM (for 2D mask extraction)
cd segmenter2d/
cd GroundingDINO/
pip install -e .
cd ../segment_anything
pip install -e .
cd ../../
```

### 4. Install Additional Dependencies

```bash
pip install scikit-image opencv-python open3d imageio plyfile
pip install -r requirements.txt
```

### 5. Download Pretrained Models

```bash
mkdir -p pretrains/foundation_models
cd pretrains/foundation_models

# Grounding DINO and Segment Anything models
wget https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth
wget https://github.com/IDEA-Research/GroundingDINO/releases/download/v0.1.0-alpha/groundingdino_swint_ogc.pth
```

## 📁 Data Preparation

SGS-3D focuses on **ScanNet200** dataset. We also provide support for ScanNet++ and KITTI-360 for reference.

### ScanNet200 (Primary Dataset)

#### Data Structure
```
data/
├── Scannet200/
│   ├── Scannet200_2D/          
│   │   ├── val/
│   │   │   ├── scene0011_00/
│   │   │   │   ├── color/                # RGB images
│   │   │   │   │   ├── 00000.jpg
│   │   │   │   │   └── ...
│   │   │   │   ├── depth/                # Depth images  
│   │   │   │   │   ├── 00000.png
│   │   │   │   │   └── ...
│   │   │   │   ├── pose/                 # Camera poses
│   │   │   │   │   ├── 00000.txt
│   │   │   │   │   └── ...
│   │   │   │   ├── intrinsic.txt         # Camera intrinsics
│   │   │   │   └── intrinsic_depth.txt
│   │   │   └── ...
│   │   ├── train/
│   │   └── test/
│   └── Scannet200_3D/                    # 3D point clouds and annotations
│       ├── val/
│       │   ├── original_ply_files/       # Original PLY files
│       │   │   ├── scene0011_00.ply
│       │   │   └── ...
│       │   ├── groundtruth/              # Ground truth annotations
│       │   │   ├── scene0011_00.pth
│       │   │   └── ...
│       │   ├── superpoints/              # Superpoint segmentation
│       │   │   ├── scene0011_00.pth
│       │   │   └── ...
│       │   └── dc_feat_scannet200/       # Deep features (optional)
│       │       ├── scene0011_00.pth
│       │       └── ...
│       ├── train/
│       └── test/
```

#### Data Preparation Steps

1. **Download ScanNet200 Dataset**
   - Download from [ScanNetV2 official website](http://www.scan-net.org/)
   - Follow ScanNet license requirements

2. **Generate RGB-D Images and Poses**
   - Use [ScanNet SensReader](https://github.com/ScanNet/ScanNet/tree/master/SensReader/python)
   - Extract with 5-frame interval for efficiency

3. **Generate Superpoints**
   - Follow [ScanNet preprocessing](https://github.com/ScanNet/ScanNet/tree/master/Segmentator)
   - Generate superpoint segmentation

4. **Generate 3D Features**
   - Follow [Mask3D](https://github.com/JonasSchult/Mask3D)
   - Generate high-dimension features of points

### ScanNet++ (Reference)

Similar structure to ScanNet200:

```
data/
├── Scannetpp/
│   ├── Scannetpp_2D/
│   │   ├── val/
│   │   │   ├── 0d2ee665be/
│   │   │   │   ├── color/
│   │   │   │   ├── depth/
│   │   │   │   ├── pose/
│   │   │   │   ├── intrinsic/            # Per-frame intrinsics
│   │   │   │   └── ...
│   │   │   └── ...
│   └── Scannetpp_3D/
│       ├── val/
│       │   ├── original_ply_files/
│       │   ├── groundtruth/
│       │   ├── superpoints/
│       │   └── dc_feat_scannetpp/
│       └── ...
```

### KITTI-360 (Reference)

For outdoor scene evaluation:

```
data/
├── Kitti360/
│   ├── Kitti360_2D/
│   │   ├── val/
│   │   │   ├── 0010_sync_0000002756_0000002920/
│   │   │   │   ├── color/              # RGB images
│   │   │   │   │   ├── 0000002756.png
│   │   │   │   │   ├── 0000002759.png
│   │   │   │   │   └── ...
│   │   │   │   ├── depth/              # Depth images
│   │   │   │   │   ├── 0000002756.png
│   │   │   │   │   ├── 0000002759.png
│   │   │   │   │   └── ...
│   │   │   │   ├── pose/               # Camera poses
│   │   │   │   │   ├── 0000002756.txt
│   │   │   │   │   ├── 0000002759.txt
│   │   │   │   │   └── ...
│   │   │   │   └── intrinsic.txt       # Camera intrinsics
│   │   │   └── ...
│   │   ├── train/
│   │   └── test/
│   └── Kitti360_3D/                    # 3D point clouds and annotations
│       ├── val/
│       │   ├── original_ply_files/     # Original PLY files
│       │   │   ├── 0010_sync_0000002756_0000002920.ply
│       │   │   └── ...
│       │   ├── groundtruth/            # Ground truth annotations
│       │   │   ├── 0010_sync_0000002756_0000002920.pth
│       │   │   └── ...
│       │   ├── superpoints/            # Superpoint segmentation
│       │   │   ├── 0010_sync_0000002756_0000002920.pth
│       │   │   └── ...
│       │   └── dc_feat_kitti360/     # Deep features
│       │       ├── 0010_sync_0000002756_0000002920.pth
│       │       └── ...
│       ├── train/
│       └── test/
```

## 🚀 Quick Start

### Option 1: Evaluate with Provided Results (Recommended)

We provide complete SGS-3D results for all 312 ScanNet200 validation scenes for anonymous review.

#### Download Required Files

**For Reviewers:** Please download the following files to reproduce our evaluation results:

1. **Ground Truth Data** (Required for evaluation)
   - Download: [groundtruth.zip](https://gofile.io/d/uac9DT)
   - Extract to: `data/scannet200/val/groundtruth/`
   - Contains: 312 `.pth` files with ground truth annotations

2. **SGS-3D Results** (Our method's predictions)
   - Download: [scannet200_312_scenes.zip](https://gofile.io/d/xhX1eX)
   - Extract to: `results/scannet200_312_scenes/`
   - Contains: 312 `.pth` files with our predictions

#### Setup Instructions

```bash
# 1. Create directories
mkdir -p data/scannet200/val/groundtruth
mkdir -p results/scannet200_312_scenes

# 2. Download and extract ground truth
# Download groundtruth.zip from https://gofile.io/d/uac9DT
unzip groundtruth.zip -d data/scannet200/val/groundtruth/

# 3. Download and extract results
# Download scannet200_312_scenes.zip from https://gofile.io/d/xhX1eX
unzip scannet200_312_scenes.zip -d results/

# 4. Run evaluation
python tools/eval_classagnostic.py \
    --config configs/scannet200_sgs3d.yaml \
    --type 2D
```

### Option 2: Run Complete Pipeline

If you have ScanNet200 data prepared:

```bash
# Run full pipeline
./run_full_pipeline.sh configs/scannet200_sgs3d.yaml

# Or step by step:
sh scripts/extract_2d_masks.sh configs/scannet200_sgs3d.yaml
sh scripts/generate_3d_inst.sh configs/scannet200_sgs3d.yaml
sh scripts/eval_classagnostic.sh configs/scannet200_sgs3d.yaml
```

### Option 3: Demo Mode

```bash
python demo.py --mode full    # Complete pipeline demo
python demo.py --mode 2d      # 2D mask extraction demo
python demo.py --mode 3d      # 3D instance generation demo
python demo.py --mode eval    # Evaluation demo
```

### Anonymous Review Notes

**What's Included:**
- ✅ Complete working pipeline with simplified algorithms
- ✅ Full evaluation framework with class-agnostic metrics  
- ✅ Complete results for all 312 ScanNet200 scenes
- ✅ Reproducible evaluation using provided results

**What's Simplified:**
- 🔒 Core clustering algorithms (simplified for review)
- 🔒 Advanced filtering techniques (basic version provided)
- 🔒 Optimization strategies (performance optimizations hidden)

**Full Version Promise:**
Upon paper acceptance, we will release:
- Complete implementation with all optimizations
- Detailed experimental code and configurations
- Pre-trained models and processed datasets
- Comprehensive documentation and tutorials

## 📊 File Structure

```
SGS-3D/
├── README.md                           # This comprehensive guide
├── configs/scannet200_sgs3d.yaml      # Main configuration
├── results/scannet200_312_scenes/     # Complete results (312 scenes)
├── tools/
│   ├── extract_2d_masks.py           # 2D mask extraction
│   ├── generate_3d_inst.py           # 3D instance generation
│   └── eval_classagnostic.py         # Evaluation script
├── scripts/                           # Shell scripts for each step
├── src/clustering/                    # Clustering algorithms (simplified)
├── util2d/                           # 2D processing utilities
├── evaluation/                       # Evaluation framework
├── dataset/                          # Dataset loaders (ScanNet200 focus)
├── demo.py                           # Demo and testing
```

## 📖 Usage Examples

### Quick Evaluation
```bash
# Evaluate with provided results
python tools/eval_classagnostic.py \
    --config configs/scannet200_sgs3d.yaml \
    --type 2D
```

### Full Pipeline
```bash
# Complete pipeline
./run_full_pipeline.sh configs/scannet200_sgs3d.yaml
```

### Step-by-Step
```bash
# Step 1: Extract 2D masks
sh scripts/extract_2d_masks.sh configs/scannet200_sgs3d.yaml

# Step 2: Generate 3D instances
sh scripts/generate_3d_inst.sh configs/scannet200_sgs3d.yaml

# Step 3: Evaluate results
sh scripts/eval_classagnostic.sh configs/scannet200_sgs3d.yaml
```

### Anonymous Review Data

**Important for Reviewers:**

- All data is provided via anonymous file sharing links


## 🙏 Acknowledgments

Our implementation is mainly based on the following repositories. Thanks to their authors:

- [PyTorch3D](https://github.com/facebookresearch/pytorch3d)

- [Open3DIS](https://github.com/VinAIResearch/Open3DIS)



