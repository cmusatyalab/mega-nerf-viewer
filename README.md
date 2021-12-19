# Mega-NeRF Dynamic Viewer

This repository contains the code needed to visualize trained Mega-NeRF models.

The codebase for training Mega-NeRF models and extract sparse voxel octrees can be found [here](https://github.com/cmusatyalab/mega-nerf).

**Note:** This is a preliminary release and there may still be outstanding bugs.

## Setup

```
mkdir build && cd build
cmake ..
make -j12
```
You will need a recent version of cmake, LibTorch, and glfw3. The full list of dependencies can be found in the [CMakeLists](CMakeLists.txt#L19) file.

The codebase has been mainly tested against CUDA >= 11.1 and 32GB V100 GPUs.

## Usage

```
./mega-nerf-viewer $OCTREE_PATH --model_path $MODEL_PATH
```

The ```M``` key toggles dynamic octree refinement. The ```R``` key toggles guided ray sampling.

## Acknowledgements

Large parts of this codebase are based on the Plenoctree [renderer](https://github.com/sxyu/volrend).
