#!/bin/bash -l

# As of May 2022
# This script will install TensorFlow, PyTorch, and Horovod on Polaris, all from source
# 1 - Login to Polaris login-node
# 2 - Run './<this script> /path/to/install/base/'
# 3 - script installs everything down in /path/to/install/base/
# 4 - wait for it to complete

# KGF: check HARDCODE points for lines that potentially require manual edits to pinned package versions

BASE_PATH=$1
DATE_PATH="$(basename $BASE_PATH)"

export PYTHONNOUSERSITE=1
umask 0022

# move primary conda packages directory/cache away from ~/.conda/pkgs (4.2 GB currently)
# hardlinks should be preserved even if these files are moved (not across filesystem boundaries)
export CONDA_PKGS_DIRS=/soft/applications/conda/pkgs

#########################################################
# Check for outside communication
# (be sure not to inherit these vars from dotfiles)
#########################################################
unset https_proxy
unset http_proxy

wget -q --spider -T 10 http://google.com
if [ $? -eq 0 ]; then
    echo "Network Online"
else
    echo "Network Offline, setting proxy envs"
    export https_proxy=http://proxy.alcf.anl.gov:3128
    export http_proxy=http://proxy.alcf.anl.gov:3128
fi

module list
module load PrgEnv-nvidia  # not actually using NVHPC compilers to build TF
#module load PrgEnv-gnu
module load gcc-native-mixed/14.2 # get 14.2.0 (Aug 2024) instead of /usr/bin/gcc 7.5 (2019)
module load cuda
module load craype-accel-nvidia80  # wont load for PrgEnv-gnu; see HPE Case 5367752190
module unload cuda
module load craype-x86-milan
export MPICH_GPU_SUPPORT_ENABLED=1
module unload xalt
module list
echo $MPICH_DIR

# -------------------- begin HARDCODE of major built-from-source frameworks etc.
# unset *_TAG variables to build latest master/main branch (or "develop" in the case of DeepHyper)
#DH_REPO_TAG="0.4.2"
DH_REPO_URL=https://github.com/deephyper/deephyper.git

# KGF: build master for CUDA 12.9.1 compatibility
#TF_REPO_TAG="v2.20.0"  # 2025-08-13
# try CUDA 12.8.1 with hermetic build of TF 2.20.1
TF_REPO_TAG=""
#TF_REPO_TAG="v2.17.1"   # 2024-10-24
PT_REPO_TAG="v2.8.0"
#HOROVOD_REPO_TAG="v0.28.1"
HOROVOD_REPO_TAG=""
TF_REPO_URL=https://github.com/tensorflow/tensorflow.git
HOROVOD_REPO_URL=https://github.com/uber/horovod.git
PT_REPO_URL=https://github.com/pytorch/pytorch.git

############################
# Manual version checks/changes below that must be made compatible with TF/Torch/CUDA versions above:
# - pytorch vision
# - magma-cuda
# - tensorflow_probability
# - torch-geometric, torch-sparse, torch-scatter, pyg-lib
# - cupy
# - jax
###########################

#################################################
# CUDA path and version information
#################################################

CUDA_VERSION_MAJOR=12
CUDA_VERSION_MINOR=9
CUDA_VERSION_MINI=1

CUDA_VERSION=$CUDA_VERSION_MAJOR.$CUDA_VERSION_MINOR
CUDA_VERSION_FULL=$CUDA_VERSION.$CUDA_VERSION_MINI
# KGF: this might be a problem, does not match syntax of NCCL_VERSION, etc.

CUDA_TOOLKIT_BASE=/soft/compilers/cudatoolkit/cuda-${CUDA_VERSION_FULL}
CUDA_HOME=${CUDA_TOOLKIT_BASE}

CUDA_DEPS_BASE=/soft/libraries/

CUDNN_VERSION_MAJOR=9
CUDNN_VERSION_MINOR=13.0
CUDNN_VERSION_EXTRA=50
CUDNN_VERSION=$CUDNN_VERSION_MAJOR.$CUDNN_VERSION_MINOR.$CUDNN_VERSION_EXTRA

# HARDCODE: manually renaming default cuDNN tarball name to fit this schema:
CUDNN_BASE=$CUDA_DEPS_BASE/cudnn/cudnn-cuda$CUDA_VERSION_MAJOR-linux-x64-v$CUDNN_VERSION

NCCL_VERSION_MAJOR=2
NCCL_VERSION_MINOR=28.3-1
NCCL_VERSION=$NCCL_VERSION_MAJOR.$NCCL_VERSION_MINOR
NCCL_BASE=$CUDA_DEPS_BASE/nccl/nccl_$NCCL_VERSION+cuda${CUDA_VERSION}_x86_64

TENSORRT_VERSION_MAJOR=10
TENSORRT_VERSION_MINOR=13.3.9
# TENSORRT_VERSION_MAJOR=8
# TENSORRT_VERSION_MINOR=6.1.6
TENSORRT_VERSION=$TENSORRT_VERSION_MAJOR.$TENSORRT_VERSION_MINOR
# HARDCODE
TENSORRT_BASE=$CUDA_DEPS_BASE/trt/TensorRT-$TENSORRT_VERSION.Linux.x86_64-gnu.cuda-12.9
echo "TENSORRT_BASE=${TENSORRT_BASE}"

#################################################
# TensorFlow Config flags (for ./configure run)
#################################################
export TF_CUDA_COMPUTE_CAPABILITIES=8.0
# Note that TF_CUDA_VERSION and TF_CUDNN_VERSION should consist of major and minor versions only (e.g. 12.3 for CUDA and 9.1 for CUDNN).
# https://openxla.org/xla/hermetic_cuda
export TF_CUDA_VERSION=${CUDA_VERSION_MAJOR}.${CUDA_VERSION_MINOR}
export TF_CUDNN_VERSION=${CUDNN_VERSION_MAJOR}.${CUDNN_VERSION_MINOR}
export TF_TENSORRT_VERSION=$TENSORRT_VERSION_MAJOR
export TF_NCCL_VERSION=${NCCL_VERSION_MAJOR}.${NCCL_VERSION_MINOR}
# KGF: double check above changes to syntax
export CUDA_TOOLKIT_PATH=$CUDA_TOOLKIT_BASE
export CUDNN_INSTALL_PATH=$CUDNN_BASE
export NCCL_INSTALL_PATH=$NCCL_BASE
export TENSORRT_INSTALL_PATH=$TENSORRT_BASE
export TF_NEED_OPENCL_SYCL=0
export TF_NEED_COMPUTECPP=0
export TF_CUDA_CLANG=0   # KGF?
export TF_NEED_OPENCL=0
export TF_NEED_MPI=0
export TF_NEED_ROCM=0
export TF_NEED_CUDA=1
export TF_NEED_TENSORRT=0
export TF_CUDA_PATHS=$CUDA_TOOLKIT_BASE,$CUDNN_BASE,$NCCL_BASE
# KGF: last env var lets 12.9.0 leak in when 12.9.1 is specified, etc.
#export TF_CUDA_PATHS=$CUDA_TOOLKIT_BASE,$CUDNN_BASE,$NCCL_BASE,$TENSORRT_BASE
#export GCC_HOST_COMPILER_PATH=$(which gcc)

# HARDCODE
export TF_PYTHON_VERSION=3.12
# KGF: check this
export GCC_HOST_COMPILER_PATH=/usr/bin/gcc-14
#/opt/cray/pe/gcc/12.2.0/snos/bin/gcc
export CC_OPT_FLAGS="-march=native -Wno-sign-compare"
export TF_SET_ANDROID_WORKSPACE=0

#################################################
## Installing Miniconda
#################################################

# set Conda installation folder and where downloaded content will stay
CONDA_PREFIX_PATH=$BASE_PATH/mconda3
DOWNLOAD_PATH=$BASE_PATH/DOWNLOADS
WHEELS_PATH=$BASE_PATH/wheels

mkdir -p $CONDA_PREFIX_PATH
mkdir -p $DOWNLOAD_PATH
mkdir -p $WHEELS_PATH
cd $BASE_PATH
# HARDCODE
# Download and install conda for a base python installation
# CONDAVER='py312_25.7.0-2'
# CONDA_DOWNLOAD_URL=https://repo.continuum.io/miniconda
# CONDA_INSTALL_SH=Miniconda3-$CONDAVER-Linux-x86_64.sh
#echo "Downloading miniconda installer"
echo "Downloading miniforge installer"
CONDA_DOWNLOAD_URL="https://github.com/conda-forge/miniforge/releases/latest/download"
CONDA_INSTALL_SH="Miniforge3-$(uname)-$(uname -m).sh"
wget $CONDA_DOWNLOAD_URL/$CONDA_INSTALL_SH -P $DOWNLOAD_PATH
chmod +x $DOWNLOAD_PATH/$CONDA_INSTALL_SH

echo "Installing Miniconda"
echo "bash $DOWNLOAD_PATH/$CONDA_INSTALL_SH -b -p $CONDA_PREFIX_PATH -u"
bash $DOWNLOAD_PATH/$CONDA_INSTALL_SH -b -p $CONDA_PREFIX_PATH -u

cd $CONDA_PREFIX_PATH

#########
# create a setup file
cat > setup.sh << EOF
preferred_shell=\$(basename \$SHELL)

module load PrgEnv-gnu

if [ -n "\$ZSH_EVAL_CONTEXT" ]; then
    DIR=\$( cd "\$( dirname "\$0" )" && pwd )
else  # bash, sh, etc.
    DIR=\$( cd "\$( dirname "\${BASH_SOURCE[0]}" )" && pwd )
fi

eval "\$(\$DIR/bin/conda shell.\${preferred_shell} hook)"

# test network
unset https_proxy
unset http_proxy
wget -q --spider -T 10 http://google.com
if [ \$? -eq 0 ]; then
    echo "Network Online"
else
   echo "Network Offline, setting proxy envs"
   export https_proxy=http://proxy.alcf.anl.gov:3128
   export http_proxy=http://proxy.alcf.anl.gov:3128
fi

export CUDA_TOOLKIT_BASE=$CUDA_TOOLKIT_BASE
export CUDNN_BASE=$CUDNN_BASE
export NCCL_BASE=$NCCL_BASE
export TENSORRT_BASE=$TENSORRT_BASE
export LD_LIBRARY_PATH=\$CUDA_TOOLKIT_BASE/lib64:\$CUDNN_BASE/lib:\$NCCL_BASE/lib:\$TENSORRT_BASE/lib:\$LD_LIBRARY_PATH:
export PATH=\$CUDA_TOOLKIT_BASE/bin:\$PATH
EOF

PYTHON_VER=$(ls -d lib/python?.?? | grep -oP '(?<=python)\d+\.\d+')
echo PYTHON_VER=$PYTHON_VER

cat > .condarc << EOF
channels:
   - conda-forge
   - pytorch
env_prompt: "(${DATE_PATH}/{default_env}) "
pkgs_dirs:
   - ${CONDA_PKGS_DIRS}
   - \$HOME/.conda/pkgs
EOF

# move to base install directory
cd $BASE_PATH
echo "cd $BASE_PATH"

# setup conda environment
source $CONDA_PREFIX_PATH/setup.sh
echo "after sourcing conda"
module unload xalt

echo "CONDA BINARY: $(which conda)"
echo "CONDA VERSION: $(conda --version)"
echo "PYTHON VERSION: $(python --version)"

set -e

################################################
### Install TensorFlow
################################################

echo "Conda install some dependencies"
conda install -y -n base conda-libmamba-solver
conda config --set solver libmamba

conda install -y -c conda-forge cmake zip unzip astunparse setuptools future six requests dataclasses graphviz numba numpy pymongo conda-build pip libaio
conda install -y -c conda-forge mkl mkl-include  # onednn mkl-dnn git-lfs ### on ThetaGPU

# CUDA only: Add LAPACK support for the GPU if needed
# HARDCODE
conda install -y -c pytorch -c conda-forge magma-cuda126 # ${CUDA_VERSION_MAJOR}${CUDA_VERSION_MINOR}
conda install -y -c conda-forge mamba ccache

echo "Clone TensorFlow"
cd $BASE_PATH
git clone $TF_REPO_URL
cd tensorflow

if [[ -z "$TF_REPO_TAG" ]]; then
    echo "Checkout TensorFlow master"
else
    echo "Checkout TensorFlow tag $TF_REPO_TAG"
    git checkout --recurse-submodules $TF_REPO_TAG
fi
BAZEL_VERSION=$(head -n1 .bazelversion)
echo "Found TensorFlow depends on Bazel version $BAZEL_VERSION"

cd $BASE_PATH
echo "Download Bazel binaries"
BAZEL_DOWNLOAD_URL=https://github.com/bazelbuild/bazel/releases/download/$BAZEL_VERSION
BAZEL_INSTALL_SH=bazel-$BAZEL_VERSION-installer-linux-x86_64.sh
BAZEL_INSTALL_PATH=$BASE_PATH/bazel-$BAZEL_VERSION
echo "wget $BAZEL_DOWNLOAD_URL/$BAZEL_INSTALL_SH -P $DOWNLOAD_PATH"
wget $BAZEL_DOWNLOAD_URL/$BAZEL_INSTALL_SH -P $DOWNLOAD_PATH
chmod +x $DOWNLOAD_PATH/$BAZEL_INSTALL_SH
echo "Install Bazel in $BAZEL_INSTALL_PATH"
bash $DOWNLOAD_PATH/$BAZEL_INSTALL_SH --prefix=$BAZEL_INSTALL_PATH
export PATH=$PATH:/$BAZEL_INSTALL_PATH/bin

cd $BASE_PATH

echo "Install TensorFlow Dependencies"
pip install -U numpy numba ninja
pip install -U pip wheel mock gast portpicker pydot packaging pyyaml
pip install -U keras_applications --no-deps
pip install -U keras_preprocessing --no-deps

echo "Configure TensorFlow"
cd tensorflow
export PYTHON_BIN_PATH=$(which python)
export PYTHON_LIB_PATH=$(python -c 'import site; print(site.getsitepackages()[0])')
export TMP=/tmp
./configure

echo "Bazel Build TensorFlow"

# HARDCODE
module use /soft/modulefiles
module load llvm/release-18.1.6  # llvm/release-19.1.7
export CC=/soft/compilers/llvm/release-18.1.6/bin/clang  # TF 2.20.0 tested with Clang 18.1.8
export BAZEL_COMPILER=$CC

# 2.17:
#HOME=$DOWNLOAD_PATH bazel build --jobs=500 --local_cpu_resources=32 --verbose_failures --config=cuda --cxxopt="-D_GLIBCXX_USE_CXX11_ABI=0" //tensorflow/tools/pip_package:wheel
# 2.20:
# export LOCAL_CUDA_PATH=$CUDA_TOOLKIT_BASE
# export LOCAL_CUDNN_PATH=$CUDNN_BASE
# export LOCAL_NCCL_PATH=$NCCL_BASE
# export LOCAL_NVSHMEM_PATH="/soft/libraries/nvshmem/libnvshmem-linux-x86_64-3.3.9_cuda12-archive/"

# # make sure the CUPTI .so’s are findable at link/run time
# export LD_LIBRARY_PATH="$CUDA_TOOLKIT_BASE/extras/CUPTI/lib64:$CUDA_TOOLKIT_BASE/targets/x86_64-linux/lib:$CUDA_TOOLKIT_BASE/lib64:${LD_LIBRARY_PATH:-}"

export HERMETIC_CUDA_VERSION=$CUDA_VERSION_FULL
export HERMETIC_CUDNN_VERSION=$CUDNN_VERSION_MAJOR.$CUDNN_VERSION_MINOR
# $CUDNN_VERSION  # 9.13.0.50.
export HERMETIC_NCCL_VERSION=$NCCL_VERSION
export HERMETIC_NVSHMEM_VERSION="3.3.9"
export HERMETIC_CUDA_COMPUTE_CAPABILITIES="sm_80"

# ERROR: Error computing the main repository mapping: no such package '@@standalone_cuda_redist_json//': The support
# ed CUDA versions are ["11.8", "12.1.1", "12.2.0", "12.3.1", "12.3.2", "12.4.0", "12.4.1", "12.5.0", "12.5.1", "12.
# 6.0", "12.6.1", "12.6.2", "12.6.3", "12.8.0", "12.8.1"]. Please provide a supported version in ["HERMETIC_CUDA_VER
# SION", "TF_CUDA_VERSION"] environment variable(s) or add JSON URL for CUDA version=12.9.0.


# force repo reconfig once when flipping modes
#bazel sync --configure

# echo "HOME=$DOWNLOAD_PATH bazel build --announce_rc --jobs=32 --loading_phase_threads=6 --verbose_failures --config=cuda --config=cuda_nvcc --config=cuda_wheel --@local_config_cuda//cuda:override_include_cuda_libs=false \
#     --repo_env=TF_CUDA_COMPUTE_CAPABILITIES=${TF_CUDA_COMPUTE_CAPABILITIES} \
#     --repo_env=TF_CUDA_VERSION=${TF_CUDA_VERSION} \
#     --repo_env=TF_CUDNN_VERSION=${TF_CUDNN_VERSION} \
#     --repo_env=TF_TENSORRT_VERSION=${TF_TENSORRT_VERSION} \
#     --repo_env=TF_NCCL_VERSION=${TF_NCCL_VERSION} \
#     --repo_env=CUDA_TOOLKIT_PATH=${CUDA_TOOLKIT_PATH} \
#     --repo_env=CUDNN_INSTALL_PATH=${CUDNN_INSTALL_PATH} \
#     --repo_env=NCCL_INSTALL_PATH=${NCCL_INSTALL_PATH} \
#     --repo_env=TF_CUDA_PATHS=${TF_CUDA_PATHS} \
#     --repo_env=HERMETIC_CUDA_VERSION= \
#     --repo_env=HERMETIC_CUDNN_VERSION= \
#     --repo_env=HERMETIC_NCCL_VERSION= \
#     --repo_env=HERMETIC_NVSHMEM_VERSION= \
#     --repo_env=HERMETIC_CUDA_COMPUTE_CAPABILITIES= \
#     --repo_env=CC=${CC} \
#     --action_env=CC=${CC} \
#     --action_env=LD_LIBRARY_PATH=${LD_LIBRARY_PATH} \
#     --copt=-Wno-error=unused-command-line-argument --cxxopt=-D_GLIBCXX_USE_CXX11_ABI=0 //tensorflow/tools/pip_package:wheel
# "

HOME=$DOWNLOAD_PATH bazel build --announce_rc --jobs=128 --loading_phase_threads=6 --verbose_failures --config=cuda --config=cuda_wheel --@local_config_cuda//cuda:override_include_cuda_libs=false \
    --repo_env=HERMETIC_CUDA_VERSION=${HERMETIC_CUDA_VERSION} \
    --repo_env=HERMETIC_CUDNN_VERSION=${HERMETIC_CUDNN_VERSION} \
    --repo_env=HERMETIC_NCCL_VERSION=${HERMETIC_NCCL_VERSION} \
    --repo_env=HERMETIC_NVSHMEM_VERSION=${HERMETIC_NVSHMEM_VERSION} \
    --repo_env=HERMETIC_CUDA_COMPUTE_CAPABILITIES=${HERMETIC_CUDA_COMPUTE_CAPABILITIES} \
    --copt="-Wno-error=unused-command-line-argument" --cxxopt="-D_GLIBCXX_USE_CXX11_ABI=0" //tensorflow/tools/pip_package:wheel

# NOTE: --config=cuda_nvcc still requires --config=cuda, maybe?

# --config=cuda --@local_config_cuda//cuda:override_include_cuda_libs=true

# clang-18: error: unknown argument: '-Xcuda-fatbinary=--compress-all'
# clang-18: error: unknown argument: '-nvcc_options=expt-relaxed-constexpr'
# clang-18: warning: CUDA version is newer than the latest supported version 12.3 [-Wunknown-cuda-version]

# --config=cuda_wheel --@local_config_cuda//:cuda_compiler=clang
# --config=cuda_wheel
# --@local_config_cuda//:cuda_compiler=clang
    # --copt=-D_GLIBCXX_USE_FLOAT128=0 \
    # --cxxopt=-D_GLIBCXX_USE_FLOAT128=0 \

    # --repo_env=HERMETIC_CUDA_VERSION=${HERMETIC_CUDA_VERSION} \
    # --repo_env=HERMETIC_CUDNN_VERSION=${HERMETIC_CUDNN_VERSION} \
    # --repo_env=HERMETIC_NCCL_VERSION=${HERMETIC_NCCL_VERSION} \
    # --repo_env=HERMETIC_NVSHMEM_VERSION=${HERMETIC_NVSHMEM_VERSION} \
    # --repo_env=HERMETIC_CUDA_COMPUTE_CAPABILITIES=${HERMETIC_CUDA_COMPUTE_CAPABILITIES} \


    # --repo_env=HERMETIC_CUDA_VERSION= \
    # --repo_env=HERMETIC_CUDNN_VERSION= \
    # --repo_env=HERMETIC_NCCL_VERSION= \
    # --repo_env=HERMETIC_NVSHMEM_VERSION= \
    # --repo_env=HERMETIC_CUDA_COMPUTE_CAPABILITIES= \


    # --repo_env=TF_CUDA_COMPUTE_CAPABILITIES=${TF_CUDA_COMPUTE_CAPABILITIES} \
    # --repo_env=TF_CUDA_VERSION=${TF_CUDA_VERSION} \
    # --repo_env=TF_CUDNN_VERSION=${TF_CUDNN_VERSION} \
    # --repo_env=TF_TENSORRT_VERSION=${TF_TENSORRT_VERSION} \
    # --repo_env=TF_NCCL_VERSION=${TF_NCCL_VERSION} \
    # --repo_env=CUDA_TOOLKIT_PATH=${CUDA_TOOLKIT_PATH} \
    # --repo_env=CUDNN_INSTALL_PATH=${CUDNN_INSTALL_PATH} \
    # --repo_env=NCCL_INSTALL_PATH=${NCCL_INSTALL_PATH} \
    # --repo_env=TF_CUDA_PATHS=${TF_CUDA_PATHS} \
    # --repo_env=CC=${CC} \
    # --action_env=CC=${CC} \
    # --action_env=LD_LIBRARY_PATH=${LD_LIBRARY_PATH} \


# https://github.com/openxla/xla/blob/main/docs/hermetic_cuda.md
# --config=cuda  # Primary approach. Successful Sophia build just used this, but if I use it in this build on Polaris, it throws an error (need to add another flag)
# --config=cuda_wheel  # previous polaris build used both flags?

# TF_CUDA_CLANG=0, --config=cuda  --@local_config_cuda//cuda:override_include_cuda_libs=true --copt="-Wno-error=unused-command-line-argument"
# ------------------------
# got very far: [12,120 / 34,809]
# clang-18: warning: argument unused during compilation: '--cuda-path=external/cuda_nvcc' [-Wunused-command-line-argument]
# In file included from external/local_xla/xla/backends/profiler/gpu/cupti_wrapper.cc:16:
# external/local_xla/xla/backends/profiler/gpu/cupti_wrapper.h:22:10: fatal error: 'third_party/gpus/cuda/extras/CUPTI/include/cupti.h' file not found
#    22 | #include "third_party/gpus/cuda/extras/CUPTI/include/cupti.h"
#       |          ^~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# 1 error generated.

# See discussion here: https://github.com/jax-ml/jax/issues/23689#issuecomment-2563911216
# and here: https://github.com/google-ml-infra/rules_ml_toolchain/tree/main/gpu
# TF expects cuda-12.9.0/lib/libcupti...
# Solution: modify LD_LIBRARY_PATH and add   --copt=-I"$CUDA_TOOLKIT_BASE/extras/CUPTI/include" \
    #
# The include path '/soft/compilers/cudatoolkit/cuda-12.9.0/extras/CUPTI/include' references a path outside of the execution root.
# After removing that flag,
# clang-18: error: GPU arch sm_35 is supported by CUDA versions between 7.0 and 11.8 (inclusive), but installation at external/cuda_nvcc is ; use '--cuda-path' to specify a different CUDA install, pass a different GPU arch with '--cuda-gpu-arch', or pass '--no-cuda-version-check'
# Now trying --@local_config_cuda//:cuda_compiler=clang
# and re-add --config=cuda_wheel

# The cuda_compiler=clang option was great--- fixed all the "no such compiler option" warnings

# --@local_config_cuda//:cuda_compiler=clang_local
# Error in fail: Error setting @local_config_cuda//:cuda_compiler: invalid value 'clang_local'. Allowed values are ["clang", "nvcc"]


# TF_CUDA_CLANG=0, --config=cuda
# ------------------------
# Error in fail: TF wheel shouldn't be built with CUDA dependencies. Please provide `--config=cuda_wheel` for bazel build command. If you absolutely need to add CUDA dependencies, provide `--@local_config_cuda//cuda:override_include_cuda_libs=true`

# TF_CUDA_CLANG=1, --config=cuda_wheel
# ------------------------
# Please add max PTX version supported by Clang major version=7.

# TF_CUDA_CLANG=0, --config=cuda_wheel
# ------------------------
# clang-18: error: argument unused during compilation: '--cuda-path=external/cuda_nvcc' [-Werror,-Wunused-command-line-argument]

# (now trying to add --copt=-Wno-error=unused-command-line-argument)
# clang-18: error: unknown argument: '-Xcuda-fatbinary=--compress-all'
# clang-18: error: unknown argument: '-nvcc_options=expt-relaxed-constexpr'
# clang-18: error: GPU arch sm_35 is supported by CUDA versions between 7.0 and 11.8 (inclusive), but installation at external/cuda_nvcc is ; use '--cuda-path' to specify a different CUDA install, pass a different GPU arch with '--cuda-gpu-arch', or pass '--no-cuda-version-check'

#--local_resources=cpus=32

#HOME=$DOWNLOAD_PATH bazel build --jobs=500 --local_resources=cpus=32 --verbose_failures --config=cuda --config=cuda_wheel --@local_config_cuda//cuda:override_include_cuda_libs=true --cxxopt="-D_GLIBCXX_USE_CXX11_ABI=0" //tensorflow/tools/pip_package:wheel
echo "Run wheel building"
cp ./bazel-bin/tensorflow/tools/pip_package/wheel_house/*.whl $WHEELS_PATH
echo "Install TensorFlow"
pip install $(find $WHEELS_PATH/ -name "tensorflow*.whl" -type f)

# 2.17 and later:
unset CC

#################################################
### Install PyTorch
#################################################

cd $BASE_PATH
echo "Clone PyTorch"

git clone $PT_REPO_URL
cd pytorch
if [[ -z "$PT_REPO_TAG" ]]; then
    echo "Checkout PyTorch master"
else
    echo "Checkout PyTorch tag $PT_REPO_TAG"
    git checkout --recurse-submodules $PT_REPO_TAG
    echo "git submodule sync"
    git submodule sync
    echo "git submodule update"
    git submodule update --init --recursive
fi
# HARDCODE
export CUDNN_INCLUDE_DIR=$CUDNN_BASE/include
export CPATH="$CPATH:$CUDNN_INCLUDE_DIR"

echo "Install PyTorch"
#module unload gcc-mixed
module load PrgEnv-gnu
export CRAY_ACCEL_TARGET="nvidia80"
export CRAY_TCMALLOC_MEMFS_FORCE="1"
export CRAYPE_LINK_TYPE="dynamic"
export CRAY_ACCEL_VENDOR="nvidia"
export CRAY_CPU_TARGET="x86-64"

module unload xalt
module list
echo "CRAY_ACCEL_TARGET= $CRAY_ACCEL_TARGET"
echo "CRAYPE_LINK_TYPE = $CRAYPE_LINK_TYPE"

export USE_CUDA=1
export USE_CUDNN=1
export TORCH_CUDA_ARCH_LIST=8.0
echo "CUDNN_ROOT=$CUDNN_BASE"
export CUDNN_ROOT_DIR=$CUDNN_BASE
export CUDNN_INCLUDE_DIR=$CUDNN_BASE/include

export USE_SYSTEM_NCCL=1
export NCCL_ROOT=$NCCL_BASE
export NCCL_INCLUDE_DIR=$NCCL_BASE/include
export NCCL_LIB_DIR=$NCCL_BASE/lib

export USE_TENSORRT=ON
export TENSORRT_ROOT=$TENSORRT_BASE
export CMAKE_PREFIX_PATH=${CONDA_PREFIX:-"$(dirname $(which conda))/../"}

export PYTORCH_BUILD_VERSION="${PT_REPO_TAG:1}"
export PYTORCH_BUILD_NUMBER=1

export TENSORRT_INCLUDE_DIR="TENSORRT_BASE/include"
export TENSORRT_LIBRARY="$TENSORRT_BASE/lib/libmyelin.so"

export USE_CUSPARSELT=1
# HARDCODE
export CUSPARSELT_ROOT="/soft/libraries/cusparselt/libcusparse_lt-linux-x86_64-0.8.1.1_cuda12-archive/"
export CUSPARSELT_INCLUDE_PATH="${CUSPARSELT_ROOT}include"
# -------------

echo "TENSORRT_INCLUDE_DIR=${TENSORRT_INCLUDE_DIR}"
echo "TENSORRT_LIBRARY=${TENSORRT_LIBRARY}"
echo "CUSPARSELT_ROOT=${CUSPARSELT_ROOT}"
echo "CUSPARSELT_INCLUDE_PATH=${CUSPARSELT_INCLUDE_PATH}"

echo "PYTORCH_BUILD_VERSION=$PYTORCH_BUILD_VERSION and PYTORCH_BUILD_NUMBER=$PYTORCH_BUILD_NUMBER"
#echo "CC=/opt/cray/pe/gcc/12.2.0/snos/bin/gcc CXX=/opt/cray/pe/gcc/12.2.0/snos/bin/g++ python setup.py bdist_wheel"
echo "BUILD_TEST=0 CUDAHOSTCXX=g++-14 CC=/usr/bin/gcc-14 CXX=/usr/bin/g++-14 python setup.py bdist_wheel"
#echo "CC=$(which cc) CXX=$(which CC) python setup.py bdist_wheel"
#CC=$(which cc) CXX=$(which CC) python setup.py bdist_wheel

# HARDCODE
BUILD_TEST=0 CUDAHOSTCXX=g++-14 CC=/usr/bin/gcc-14 CXX=/usr/bin/g++-14 python setup.py bdist_wheel
PT_WHEEL=$(find dist/ -name "torch*.whl" -type f)
echo "copying pytorch wheel file $PT_WHEEL"
cp $PT_WHEEL $WHEELS_PATH/
cd $WHEELS_PATH
echo "pip installing $(basename $PT_WHEEL)"
pip install $(basename $PT_WHEEL)

# HARDCODE
#pip install torchtriton --extra-index-url "https://download.pytorch.org/whl/nightly/cu124"
# https://pytorch.org/tutorials/intermediate/torch_compile_tutorial.html

################################################
### Install Horovod
#################################################

cd $BASE_PATH

# echo "Clone Horovod"

# git clone --recursive $HOROVOD_REPO_URL
# cd horovod

# if [[ -z "$HOROVOD_REPO_TAG" ]]; then
#     echo "Checkout Horovod master"
# else
#     echo "Checkout Horovod tag $HOROVOD_REPO_TAG"
#     git checkout --recurse-submodules $HOROVOD_REPO_TAG
# fi

# echo "Build Horovod Wheel using MPI from $MPICH_DIR and NCCL from ${NCCL_BASE}"

# # https://github.com/horovod/horovod/issues/3696#issuecomment-1248921736
# echo "CUDAHOSTCXX=g++-14 CC=/usr/bin/gcc-14 CXX=/usr/bin/g++-14 HOROVOD_WITH_MPI=1 HOROVOD_CUDA_HOME=${CUDA_TOOLKIT_BASE} HOROVOD_NCCL_HOME=$NCCL_BASE HOROVOD_CMAKE=$(which cmake) HOROVOD_GPU_OPERATIONS=NCCL HOROVOD_WITH_TENSORFLOW=1 HOROVOD_WITHOUT_PYTORCH=1 HOROVOD_WITHOUT_MXNET=1 python setup.py bdist_wheel"

# # HARDCODE: temp disable Horovod 0.28.1 + PyTorch >=2.1.x integration; C++17 required in PyTorch now (https://github.com/pytorch/pytorch/pull/100557)
# # https://github.com/horovod/horovod/pull/3998
# # https://github.com/horovod/horovod/issues/3996
# #CUDAHOSTCXX=g++-14 CC=/usr/bin/gcc-14 CXX=/usr/bin/g++-14 HOROVOD_WITH_MPI=1 HOROVOD_CUDA_HOME=${CUDA_TOOLKIT_BASE} HOROVOD_NCCL_HOME=$NCCL_BASE HOROVOD_CMAKE=$(which cmake) HOROVOD_GPU_OPERATIONS=NCCL HOROVOD_WITH_TENSORFLOW=1 HOROVOD_WITHOUT_PYTORCH=1 HOROVOD_WITHOUT_MXNET=1 python setup.py bdist_wheel

# # KGF: using CMake 4.1.1, which dropped CMakes older than 3.5
# export CMAKE_ARGS="-DCMAKE_POLICY_VERSION_MINIMUM=3.5"
# # sed -i 's/VERSION 2\.8\.12/VERSION 3.5/' third_party/gloo/CMakeLists.txt

# CUDAHOSTCXX=g++-14 CC=/usr/bin/gcc-14 CXX=/usr/bin/g++-14 HOROVOD_WITH_MPI=1 HOROVOD_CUDA_HOME=${CUDA_TOOLKIT_BASE} HOROVOD_NCCL_HOME=$NCCL_BASE HOROVOD_CMAKE=$(which cmake) HOROVOD_GPU_OPERATIONS=NCCL HOROVOD_WITH_TENSORFLOW=1 HOROVOD_WITH_PYTORCH=1 HOROVOD_WITHOUT_MXNET=1 python setup.py bdist_wheel
# #-DCMAKE_POLICY_VERSION_MINIMUM=3.5

# HVD_WHL=$(find dist/ -name "horovod*.whl" -type f)
# cp $HVD_WHL $WHEELS_PATH/
# HVD_WHEEL=$(find $WHEELS_PATH/ -name "horovod*.whl" -type f)
# echo "Install Horovod $HVD_WHEEL"
# pip install --force-reinstall --no-cache-dir $HVD_WHEEL

echo "Pip install TensorBoard profiler plugin"
pip install tensorboard_plugin_profile tensorflow-datasets

cd $BASE_PATH
# KGF (2022-09-09):
MPICC="cc -shared -target-accel=nvidia80" pip install --force-reinstall --no-cache-dir --no-binary=mpi4py mpi4py

echo "Pip install parallel h5py"
cd $BASE_PATH
git clone https://github.com/h5py/h5py.git
cd h5py
module load cray-hdf5-parallel
export CC=cc
export HDF5_MPI="ON"
pip install .

echo "Pip install other packages"
pip install pandas matplotlib scikit-learn scipy pytest
pip install sacred wandb

echo "Adding module snooper so we can tell what modules people are using"
# KGF: TODO, modify this path at the top of the script somehow; pick correct sitecustomize_polaris.py, etc.
# wont error out if first path does not exist; will just make a broken symbolic link
# https://github.com/argonne-lcf/PyModuleSnooper.git (fork of msalim2)
ln -s /soft/applications/PyModuleSnooper/sitecustomize.py $(python -c 'import site; print(site.getsitepackages()[0])')/sitecustomize.py

# DeepHyper stuff
# HARDCODE
pip install 'tensorflow_probability==0.25.0'
# KGF: 0.25.0 (2024-11-08) tested against TF 2.18 and JAX 0.4.35
# KGF: 0.24.0 (2024-03-12) tested against TF 2.16.1 and JAX 0.4.25

if [[ -z "$DH_REPO_TAG" ]]; then
    echo "Clone and checkout DeepHyper develop branch from git"
    cd $BASE_PATH
    git clone $DH_REPO_URL
    cd deephyper
    git checkout develop
    pip install ".[hps,hps-tl,nas,autodeuq,jax-gpu,automl,mpi,ray,redis-hiredis]"

    cd ..
    cd $BASE_PATH
else
    echo "Build DeepHyper tag $DH_REPO_TAG and Balsam from PyPI"
    pip install "deephyper[analytics,hvd,nas,popt,autodeuq]==${DH_REPO_TAG}"
fi

pip install 'libensemble'

# HARDCODE
pip install pyg_lib torch_scatter torch_sparse torch_cluster torch_spline_conv -f https://data.pyg.org/whl/torch-2.8.0+cu129.html
#pip install torch_spline_conv -f https://data.pyg.org/whl/torch-2.2.2+cu${CUDA_VERSION_MAJOR}1.html
# pytorch 2.8 support:
# https://github.com/rusty1s/pytorch_spline_conv/commit/a80c34c7da96801edea12b29655b93cfa2e51ad5
# build the rest from source:
# CUDAHOSTCXX=g++-14 CC=/usr/bin/gcc-14 CXX=/usr/bin/g++-14 pip install --verbose git+https://github.com/pyg-team/pyg-lib.git
# export CPATH=${CUDA_TOOLKIT_BASE}/include:$CPATH
# CUDAHOSTCXX=g++-14 CC=/usr/bin/gcc-14 CXX=/usr/bin/g++-14 pip install --verbose torch_sparse
# ---------------------------------------
pip install torch-geometric
pip install pillow
#pip install "pillow!=8.3.0,>=6.2.0"

cd $BASE_PATH
echo "Install PyTorch Vision from source"
git clone https://github.com/pytorch/vision.git
cd vision
# HARDCODE
git checkout v0.23.0

# HARDCODE
CUDAHOSTCXX=g++-14 CC=/usr/bin/gcc-14 CXX=/usr/bin/g++-14 python setup.py bdist_wheel
VISION_WHEEL=$(find dist/ -name "torchvision*.whl" -type f)
cp $VISION_WHEEL $WHEELS_PATH/
cd $WHEELS_PATH
echo "pip installing $(basename $VISION_WHEEL)"
pip install --force-reinstall --no-deps $(basename $VISION_WHEEL)

cd $BASE_PATH

pip install --no-deps timm
pip install opencv-python-headless

# HARDCODE
pip install 'onnx==1.19.0' 'onnxruntime-gpu==1.22.2'
pip install tf2onnx
pip install onnx-tf
pip install huggingface-hub
pip install transformers evaluate datasets accelerate
pip install --no-deps xformers
pip install flash-attn --no-build-isolation
pip install scikit-image
pip install ipython
pip install line_profiler
pip install torch-tb-profiler
pip install torchinfo
# HARDCODE
pip install cupy-cuda${CUDA_VERSION_MAJOR}x
pip install pytorch-lightning
pip install ml-collections
pip install gpytorch xgboost multiprocess py4j
# HARDCODE
CUDAHOSTCXX=g++-14 CC=/usr/bin/gcc-14 CXX=/usr/bin/g++-14 pip install --no-build-isolation git+https://github.com/FalkonML/falkon.git
pip install pykeops   # wants nonstandard env var set: CUDA_PATH=$CUDA_HOME
pip install hydra-core hydra_colorlog accelerate arviz pyright celerite seaborn xarray bokeh matplotx aim torchviz rich parse
pip install jupyter
pip install climetlab
pip install tensorboardX

# HARDCODE
pip install triton
cd $BASE_PATH
echo "Install CUTLASS from source"
git clone https://github.com/NVIDIA/cutlass
cd cutlass

export CUTLASS_PATH="${BASE_PATH}/cutlass"
mkdir build && cd build
export CUDNN_PATH=${CUDNN_BASE}
export CUDA_INSTALL_PATH=${CUDA_HOME}
export CUDACXX=${CUDA_INSTALL_PATH}/bin/nvcc
echo "About to run CMake for CUTLASS python = $(which python)"
conda info
CUDAHOSTCXX=g++-14 CC=/usr/bin/gcc-14 CXX=/usr/bin/g++-14 cmake .. -DCUTLASS_NVCC_ARCHS=80 -DCUTLASS_ENABLE_CUBLAS=ON -DCUTLASS_ENABLE_CUDNN=ON
make cutlass_profiler -j32

cd $BASE_PATH
echo "Install DeepSpeed from source"
git clone https://github.com/deepspeedai/DeepSpeed
cd DeepSpeed
# HARDCODE
git checkout v0.17.6
export CFLAGS="-I${CONDA_PREFIX}/include/"
export LDFLAGS="-L${CONDA_PREFIX}/lib/ -Wl,--enable-new-dtags,-rpath,${CONDA_PREFIX}/lib"
pip install deepspeed-kernels

CUDAHOSTCXX=g++-14 CC=/usr/bin/gcc-14 CXX=/usr/bin/g++-14 NVCC_PREPEND_FLAGS="--forward-unknown-opts" DS_BUILD_SPARSE_ATTN=0 DS_BUILD_OPS=1 DS_BUILD_AIO=1 pip install --verbose . ### --global-option="build_ext" --global-option="-j16"
# the parallel build options seem to cause issues

# > ds_report
cd $BASE_PATH

# HARDCODE
# Apex (for Megatron-Deepspeed)
git clone https://github.com/NVIDIA/apex
cd apex
#  with CUDA and C++ extensions using environment variables:
CUDAHOSTCXX=g++-14 CC=/usr/bin/gcc-14 CXX=/usr/bin/g++-14 NVCC_APPEND_FLAGS="--threads 4" APEX_PARALLEL_BUILD=8 APEX_CPP_EXT=1 APEX_CUDA_EXT=1 pip install -v --no-build-isolation .

cd $BASE_PATH
# CUDAHOSTCXX=g++-14 CC=/usr/bin/gcc-14 CXX=/usr/bin/g++-14 python3 -m pip install \
# 	-vv \
# 	--disable-pip-version-check \
# 	--no-cache-dir \
# 	--no-build-isolation \
# 	--config-settings "--build-option=--cpp_ext" \
# 	--config-settings "--build-option=--cuda_ext" \
# 	"git+https://github.com/NVIDIA/apex.git@24.04.01"  # April 27 2024 release; still shows up as apex-0.1
# #       "git+https://github.com/NVIDIA/apex.git@52e18c894223800cb611682dce27d88050edf1de"
# # commit corresponds to PR from Sept 2023: https://github.com/NVIDIA/apex/pull/1721

python3 -m pip install "git+https://github.com/microsoft/Megatron-DeepSpeed.git"

# HARDCODE
pip install --upgrade "jax[cuda${CUDA_VERSION_MAJOR}_local]" -f https://storage.googleapis.com/jax-releases/jax_cuda_releases.html
pip install pymongo optax flax
pip install "numpyro[cuda]" -f https://storage.googleapis.com/jax-releases/jax_cuda_releases.html

# --- MPI4JAX
pip install cython
git clone https://github.com/mpi4jax/mpi4jax.git
cd mpi4jax
CUDA_ROOT=$CUDA_TOOLKIT_BASE pip install --no-build-isolation --no-cache-dir --no-binary=mpi4jax -v .
cd $BASE_PATH

# ---- Adding inference packages to Fall 2025 Anaconda build
# porting some over from Fall 2024 standalone workshop (no TF) module

# vLLM
# https://docs.vllm.ai/en/latest/getting_started/installation/gpu.html#build-wheel-from-source
git clone https://github.com/vllm-project/vllm.git
cd vllm
python use_existing_torch.py
#pip install -r requirements-build.txt
#pip install -e . --no-build-isolation
uv pip install -r requirements-build.txt
VLLM_CUTLASS_SRC_DIR=$CUTLASS_PATH CUDAHOSTCXX=g++-12 CC=/usr/bin/gcc-12 CXX=/usr/bin/g++-12 uv pip install . --no-build-isolation
cd $BASE_PATH

# verl
# https://verl.readthedocs.io/en/latest/start/install.html
git clone https://github.com/volcengine/verl.git
cd verl
# If you need to run with megatron
bash scripts/install_vllm_sglang_mcore.sh
# Or if you simply need to run with FSDP
#USE_MEGATRON=0 bash scripts/install_vllm_sglang_mcore.sh
pip install --no-deps .
cd $BASE_PATH

# TRT-LLM
git clone https://github.com/argonne-lcf/LLM-Inference-Bench.git
cd LLM-Inference-Bench/TensorRT-LLM/A100/Benchmarking_Throughput
#MPICC=$(which mpicc) MPICXX=$(which mpicxx) pip install -r requirements.txt
MPICC=$(which cc) MPICXX=$(which CC) pip install -r requirements.txt
cd $BASE_PATH

echo "Cleaning up"
chmod -R u+w $DOWNLOAD_PATH/
rm -rf $DOWNLOAD_PATH

conda list

chmod -R a-w $BASE_PATH/

set +e
