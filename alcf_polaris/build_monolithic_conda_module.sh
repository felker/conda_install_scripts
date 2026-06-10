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
# KGF (2026-06-10): bumped to match Sophia 2026-06 build. Verify each tag against
# upstream "Latest" before building.
#DH_REPO_TAG="0.4.2"
DH_REPO_URL=https://github.com/deephyper/deephyper.git

# Versions verified against upstream "Latest" release pages on 2026-06-10.
TF_REPO_TAG="v2.21.0"   # 2026-03-06
PT_REPO_TAG="v2.12.0"   # 2026-05-13
# Horovod dropped: 0.28.1 incompatible with PyTorch >=2.1 (C++17), upstream dormant. Section commented out below.
#HOROVOD_REPO_TAG=""
TF_REPO_URL=https://github.com/tensorflow/tensorflow.git
#HOROVOD_REPO_URL=https://github.com/uber/horovod.git
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
export TF_PYTHON_VERSION=3.13
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
# Miniforge's setup.sh only defines the `conda` shell function; it does *not*
# `conda activate base`. Without that, $CONDA_PREFIX stays empty, and later
# steps that expand ${CONDA_PREFIX}/... (magma extract, CUDA bindings
# redirector patch, etc.) silently become /... and fail.
conda activate base
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

# --override-channels: ignore any `defaults` channel leaking in from ~/.condarc.
# That channel still serves an ancient graphviz=2.38.0 that blocks the solve on
# modern Python.
# pymongo dropped from conda spec: conda-forge has not yet published a py313
# build, which fails the solve under the python=3.13 pin. It is pip-installed
# later in the script.
# cmake pinned <4: PyTorch 2.12 still vendors ancient subprojects (NNPACK/confu/
# six, FXdiv, FP16, psimd, protobuf, ittapi, pthreadpool) whose CMakeLists.txt
# declare cmake_minimum_required < 3.5, which CMake 4.0 dropped support for
# outright. Until upstream PyTorch bumps every vendored CMakeLists (or sets
# CMAKE_POLICY_VERSION_MINIMUM on ExternalProject_Add downloads), stay on 3.x.
# NOTE: do NOT add `rust` (conda-forge) to this install. It pulls in
# rust_linux-64 -> gcc_linux-64 -> gcc_impl_linux-64 / binutils_impl_linux-64
# / sysroot_linux-64 / kernel-headers_linux-64, which makes the conda
# _compiler_compat/ld a live symlink to x86_64-conda-linux-gnu-ld. That linker
# is sysroot-locked to conda's CDT and won't search /usr/lib64 or the Cray PE
# search paths, so subsequent mpi4py / h5py / any-MPI-linking build fails to
# resolve transitive NEEDED libs from libmpi.so (link warns "not found",
# undefined refs follow). libprotobuf is innocent (only pulls libstdcxx-ng
# runtime) and isn't needed once we drop SGLang. If a future package needs
# rustc, install it via rustup outside the conda env, not via conda-forge `rust`.
conda install -y --override-channels -c conda-forge "cmake>=3.27,<4" zip unzip astunparse setuptools future six requests dataclasses graphviz numba numpy conda-build pip libaio
conda install -y --override-channels -c conda-forge mkl mkl-include git-lfs  # onednn mkl-dnn  ### on ThetaGPU

# MAGMA (CUDA LAPACK): the magma-cuda{NN} conda package is no longer published
# on any channel as of late 2024 (anaconda.org returns empty for conda-forge /
# pytorch / pytorch-nightly). PyTorch's own CI script switched to extracting the
# prebuilt static tarball from S3 directly.
# https://github.com/pytorch/pytorch/issues/138506
MAGMA_CUDA_TAG="cuda${CUDA_VERSION_MAJOR}${CUDA_VERSION_MINOR}"   # e.g. cuda129
MAGMA_TARBALL="magma-${MAGMA_CUDA_TAG}-2.6.1-1.tar.bz2"
(
    set -x
    cd $DOWNLOAD_PATH
    curl -fOLs "https://ossci-linux.s3.us-east-1.amazonaws.com/${MAGMA_TARBALL}"
    mkdir -p magma_extract && cd magma_extract
    tar -xf ../${MAGMA_TARBALL}
    cp -r include/* ${CONDA_PREFIX}/include/
    cp -r lib/*     ${CONDA_PREFIX}/lib/
)

conda install -y --override-channels -c conda-forge mamba ccache

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

# HARDCODE: TF 2.21.0 hermetic build expects Clang 18.x as host compiler.
# Polaris stages clang via /soft/modulefiles/llvm/release-18.1.6.
module use /soft/modulefiles
module load llvm/release-18.1.6
export GCC_HOST_COMPILER_PATH=$(which gcc-14)
export CC=/soft/compilers/llvm/release-18.1.6/bin/clang
export BAZEL_COMPILER=$CC

# Hermetic CUDA build (XLA): no longer uses TF_CUDA_PATHS at compile time, takes
# paths via HERMETIC_*_VERSION env vars. https://openxla.org/xla/hermetic_cuda
#
# These must be exported BEFORE ./configure so it doesn't prompt for them interactively.
# TF 2.21.0's allowlist for 12.x is the constraint; if 12.9.1 isn't in it, bazel
# will fail with: "supported CUDA versions are [...]. Please provide a supported
# version". In that case, try 12.8.1 / 12.6.3 (TF historically lags the latest CUDA
# release by ~1 quarter). CUDNN_VERSION_MAJOR.CUDNN_VERSION_MINOR = 9.13 / NCCL_VERSION
# 2.28.3-1 / NVSHMEM 3.3.9 likewise need to be in the TF-internal allowlist.
export HERMETIC_CUDA_VERSION=$CUDA_VERSION_FULL
export HERMETIC_CUDNN_VERSION=$CUDNN_VERSION_MAJOR.$CUDNN_VERSION_MINOR
export HERMETIC_NCCL_VERSION=$NCCL_VERSION
export HERMETIC_NVSHMEM_VERSION="3.3.9"
export HERMETIC_CUDA_COMPUTE_CAPABILITIES="sm_80"

yes "" | ./configure

echo "Bazel Build TensorFlow"

HOME=$DOWNLOAD_PATH bazel build --announce_rc --jobs=128 --loading_phase_threads=6 \
    --verbose_failures --config=cuda --config=cuda_wheel \
    --@local_config_cuda//cuda:override_include_cuda_libs=false \
    --repo_env=HERMETIC_CUDA_VERSION=${HERMETIC_CUDA_VERSION} \
    --repo_env=HERMETIC_CUDNN_VERSION=${HERMETIC_CUDNN_VERSION} \
    --repo_env=HERMETIC_NCCL_VERSION=${HERMETIC_NCCL_VERSION} \
    --repo_env=HERMETIC_NVSHMEM_VERSION=${HERMETIC_NVSHMEM_VERSION} \
    --repo_env=HERMETIC_CUDA_COMPUTE_CAPABILITIES=${HERMETIC_CUDA_COMPUTE_CAPABILITIES} \
    --copt="-Wno-error=unused-command-line-argument" --cxxopt="-D_GLIBCXX_USE_CXX11_ABI=0" \
    //tensorflow/tools/pip_package:wheel

echo "Run wheel building"
cp ./bazel-bin/tensorflow/tools/pip_package/wheel_house/*.whl $WHEELS_PATH
echo "Install TensorFlow"
pip install $(find $WHEELS_PATH/ -name "tensorflow*.whl" -type f)

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
export TORCH_CUDA_ARCH_LIST="8.0"
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

# Build with MPI support via the Cray PE wrappers. -lmpi_gtl_cuda is needed so the
# GPU-Transport-Layer libmpi is linked rather than the non-GTL libmpi_gnu_*.so.12
# that PrgEnv-gnu would otherwise hand us (would silently break CUDA-aware MPI).
export USE_MPI=1
BUILD_TEST=0 CUDAHOSTCXX=g++-14 CC=cc CXX=CC \
    LDFLAGS="-L/opt/cray/pe/lib64 -Wl,-rpath,/opt/cray/pe/lib64 -lmpi_gtl_cuda ${LDFLAGS}" \
    python setup.py bdist_wheel
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
# Tripwire: if anything in the conda install above resurrected the conda
# compiler toolchain (gcc_impl_linux-64 / binutils_impl_linux-64 /
# sysroot_linux-64), _compiler_compat/ld becomes a live symlink to
# x86_64-conda-linux-gnu-ld and the mpi4py link below will fail (sysroot-locked
# linker won't honor the Cray PE / system search paths). Bail loudly here
# instead of churning on the link error. Confirmed culprit historically:
# conda-forge `rust`.
if conda list 2>/dev/null | grep -qE '^(gcc_impl_linux-64|binutils_impl_linux-64|sysroot_linux-64) '; then
    echo "ERROR: conda compiler toolchain detected in env; mpi4py link will fail."
    echo "Check what pulled in gcc_impl/binutils_impl/sysroot_linux-64 (conda-forge"
    echo "\`rust\` is the usual suspect). Do not install rust via conda-forge."
    conda list | grep -E '^(sysroot|binutils|gcc|gxx|kernel-headers|libgcc|libstdcxx|libcxx|compiler_)' || true
    exit 1
fi

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

# PyG wheels bundle for current torch+CUDA. Verify wheel URL exists at https://data.pyg.org/whl/
# before uncommenting; pyg-team routinely lags behind new torch releases, and
# the torch-2.12.0+cu129 wheel set may not be published yet. If pyg lags, fall
# back to torch-geometric (pure Python) and source-build pyg_lib only if needed.
# HARDCODE
# pip install pyg_lib torch_scatter torch_sparse torch_cluster torch_spline_conv \
#     -f https://data.pyg.org/whl/torch-2.12.0+cu${CUDA_VERSION_MAJOR}${CUDA_VERSION_MINOR}.html
pip install torch-geometric
pip install pillow
#pip install "pillow!=8.3.0,>=6.2.0"

cd $BASE_PATH
echo "Install PyTorch Vision from source"
git clone https://github.com/pytorch/vision.git
cd vision
# HARDCODE: torchvision version pairs with PyTorch; v0.27.0 matches torch 2.12.0.
git checkout v0.27.0

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
pip install 'onnx==1.21.0' 'onnxruntime-gpu==1.26.0'
# tf2onnx removed: pulls protobuf~=3.20 which downgrades onnx/protobuf and breaks the env.
#pip install tf2onnx
pip install onnx-tf
pip install huggingface-hub
pip install transformers evaluate datasets accelerate
pip install --no-deps xformers
# Flash-attention: pin to last stable 2.x. fa4-v4.0.0.beta* is the new architecture
# (different API) and still in beta as of May 2026.
#
# Subshell to scope the env tweaks (no pollution into later pip installs):
#   * MAX_JOBS / NVCC_THREADS: flash-attn .cu files are CUTLASS-heavy and each
#     nvcc invocation can use 4-8 GB. Without throttling, torch.cpp_extension
#     spawns nproc parallel nvcc jobs (~64 on a Polaris login node) -> OOM-kill
#     and a stream of bare "FAILED: [code=255]" with no nvcc stderr. flash-attn's
#     README documents MAX_JOBS as the official knob.
#   * TORCH_CUDA_ARCH_LIST=8.0: Polaris is A100 (sm_80) only. Without this pin,
#     flash-attn 2.8.3 also builds sm_90 (Hopper), sm_100 / sm_120 (Blackwell)
#     by default, ~4xing build time and per-file memory for no reachable hardware.
#   * unset CC CXX: the modulefile sets CC=/usr/bin/gcc-14. flash-attn does not
#     need a specific host compiler; letting torch.cpp_extension pick its default
#     (matching what built libtorch) avoids ABI surprises.
(
    unset CC CXX
    export MAX_JOBS=4
    export NVCC_THREADS=2
    export TORCH_CUDA_ARCH_LIST="8.0"
    export FLASH_ATTENTION_FORCE_BUILD=TRUE   # skip the +cu12 wheel-URL guess
    pip install --no-build-isolation "flash-attn==2.8.3"
)
pip install scikit-image
pip install ipython
pip install line_profiler
pip install torch-tb-profiler
pip install torchinfo
# HARDCODE
pip install cupy-cuda${CUDA_VERSION_MAJOR}x
pip install lightning # pytorch-lightning
#pip install "git+https://github.com/saforem2/ezpz"
pip install "git+https://github.com/saforem2/ezpz.git@v0.9.0"
#pip install "git+https://github.com/saforem2/ezpz.git@saforem2/tests"
# make sure TERM is set, run "wandb login", add API key, then run ezpz-test
pip install ml-collections
pip install gpytorch
#pip install xgboost  # KGF: TODO, this installs "nvidia-nccl-cu12" https://github.com/dmlc/xgboost/blob/master/python-package/pyproject.toml
# https://xgboost.readthedocs.io/en/stable/changes/v2.1.0.html#nccl-is-now-fetched-from-pypi

# xgboost
git clone --recursive https://github.com/dmlc/xgboost
cd xgboost
cmake -B build -S . -DUSE_CUDA=ON -DUSE_NCCL=ON -DNCCL_ROOT=$NCCL_BASE -DCMAKE_CUDA_ARCHITECTURES=80 -DCMAKE_CXX_COMPILER=/usr/bin/g++-14 -DCMAKE_C_COMPILER=/usr/bin/gcc-14 -DUSE_DLOPEN_NCCL=ON -GNinja
# USE_DLOPEN_NCCL, BUILD_WITH_SHARED_NCCL, -DPLUGIN_FEDERATED=ON
cd build && ninja
cd ../python-package
pip install . --config-settings use_cuda=True --config-settings use_nccl=True --config-settings use_dlopen_nccl=True
# TODO: enable RAPIDS integration https://xgboost.readthedocs.io/en/stable/python/rmm-examples/index.html

# pip still drops nvidia-nccl-cu12 into the env even with the cmake flags above; remove it.
# TODO: test if distributed GPU xgboost with Dask works after removing this PyPI package
# https://xgboost.readthedocs.io/en/stable/gpu/index.html#multi-node-multi-gpu-training
# https://xgboost.readthedocs.io/en/stable/tutorials/dask.html
pip uninstall -y nvidia-nccl-cu12 || true
cd $BASE_PATH

pip install multiprocess py4j
# HARDCODE
CUDAHOSTCXX=g++-14 CC=/usr/bin/gcc-14 CXX=/usr/bin/g++-14 pip install --no-build-isolation git+https://github.com/FalkonML/falkon.git
pip install pykeops   # wants nonstandard env var set: CUDA_PATH=$CUDA_HOME
pip install hydra-core hydra_colorlog accelerate arviz pyright celerite seaborn xarray bokeh matplotx torchviz rich parse
# pip install aim # no aimrocks wheel 0.5.x for python 3.13.x. Latest is 0.5.2 for PyTorch 3.12
pip install jupyter
#pip install climetlab
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
git clone https://github.com/deepspeedai/DeepSpeed.git
cd DeepSpeed
# HARDCODE
git checkout v0.19.0
export CFLAGS="-I${CONDA_PREFIX}/include/"
export LDFLAGS="-L${CONDA_PREFIX}/lib/ -Wl,--enable-new-dtags,-rpath,${CONDA_PREFIX}/lib"
pip install deepspeed-kernels

# HARDCODE: patch op_builder/dc.py to include NCCL header when NCCL_INCLUDE_DIR is set.
# Without this, DeepCompile fails with "fatal error: nccl.h: No such file or directory".
# May or may not be needed on v0.19.0 (was needed on v0.17.6); apply || true so it's a no-op
# if upstream has merged the equivalent fix.
git apply <<'PATCH' || true
diff --git a/op_builder/dc.py b/op_builder/dc.py
index 15b25bf3..bce4e97d 100644
--- a/op_builder/dc.py
+++ b/op_builder/dc.py
@@ -33,6 +33,10 @@ class DeepCompileBuilder(TorchCPUOpBuilder):
             CUDA_INCLUDE = []
         elif not self.is_rocm_pytorch():
             CUDA_INCLUDE = [os.path.join(torch.utils.cpp_extension.CUDA_HOME, "include")]
+            # If set, append a single NCCL include dir.
+            _nccl_inc = os.environ.get("NCCL_INCLUDE_DIR")
+            if _nccl_inc and _nccl_inc not in CUDA_INCLUDE:
+                CUDA_INCLUDE.append(_nccl_inc)
         else:
             CUDA_INCLUDE = [
                 os.path.join(torch.utils.cpp_extension.ROCM_HOME, "include"),
PATCH

# pip >= 25.3 deprecated --global-option / --build-option (PEP517 always-on) and dropped
# setup.py bdist_wheel. DeepSpeed is not PEP517-compliant, so we need --no-build-isolation
# plus the new -C config-settings syntax.
# https://pip.pypa.io/en/latest/news/#v25-3
# https://github.com/deepspeedai/DeepSpeed/issues/7031
#
# DS_BUILD_CCL_COMM=0: skip deepspeed.ops.comm.deepspeed_ccl_comm_op (Intel oneCCL
# CPU collective backend). Polaris is NVIDIA-only; we don't ship oneapi/ccl.hpp. The
# CCLCommBuilder.is_compatible() gate evidently regressed in v0.19.0 and the op
# now compiles unconditionally under DS_BUILD_OPS=1, failing with
# "fatal error: oneapi/ccl.hpp: No such file or directory". NCCL is what DeepSpeed
# actually uses for GPU collectives here (via PyTorch's torch.distributed), so
# dropping the CCL backend is harmless.
TORCH_CUDA_ARCH_LIST="8.0" CUDAHOSTCXX=g++-14 CC=/usr/bin/gcc-14 CXX=/usr/bin/g++-14 \
    NVCC_PREPEND_FLAGS="--forward-unknown-opts" \
    DS_BUILD_OPS=1 DS_BUILD_CCL_COMM=0 \
    pip install -v . -C="--global-option=build_ext" -C="--build-option=-j8" --no-build-isolation

# > ds_report  -- run this after build to confirm op compilation; expect [YES] for fused_adam,
#   cpu_adam, gds, transformer*, etc. fp_quantizer/sparse_attn will be [NO] (incompatible).
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

python3 -m pip install "git+https://github.com/deepspeedai/Megatron-DeepSpeed.git"

# HARDCODE
pip install --upgrade "jax[cuda${CUDA_VERSION_MAJOR}_local]" -f https://storage.googleapis.com/jax-releases/jax_cuda_releases.html
pip install pymongo optax flax
#pip install "numpyro[cuda]" -f https://storage.googleapis.com/jax-releases/jax_cuda_releases.html
# starts downloading nvidia_cudnn_cu12-9.13.1.26-py3-none-manylinux_2_27_x86_64.whl
# nvshmem, nvjitlink, ...

git clone https://github.com/pyro-ppl/numpyro.git
cd numpyro
CC=/usr/bin/gcc-14 CXX=/usr/bin/g++-14 pip install .
cd $BASE_PATH

# --- MPI4JAX
# nanobind is now required for mpi4jax's from-source build.
pip install cython nanobind
git clone https://github.com/mpi4jax/mpi4jax.git
cd mpi4jax
CUDA_ROOT=$CUDA_TOOLKIT_BASE pip install --no-build-isolation --no-cache-dir --no-binary=mpi4jax -v .
cd $BASE_PATH

###############################################################################
# Inference stack (mamba-ssm, megatron-core, TransformerEngine, vLLM, FlashInfer,
# verl). Ported from Sophia 2026-06 build.
#
# verl + vLLM + TransformerEngine have tight inter-version coupling.
# If "latest" tags break, fall back to Polaris-validated Fall 2025 combo:
#   vLLM v0.9.1, SGLang v0.5.3rc0, TransformerEngine v2.7, verl v0.5.0,
#   FlashInfer v0.3.1, transformers <4.54.0.
###############################################################################

# mamba-ssm + causal-conv1d are torch CUDA extensions: their setup.py does
# `import torch` at import time, so they need --no-build-isolation to see our
# from-source torch. Same OOM/arch/MPI-leak knobs as flash-attn (see above).
(
    unset CC CXX
    export MAX_JOBS=4
    export NVCC_THREADS=2
    export TORCH_CUDA_ARCH_LIST="8.0"
    pip install --no-build-isolation "mamba-ssm[causal-conv1d]"
)
pip install megatron-core

# TransformerEngine (PyTorch + JAX bindings).
# Note: pip 25.x rejects the old `#egg=name[extras]` fragment; use PEP 508
# direct-URL syntax (`name[extras] @ url`) instead.
#
# TE's common/util/logging.h does `#include "nccl.h"` unconditionally; on Polaris
# NCCL lives under /soft (no system install), so the compiler needs to be told
# where to look. NVTE_NCCL_HOME is TE's documented var; CPATH/LIBRARY_PATH are
# belt-and-suspenders for any subproject that doesn't honor it.
# Same OOM/arch/MPI-leak knobs as flash-attn (see above) - TE has even more .cu
# TUs and will fan out to nproc by default.
(
    unset CC CXX
    export MAX_JOBS=4
    export NVCC_THREADS=2
    export TORCH_CUDA_ARCH_LIST="8.0"
    export NVTE_NCCL_HOME="$NCCL_BASE"
    export CPATH="$NCCL_BASE/include:${CPATH:-}"
    export LIBRARY_PATH="$NCCL_BASE/lib:${LIBRARY_PATH:-}"
    pip install --no-build-isolation \
        "transformer_engine[pytorch,jax] @ git+https://github.com/NVIDIA/TransformerEngine.git@v2.15"
)

pip install pylatexenc qwen-vl-utils

# vLLM from source (use_existing_torch.py reuses our PyTorch build so it doesn't
# pull a binary torch wheel that overrides ours).
git clone https://github.com/vllm-project/vllm.git
cd vllm
git checkout v0.22.1
python use_existing_torch.py
pip install uv
# vLLM 0.10+ reorganized requirements/ into per-platform subdirs.
uv pip install -r requirements/build/cuda.txt --system
# Cap build parallelism: vLLM pulls in vllm-flash-attn (CUTLASS-heavy, ~340 TUs).
# Default ninja -j$(nproc) generates transient .o piles that have OOM'd /soft.
# Same sm_80 / MPI-leak hygiene as flash-attn / TE above.
(
    unset CC CXX
    export MAX_JOBS=4
    export NVCC_THREADS=2
    export CMAKE_BUILD_PARALLEL_LEVEL=4
    export TORCH_CUDA_ARCH_LIST="8.0"
    VLLM_CUTLASS_SRC_DIR=$CUTLASS_PATH uv pip install . --no-build-isolation --system
)
cd $BASE_PATH

# FlashInfer (v0.4.0+ uses a different install procedure than v0.3.x)
git clone https://github.com/flashinfer-ai/flashinfer.git --recursive
cd flashinfer
git checkout v0.6.12
pip install apache-tvm-ffi
export FLASHINFER_CUDA_ARCH_LIST="8.0"
# cuda-bindings + NVSHMEM4Py are tied to CUDA major. Polaris is cuda12.
pip install "cuda-bindings==12.9.2" nvshmem4py-cu${CUDA_VERSION_MAJOR}
# v0.4.0+ multi-wheel install:
python -m pip install -v .
if [ -d flashinfer-cubin ]; then
    cd flashinfer-cubin && python -m build --no-isolation --wheel && python -m pip install dist/*.whl && cd ..
fi
if [ -d flashinfer-jit-cache ]; then
    cd flashinfer-jit-cache && python -m build --no-isolation --wheel && python -m pip install dist/*.whl && cd ..
fi
cd $BASE_PATH

# Suppress noisy cuda-bindings deprecation FutureWarning that fires on every Python exit
# (triggered by PyModuleSnooper iterating `cuda.__version__`).
PYTHON_VER_MINOR=$(python -c 'import sys; print(sys.version_info.minor)')
if [ -f ${CONDA_PREFIX}/lib/python3.${PYTHON_VER_MINOR}/site-packages/_cuda_bindings_redirector.py ]; then
    sed -i '16,21d' ${CONDA_PREFIX}/lib/python3.${PYTHON_VER_MINOR}/site-packages/_cuda_bindings_redirector.py
fi

# SGLang -- DISABLED (KGF 2026-06-10)
# SGLang's transitive dep outlines_core has no py3.13 wheel and falls back to a
# source build that needs rustc. Installing conda-forge `rust` pulls in the
# full conda compiler toolchain (gcc_impl/binutils_impl/sysroot_linux-64),
# which sysroot-locks _compiler_compat/ld and breaks mpi4py / h5py MPI linking
# (see the early conda install comment). Until we move rustc to rustup or
# outlines_core ships a py3.13 wheel, just skip SGLang. SGLang 0.5.12 also
# pins torch==2.11.0 hard, which would clobber our from-source torch 2.12.0
# without --no-deps gymnastics anyway.
# git clone -b v0.5.12 https://github.com/sgl-project/sglang.git
# cd sglang/python
# uv pip install . --system --no-deps
# cd $BASE_PATH

# verl: install with --no-deps to avoid stomping on our vLLM version pins,
# then add back the few deps verl actually needs.
git clone https://github.com/volcengine/verl.git
cd verl
git checkout v0.8.0
CC=/usr/bin/gcc-14 CXX=/usr/bin/g++-14 pip install --no-deps .
cd $BASE_PATH
pip install torchdata codetiming tensordict

# TRT-LLM
# git clone https://github.com/argonne-lcf/LLM-Inference-Bench.git
# cd LLM-Inference-Bench/TensorRT-LLM/A100/Benchmarking_Throughput  ## hasnt been updated since Sep 2024
##MPICC=$(which mpicc) MPICXX=$(which mpicxx) pip install -r requirements.txt

# git clone https://github.com/NVIDIA/TensorRT-LLM.git
# cd TensorRT-LLM
# git submodule update --init --recursive
# git lfs pull
# MPICC=$(which cc) MPICXX=$(which CC) pip install -r requirements.txt

# # https://nvidia.github.io/TensorRT-LLM/installation/build-from-source-linux.html#
# cd $BASE_PATH

echo "Cleaning up"
chmod -R u+w $DOWNLOAD_PATH/
# KGF: lot's of NFS errors lately with the following command
rm -rf $DOWNLOAD_PATH || true
# rm: cannot remove '/soft/applications/conda/2025-09-24/DOWNLOADS/.cache/bazel/_bazel_felker/8d63422c7e36f924c4f33033ca2fe451/server': Directory not empty
rm -rf $DOWNLOAD_PATH || true

conda list

chmod -R a-w $BASE_PATH/

set +e
