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

module load PrgEnv-nvhpc  # not actually using NVHPC compilers to build TF
#module load PrgEnv-gnu
#module load gcc-mixed # get 12.2.0 (2022) instead of /usr/bin/gcc 7.5 (2019)
module load craype-accel-nvidia80  # wont load for PrgEnv-gnu; see HPE Case 5367752190
module load craype-x86-milan
export MPICH_GPU_SUPPORT_ENABLED=1
module list
echo $MPICH_DIR

# -------------------- begin HARDCODE of major built-from-source frameworks etc.
# unset *_TAG variables to build latest master/main branch (or "develop" in the case of DeepHyper)

PT_REPO_TAG="v2.5.0"
# HOROVOD_REPO_TAG="v0.28.1"  https://github.com/horovod/horovod/commit/3a31d933a13c7c885b8a673f4172b17914ad334d

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
CUDA_VERSION_MINOR=6
CUDA_VERSION_MINI=1

CUDA_VERSION=$CUDA_VERSION_MAJOR.$CUDA_VERSION_MINOR
CUDA_VERSION_FULL=$CUDA_VERSION.$CUDA_VERSION_MINI

CUDA_TOOLKIT_BASE=/soft/compilers/cudatoolkit/cuda-${CUDA_VERSION_FULL}
CUDA_HOME=${CUDA_TOOLKIT_BASE}
# KGF: adding new env vars to help Horovod find nvcc
# Failed to find nvcc.  Compiler requires the CUDA toolkit.  Please set the CUDAToolkit_ROOT  variable.
# https://github.com/Kitware/CMake/blob/master/Modules/CMakeDetermineCUDACompiler.cmake

# Relation to other two CMake modules?
# https://github.com/Kitware/CMake/blob/master/Modules/CMakeCUDACompiler.cmake.in
# https://github.com/Kitware/CMake/blob/master/Modules/FindCUDA/run_nvcc.cmake

export CUDA_ROOT=$CUDA_TOOLKIT_BASE  # set only in the final installed package, mpi4jax, in the original script
export CUDA_TOOLKIT_PATH=$CUDA_TOOLKIT_BASE  # set only before TF build, in original script
#####export CUDAToolkit_ROOT=$CUDA_TOOLKIT_BASE
# set only before CUTLASS install, in original script:
export CUDA_INSTALL_PATH=${CUDA_HOME}
export CUDACXX=${CUDA_INSTALL_PATH}/bin/nvcc

CUDA_DEPS_BASE=/soft/libraries/

CUDNN_VERSION_MAJOR=9
CUDNN_VERSION_MINOR=4
CUDNN_VERSION_EXTRA=0.58
CUDNN_VERSION=$CUDNN_VERSION_MAJOR.$CUDNN_VERSION_MINOR.$CUDNN_VERSION_EXTRA

# HARDCODE: manually renaming default cuDNN tarball name to fit this schema:
CUDNN_BASE=$CUDA_DEPS_BASE/cudnn/cudnn-cuda$CUDA_VERSION_MAJOR-linux-x64-v$CUDNN_VERSION

NCCL_VERSION_MAJOR=2
NCCL_VERSION_MINOR=23.4-1
NCCL_VERSION=$NCCL_VERSION_MAJOR.$NCCL_VERSION_MINOR
NCCL_BASE=$CUDA_DEPS_BASE/nccl/nccl_$NCCL_VERSION+cuda${CUDA_VERSION}_x86_64

TENSORRT_VERSION_MAJOR=10
TENSORRT_VERSION_MINOR=4.0.26
# TENSORRT_VERSION_MAJOR=8
# TENSORRT_VERSION_MINOR=6.1.6
TENSORRT_VERSION=$TENSORRT_VERSION_MAJOR.$TENSORRT_VERSION_MINOR
# HARDCODE
TENSORRT_BASE=$CUDA_DEPS_BASE/trt/TensorRT-$TENSORRT_VERSION.Linux.x86_64-gnu.cuda-12.5

echo "TENSORRT_BASE=${TENSORRT_BASE}"


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
CONDAVER='py311_24.9.2-0'
CONDA_DOWNLOAD_URL=https://repo.continuum.io/miniconda
CONDA_INSTALL_SH=Miniconda3-$CONDAVER-Linux-x86_64.sh
echo "Downloading miniconda installer"
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
#module load PrgEnv-nvhpc

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


PYTHON_VER=$(ls -d lib/python?.? | tail -c4)
echo PYTHON_VER=$PYTHON_VER

cat > .condarc << EOF
channels:
   - defaults
   - pytorch
   - conda-forge
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

echo "CONDA BINARY: $(which conda)"
echo "CONDA VERSION: $(conda --version)"
echo "PYTHON VERSION: $(python --version)"

set -e

echo "Conda install some dependencies"
conda install -y -n base conda-libmamba-solver
conda config --set solver libmamba

conda install -y -c defaults -c conda-forge cmake zip unzip astunparse setuptools future six requests dataclasses graphviz numba numpy pymongo conda-build pip libaio
conda install -y -c defaults -c conda-forge mkl mkl-include

# CUDA only: Add LAPACK support for the GPU if needed
# HARDCODE
conda install -y -c defaults -c pytorch -c conda-forge magma-cuda${CUDA_VERSION_MAJOR}${CUDA_VERSION_MINOR}
conda install -y -c defaults -c conda-forge mamba ccache

echo "Install some dependencies via pip"
pip install -U numpy numba ninja
pip install -U pip wheel mock gast portpicker pydot packaging pyyaml typing_extensions

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
module load PrgEnv-gnu
export CRAY_ACCEL_TARGET="nvidia80"
export CRAY_TCMALLOC_MEMFS_FORCE="1"
export CRAYPE_LINK_TYPE="dynamic"
export CRAY_ACCEL_VENDOR="nvidia"
export CRAY_CPU_TARGET="x86-64"

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
export CUSPARSELT_ROOT="/soft/libraries/cusparselt/libcusparse_lt-linux-x86_64-0.6.0.6/"
export CUSPARSELT_INCLUDE_PATH="/soft/libraries/cusparselt/libcusparse_lt-linux-x86_64-0.6.0.6/include"
# -------------

echo "TENSORRT_INCLUDE_DIR=${TENSORRT_INCLUDE_DIR}"
echo "TENSORRT_LIBRARY=${TENSORRT_LIBRARY}"
echo "CUSPARSELT_ROOT=${CUSPARSELT_ROOT}"
echo "CUSPARSELT_INCLUDE_PATH=${CUSPARSELT_INCLUDE_PATH}"

echo "PYTORCH_BUILD_VERSION=$PYTORCH_BUILD_VERSION and PYTORCH_BUILD_NUMBER=$PYTORCH_BUILD_NUMBER"
echo "PATH=${PATH}"
echo "which nvcc=$(which nvcc)"

#echo "CC=/opt/cray/pe/gcc/12.2.0/snos/bin/gcc CXX=/opt/cray/pe/gcc/12.2.0/snos/bin/g++ python setup.py bdist_wheel"
echo "CUDA_HOME=${CUDA_TOOLKIT_BASE} USE_MPI=ON USE_DISTRIBUTED=ON MPI_CXX_COMPILER=$(which CC) MPI_C_COMPILER=$(which cc) BUILD_TEST=0 CUDAHOSTCXX=g++-12 CC=/usr/bin/gcc-12 CXX=/usr/bin/g++-12 python setup.py bdist_wheel"

#echo "CC=$(which cc) CXX=$(which CC) python setup.py bdist_wheel"
#CC=$(which cc) CXX=$(which CC) python setup.py bdist_wheel

# Sirius??? keeps forcing wrong /bin/nvcc with above MPI settings
# //Path to a program.
# CUDA_NVCC_EXECUTABLE:FILEPATH=/bin/nvcc

####export CUDA_NVCC_EXECUTABLE=/soft/compilers/cudatoolkit/cuda-12.6.1/bin/nvcc

# HARDCODE
CUDA_HOME=${CUDA_TOOLKIT_BASE} USE_MPI=ON USE_DISTRIBUTED=ON MPI_CXX_COMPILER=$(which CC) MPI_C_COMPILER=$(which cc) BUILD_TEST=0 CUDAHOSTCXX=g++-12 CC=/usr/bin/gcc-12 CXX=/usr/bin/g++-12 python setup.py bdist_wheel

#echo "USE_MPI=OFF USE_DISTRIBUTED=ON BUILD_TEST=0 CUDAHOSTCXX=g++-12 CC=/usr/bin/gcc-12 CXX=/usr/bin/g++-12 python setup.py bdist_wheel"
# USE_MPI=OFF USE_DISTRIBUTED=ON BUILD_TEST=0 CUDAHOSTCXX=g++-12 CC=/usr/bin/gcc-12 CXX=/usr/bin/g++-12 python setup.py bdist_wheel
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

echo "Clone Horovod"

git clone --recursive $HOROVOD_REPO_URL
cd horovod

if [[ -z "$HOROVOD_REPO_TAG" ]]; then
    echo "Checkout Horovod master"
else
    echo "Checkout Horovod tag $HOROVOD_REPO_TAG"
    git checkout --recurse-submodules $HOROVOD_REPO_TAG
fi

echo "Build Horovod Wheel using MPI from $MPICH_DIR and NCCL from ${NCCL_BASE}"

# https://github.com/horovod/horovod/issues/3696#issuecomment-1248921736
echo "CUDAHOSTCXX=g++-12 CC=/usr/bin/gcc-12 CXX=/usr/bin/g++-12 HOROVOD_WITH_MPI=1 HOROVOD_CUDA_HOME=${CUDA_TOOLKIT_BASE} HOROVOD_NCCL_HOME=$NCCL_BASE HOROVOD_CMAKE=$(which cmake) HOROVOD_GPU_OPERATIONS=NCCL HOROVOD_WITHOUT_TENSORFLOW=1 HOROVOD_WITH_PYTORCH=1 HOROVOD_WITHOUT_MXNET=1 python setup.py bdist_wheel"

# HARDCODE: temp disable Horovod 0.28.1 + PyTorch >=2.1.x integration; C++17 required in PyTorch now (https://github.com/pytorch/pytorch/pull/100557)
# https://github.com/horovod/horovod/pull/3998
# https://github.com/horovod/horovod/issues/3996
CUDAHOSTCXX=g++-12 CC=/usr/bin/gcc-12 CXX=/usr/bin/g++-12 HOROVOD_WITH_MPI=1 HOROVOD_CUDA_HOME=${CUDA_TOOLKIT_BASE} HOROVOD_NCCL_HOME=$NCCL_BASE HOROVOD_CMAKE=$(which cmake) HOROVOD_GPU_OPERATIONS=NCCL HOROVOD_WITHOUT_TENSORFLOW=1 HOROVOD_WITH_PYTORCH=1 HOROVOD_WITHOUT_MXNET=1 python setup.py bdist_wheel

#CUDAHOSTCXX=g++-12 CC=/usr/bin/gcc-12 CXX=/usr/bin/g++-12 HOROVOD_WITH_MPI=1 HOROVOD_CUDA_HOME=${CUDA_TOOLKIT_BASE} HOROVOD_NCCL_HOME=$NCCL_BASE HOROVOD_CMAKE=$(which cmake) HOROVOD_GPU_OPERATIONS=NCCL HOROVOD_WITH_TENSORFLOW=1 HOROVOD_WITH_PYTORCH=1 HOROVOD_WITHOUT_MXNET=1 python setup.py bdist_wheel

HVD_WHL=$(find dist/ -name "horovod*.whl" -type f)
cp $HVD_WHL $WHEELS_PATH/
HVD_WHEEL=$(find $WHEELS_PATH/ -name "horovod*.whl" -type f)
echo "Install Horovod $HVD_WHEEL"
pip install --force-reinstall --no-cache-dir $HVD_WHEEL

# echo "Pip install TensorBoard profiler plugin"
# pip install tensorboard_plugin_profile tensorflow-datasets

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


pip install "pillow!=8.3.0,>=6.2.0"

cd $BASE_PATH
echo "Install PyTorch Vision from source"
git clone https://github.com/pytorch/vision.git
cd vision
# HARDCODE
git checkout v0.19.0

# HARDCODE
CUDAHOSTCXX=g++-12 CC=/usr/bin/gcc-12 CXX=/usr/bin/g++-12 python setup.py bdist_wheel
VISION_WHEEL=$(find dist/ -name "torchvision*.whl" -type f)
cp $VISION_WHEEL $WHEELS_PATH/
cd $WHEELS_PATH
echo "pip installing $(basename $VISION_WHEEL)"
pip install --force-reinstall --no-deps $(basename $VISION_WHEEL)

cd $BASE_PATH

pip install --no-deps timm
pip install opencv-python-headless

# HARDCODE
# pip install 'onnx==1.16.2' 'onnxruntime-gpu==1.18.1'
# pip install tf2onnx
# pip install onnx-tf
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
pip install lightning  # pytorch-lightning
pip install ml-collections
pip install gpytorch xgboost multiprocess py4j
# HARDCODE
CUDAHOSTCXX=g++-12 CC=/usr/bin/gcc-12 CXX=/usr/bin/g++-12 pip install --no-build-isolation git+https://github.com/FalkonML/falkon.git
pip install pykeops   # wants nonstandard env var set: CUDA_PATH=$CUDA_HOME
pip install hydra-core hydra_colorlog arviz pyright celerite seaborn xarray bokeh matplotx aim torchviz rich parse
pip install jupyter
pip install climetlab
pip install torch_cluster # ==1.6.3
#pip install tensorboardX

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
CUDAHOSTCXX=g++-12 CC=/usr/bin/gcc-12 CXX=/usr/bin/g++-12 cmake .. -DCUTLASS_NVCC_ARCHS=80 -DCUTLASS_ENABLE_CUBLAS=ON -DCUTLASS_ENABLE_CUDNN=ON
make cutlass_profiler -j32

cd $BASE_PATH
echo "Install DeepSpeed from source"
git clone https://github.com/microsoft/DeepSpeed.git
cd DeepSpeed
# HARDCODE
git checkout v0.14.4
export CFLAGS="-I${CONDA_PREFIX}/include/"
export LDFLAGS="-L${CONDA_PREFIX}/lib/ -Wl,--enable-new-dtags,-rpath,${CONDA_PREFIX}/lib"
pip install deepspeed-kernels

CUDAHOSTCXX=g++-12 CC=/usr/bin/gcc-12 CXX=/usr/bin/g++-12 NVCC_PREPEND_FLAGS="--forward-unknown-opts" DS_BUILD_SPARSE_ATTN=0 DS_BUILD_OPS=1 DS_BUILD_AIO=1 pip install --verbose . ### --global-option="build_ext" --global-option="-j16"
# the parallel build options seem to cause issues

# > ds_report
cd $BASE_PATH

# HARDCODE
# Apex (for Megatron-Deepspeed)
CUDAHOSTCXX=g++-12 CC=/usr/bin/gcc-12 CXX=/usr/bin/g++-12 python3 -m pip install \
	-vv \
	--disable-pip-version-check \
	--no-cache-dir \
	--no-build-isolation \
	--config-settings "--build-option=--cpp_ext" \
	--config-settings "--build-option=--cuda_ext" \
	"git+https://github.com/NVIDIA/apex.git@24.04.01"  # April 27 2024 release; still shows up as apex-0.1
#       "git+https://github.com/NVIDIA/apex.git@52e18c894223800cb611682dce27d88050edf1de"
# commit corresponds to PR from Sept 2023: https://github.com/NVIDIA/apex/pull/1721

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

echo "Cleaning up"
chmod -R u+w $DOWNLOAD_PATH/
rm -rf $DOWNLOAD_PATH

conda list

chmod -R a-w $BASE_PATH/

set +e
