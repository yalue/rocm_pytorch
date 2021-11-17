# A quick script so I can remember my build command.

export CMAKE_PREFIX_PATH=${CONDA_PREFIX:-"$(dirname $(which conda))/../"}
export BUILD_TEST=0
export MAX_JOBS=4

# Run setup.py, redirecting output to build_log.txt for later review if needed.
stdbuf -oL python setup.py install 2>&1 | tee build_log.txt

