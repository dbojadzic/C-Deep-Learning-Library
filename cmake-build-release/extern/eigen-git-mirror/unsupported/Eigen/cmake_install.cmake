# Install script for directory: /home/mrtheamir/Desktop/libdl/extern/eigen-git-mirror/unsupported/Eigen

# Set the install prefix
if(NOT DEFINED CMAKE_INSTALL_PREFIX)
  set(CMAKE_INSTALL_PREFIX "/usr/local")
endif()
string(REGEX REPLACE "/$" "" CMAKE_INSTALL_PREFIX "${CMAKE_INSTALL_PREFIX}")

# Set the install configuration name.
if(NOT DEFINED CMAKE_INSTALL_CONFIG_NAME)
  if(BUILD_TYPE)
    string(REGEX REPLACE "^[^A-Za-z0-9_]+" ""
           CMAKE_INSTALL_CONFIG_NAME "${BUILD_TYPE}")
  else()
    set(CMAKE_INSTALL_CONFIG_NAME "Release")
  endif()
  message(STATUS "Install configuration: \"${CMAKE_INSTALL_CONFIG_NAME}\"")
endif()

# Set the component getting installed.
if(NOT CMAKE_INSTALL_COMPONENT)
  if(COMPONENT)
    message(STATUS "Install component: \"${COMPONENT}\"")
    set(CMAKE_INSTALL_COMPONENT "${COMPONENT}")
  else()
    set(CMAKE_INSTALL_COMPONENT)
  endif()
endif()

# Install shared libraries without execute permission?
if(NOT DEFINED CMAKE_INSTALL_SO_NO_EXE)
  set(CMAKE_INSTALL_SO_NO_EXE "1")
endif()

# Is this installation the result of a crosscompile?
if(NOT DEFINED CMAKE_CROSSCOMPILING)
  set(CMAKE_CROSSCOMPILING "FALSE")
endif()

if("x${CMAKE_INSTALL_COMPONENT}x" STREQUAL "xDevelx" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/include/eigen3/unsupported/Eigen" TYPE FILE FILES
    "/home/mrtheamir/Desktop/libdl/extern/eigen-git-mirror/unsupported/Eigen/AdolcForward"
    "/home/mrtheamir/Desktop/libdl/extern/eigen-git-mirror/unsupported/Eigen/AlignedVector3"
    "/home/mrtheamir/Desktop/libdl/extern/eigen-git-mirror/unsupported/Eigen/ArpackSupport"
    "/home/mrtheamir/Desktop/libdl/extern/eigen-git-mirror/unsupported/Eigen/AutoDiff"
    "/home/mrtheamir/Desktop/libdl/extern/eigen-git-mirror/unsupported/Eigen/BVH"
    "/home/mrtheamir/Desktop/libdl/extern/eigen-git-mirror/unsupported/Eigen/EulerAngles"
    "/home/mrtheamir/Desktop/libdl/extern/eigen-git-mirror/unsupported/Eigen/FFT"
    "/home/mrtheamir/Desktop/libdl/extern/eigen-git-mirror/unsupported/Eigen/IterativeSolvers"
    "/home/mrtheamir/Desktop/libdl/extern/eigen-git-mirror/unsupported/Eigen/KroneckerProduct"
    "/home/mrtheamir/Desktop/libdl/extern/eigen-git-mirror/unsupported/Eigen/LevenbergMarquardt"
    "/home/mrtheamir/Desktop/libdl/extern/eigen-git-mirror/unsupported/Eigen/MatrixFunctions"
    "/home/mrtheamir/Desktop/libdl/extern/eigen-git-mirror/unsupported/Eigen/MoreVectorization"
    "/home/mrtheamir/Desktop/libdl/extern/eigen-git-mirror/unsupported/Eigen/MPRealSupport"
    "/home/mrtheamir/Desktop/libdl/extern/eigen-git-mirror/unsupported/Eigen/NonLinearOptimization"
    "/home/mrtheamir/Desktop/libdl/extern/eigen-git-mirror/unsupported/Eigen/NumericalDiff"
    "/home/mrtheamir/Desktop/libdl/extern/eigen-git-mirror/unsupported/Eigen/OpenGLSupport"
    "/home/mrtheamir/Desktop/libdl/extern/eigen-git-mirror/unsupported/Eigen/Polynomials"
    "/home/mrtheamir/Desktop/libdl/extern/eigen-git-mirror/unsupported/Eigen/Skyline"
    "/home/mrtheamir/Desktop/libdl/extern/eigen-git-mirror/unsupported/Eigen/SparseExtra"
    "/home/mrtheamir/Desktop/libdl/extern/eigen-git-mirror/unsupported/Eigen/SpecialFunctions"
    "/home/mrtheamir/Desktop/libdl/extern/eigen-git-mirror/unsupported/Eigen/Splines"
    )
endif()

if("x${CMAKE_INSTALL_COMPONENT}x" STREQUAL "xDevelx" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/include/eigen3/unsupported/Eigen" TYPE DIRECTORY FILES "/home/mrtheamir/Desktop/libdl/extern/eigen-git-mirror/unsupported/Eigen/src" FILES_MATCHING REGEX "/[^/]*\\.h$")
endif()

if(NOT CMAKE_INSTALL_LOCAL_ONLY)
  # Include the install script for each subdirectory.
  include("/home/mrtheamir/Desktop/libdl/cmake-build-release/extern/eigen-git-mirror/unsupported/Eigen/CXX11/cmake_install.cmake")

endif()

