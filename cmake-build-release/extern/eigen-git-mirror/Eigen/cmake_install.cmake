# Install script for directory: /home/mrtheamir/Desktop/libdl/extern/eigen-git-mirror/Eigen

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
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/include/eigen3/Eigen" TYPE FILE FILES
    "/home/mrtheamir/Desktop/libdl/extern/eigen-git-mirror/Eigen/Cholesky"
    "/home/mrtheamir/Desktop/libdl/extern/eigen-git-mirror/Eigen/CholmodSupport"
    "/home/mrtheamir/Desktop/libdl/extern/eigen-git-mirror/Eigen/Core"
    "/home/mrtheamir/Desktop/libdl/extern/eigen-git-mirror/Eigen/Dense"
    "/home/mrtheamir/Desktop/libdl/extern/eigen-git-mirror/Eigen/Eigen"
    "/home/mrtheamir/Desktop/libdl/extern/eigen-git-mirror/Eigen/Eigenvalues"
    "/home/mrtheamir/Desktop/libdl/extern/eigen-git-mirror/Eigen/Geometry"
    "/home/mrtheamir/Desktop/libdl/extern/eigen-git-mirror/Eigen/Householder"
    "/home/mrtheamir/Desktop/libdl/extern/eigen-git-mirror/Eigen/IterativeLinearSolvers"
    "/home/mrtheamir/Desktop/libdl/extern/eigen-git-mirror/Eigen/Jacobi"
    "/home/mrtheamir/Desktop/libdl/extern/eigen-git-mirror/Eigen/KLUSupport"
    "/home/mrtheamir/Desktop/libdl/extern/eigen-git-mirror/Eigen/LU"
    "/home/mrtheamir/Desktop/libdl/extern/eigen-git-mirror/Eigen/MetisSupport"
    "/home/mrtheamir/Desktop/libdl/extern/eigen-git-mirror/Eigen/OrderingMethods"
    "/home/mrtheamir/Desktop/libdl/extern/eigen-git-mirror/Eigen/PaStiXSupport"
    "/home/mrtheamir/Desktop/libdl/extern/eigen-git-mirror/Eigen/PardisoSupport"
    "/home/mrtheamir/Desktop/libdl/extern/eigen-git-mirror/Eigen/QR"
    "/home/mrtheamir/Desktop/libdl/extern/eigen-git-mirror/Eigen/QtAlignedMalloc"
    "/home/mrtheamir/Desktop/libdl/extern/eigen-git-mirror/Eigen/SPQRSupport"
    "/home/mrtheamir/Desktop/libdl/extern/eigen-git-mirror/Eigen/SVD"
    "/home/mrtheamir/Desktop/libdl/extern/eigen-git-mirror/Eigen/Sparse"
    "/home/mrtheamir/Desktop/libdl/extern/eigen-git-mirror/Eigen/SparseCholesky"
    "/home/mrtheamir/Desktop/libdl/extern/eigen-git-mirror/Eigen/SparseCore"
    "/home/mrtheamir/Desktop/libdl/extern/eigen-git-mirror/Eigen/SparseLU"
    "/home/mrtheamir/Desktop/libdl/extern/eigen-git-mirror/Eigen/SparseQR"
    "/home/mrtheamir/Desktop/libdl/extern/eigen-git-mirror/Eigen/StdDeque"
    "/home/mrtheamir/Desktop/libdl/extern/eigen-git-mirror/Eigen/StdList"
    "/home/mrtheamir/Desktop/libdl/extern/eigen-git-mirror/Eigen/StdVector"
    "/home/mrtheamir/Desktop/libdl/extern/eigen-git-mirror/Eigen/SuperLUSupport"
    "/home/mrtheamir/Desktop/libdl/extern/eigen-git-mirror/Eigen/UmfPackSupport"
    )
endif()

if("x${CMAKE_INSTALL_COMPONENT}x" STREQUAL "xDevelx" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/include/eigen3/Eigen" TYPE DIRECTORY FILES "/home/mrtheamir/Desktop/libdl/extern/eigen-git-mirror/Eigen/src" FILES_MATCHING REGEX "/[^/]*\\.h$")
endif()

