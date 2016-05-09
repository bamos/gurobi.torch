CMAKE_MINIMUM_REQUIRED(VERSION 2.8 FATAL_ERROR)
CMAKE_POLICY(VERSION 2.8)

SET(CMAKE_MODULE_PATH "${CMAKE_CURRENT_SOURCE_DIR}/cmake"
                      "${CMAKE_MODULE_PATH}")

FIND_PACKAGE(Torch REQUIRED)

include(cmake/USE_C99.cmake)
USE_C99()
set (CMAKE_C_FLAGS "-Wall -Wno-unused-variable ${CMAKE_C_FLAGS}")

set (GUROBI_HOME "$ENV{HOME}/src/gurobi650/linux64")

FILE(GLOB luasrc *.lua)
FILE(GLOB src *.c)

INCLUDE_DIRECTORIES(
  GLOB "${GUROBI_HOME}/include" *.h
)

ADD_TORCH_PACKAGE(gurobi "${src}" "${luasrc}")
