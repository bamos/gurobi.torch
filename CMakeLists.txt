CMAKE_MINIMUM_REQUIRED(VERSION 2.8 FATAL_ERROR)
CMAKE_POLICY(VERSION 2.8)

SET(CMAKE_MODULE_PATH "${CMAKE_CURRENT_SOURCE_DIR}/cmake"
                      "${CMAKE_MODULE_PATH}")

FIND_PACKAGE(Torch REQUIRED)

include(cmake/USE_C99.cmake)
USE_C99()
find_package(OpenMP)
set (CMAKE_C_FLAGS "-Wall -Wno-unused-variable ${OpenMP_C_FLAGS} ${CMAKE_C_FLAGS}")

FILE(GLOB luasrc *.lua)
FILE(GLOB src *.c)

if(NOT DEFINED ENV{GUROBI_HOME})
  message(FATAL_ERROR "GUROBI_HOME environment variable not defined.")
endif()

INCLUDE_DIRECTORIES(
  GLOB "$ENV{GUROBI_HOME}/include" *.h
)

ADD_TORCH_PACKAGE(gurobi "${src}" "${luasrc}")
