cmake_minimum_required(VERSION 3.10)
project(SqueezeCNN)

set(CMAKE_CXX_STANDARD 11)

add_executable(SqueezeCNN main.cpp network/core_math.h network/utils.h network/network.h network/activation.h)