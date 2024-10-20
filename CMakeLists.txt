# Указываем минимальную версию CMake
cmake_minimum_required(VERSION 3.10)

# Указываем имя проекта
project(FashionMNIST)

# Указываем стандарт C++
set(CMAKE_CXX_STANDARD 11)
set(CMAKE_CXX_STANDARD_REQUIRED True)

# Указываем исполняемый файл
add_executable(fashion_mnist main.cpp)  # Замените main.cpp на имя вашего файла с кодом
