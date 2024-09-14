// Utils for mlx_nn.cpp
#pragma once

#include <iostream>
#include <string>
#include <sstream>
#include "mlx/mlx.h"

namespace mlx::core::nn
{
    // Base template for get_name
    template <typename T>
    std::string get_name(const std::string &prelimiter, const T &value);
    
    inline std::string get_name(const std::string &prelimiter, const std::string &value1, const std::string &value2);

} // namespace mlx::core::nn