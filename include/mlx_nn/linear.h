#pragma once
#include "module.h"

namespace mlx::core::nn
{
    class LinearLayer : public Module
    {
    public:
        int input_dim, output_dim;
        bool with_bias = true;

        LinearLayer() = default;
        LinearLayer(const LinearLayer &) = default;
        LinearLayer(int in_features, int out_features, bool _with_bias = true);
        ~LinearLayer() = default;

        array forward(const array &input);
    };

} // namespace
