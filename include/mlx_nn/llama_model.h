#pragma once

#include "linear.h"
#include "module.h"

namespace mlx::core::nn
{

    class CustomLayer : public Module
    {
    public:
        int input_dim, output_dim;
        std::shared_ptr<LinearLayer> l1;
        bool with_bias = true;

        CustomLayer() = default;
        CustomLayer(const CustomLayer &) = default;
        CustomLayer(int in_features, int out_features, bool _with_bias = true);

        ~CustomLayer() = default;

        array forward(const array &input);
    };

    class TestModel : public Module
    {
    public:
        std::shared_ptr<LinearLayer> fc1;
        std::shared_ptr<CustomLayer> fc2;
        std::vector<std::shared_ptr<CustomLayer>> layers{};

        TestModel();
        array forward(const array &x);
    };

} // namespace mlx::core::nn
