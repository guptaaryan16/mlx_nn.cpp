#include "mlx_nn/llama_model.h"

namespace mlx::core::nn
{

CustomLayer::CustomLayer(int in_features, int out_features, bool _with_bias)
{
    input_dim = in_features;
    output_dim = out_features;
    array weight = random::normal({in_features, out_features}, float32);
    array bias = random::normal({out_features}, float32);
    l1 = std::make_shared<LinearLayer>(out_features, out_features);

    register_parameter("weight", weight);
    register_parameter("bias", bias);
    register_module("l1", l1);
    with_bias = _with_bias;
}


array CustomLayer::forward(const array &input)
{
    // Check if input size matches number of weights in first layer
    if (input.shape(-1) != parameters.at("weight").shape(0))
    {
        throw std::invalid_argument(
            "Input size doesn't match weight vector size");
    }
    // Allocate space for the outputs
    array outputs = matmul(input, parameters.at("weight"), device);

    auto y = with_bias ? (outputs + parameters.at("bias")) : outputs;
    return l1->forward(y);
}


TestModel::TestModel()
{
    // Declare your layers only inside the constuctor to avoid "undefined"
    // behaviour
    fc1 = std::make_shared<LinearLayer>(784, 100, false);
    fc2 = std::make_shared<CustomLayer>(100, 10, false);

    for (int i = 0; i < 3; i++)
    {
        layers.push_back(std::make_shared<CustomLayer>(10, 10));
    }

    register_module("fc1", fc1);
    register_module("fc2", fc2);
    register_layer("layers", layers);
}
array TestModel::forward(const array &x)
{
    auto y = fc2->forward(fc1->forward(x));
    for (auto &l : layers)
    {
        y = l->forward(y);
    }
    return y;
}

} // namespace mlx::core::nn