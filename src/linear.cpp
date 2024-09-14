#include "mlx_nn/linear.h"

namespace mlx::core::nn
{
    
LinearLayer::LinearLayer(int in_features, int out_features, bool _with_bias)
{
    input_dim = in_features;
    output_dim = out_features;
    array weight = random::normal({in_features, out_features}, float32);
    array bias = random::normal({out_features}, float32);
    with_bias = _with_bias;

    register_parameter("weight", weight);
    register_parameter("bias", bias);
}

array LinearLayer::forward(const array &input)
{
    // Check if input size matches number of weights in first layer
    if (input.shape(-1) != parameters.at("weight").shape(0))
    {
        throw std::invalid_argument(
            "Input size doesn't match weight vector size");
    }
    // Allocate space for the outputs
    array outputs = matmul(input, parameters.at("weight"));

    return with_bias ? (outputs + parameters.at("bias")) : outputs;
}

} // namespace