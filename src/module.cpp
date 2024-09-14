#include "mlx_nn/module.h"

namespace mlx::core::nn
{

array& Module::register_parameter(std::string name, array wb)
{
    // `register_parameter` allows you to register the Weights & Biases
    // used by the NN
    parameters.insert({name, wb});
    return parameters.at(name);
}

array& Module::register_parameter(std::string name, array &&wb)
{
    // `register_parameter` allows you to register the Weights & Biases
    // used by the NN
    parameters.insert({name, wb});
    return parameters.at(name);
}

bool Module::hasEnding(const std::string &str, const std::string &suffix)
{
    if (suffix.size() > str.size())
        return false;
    return str.substr(str.size() - suffix.size()) == suffix;
}

array& Module::register_buffer(std::string name, array &wb)
{
    // `register_parameter` allows you to register the Weights & Biases
    // used by the NN
    buffers.insert({name, wb});
    return buffers.at(name);
}


// Forward method for all submodules
// TODO:: Make A general method for all forward implementations
array Module::forward(const array &input)
{
    return input;
};

void Module::named_parameters(std::string prelimiter)
{
    for (auto &[k, v] : parameters)
    {
        std::string sub_name = get_name(get_name(prelimiter, name), k);
        named_parameters_dict.insert({sub_name, v});
    }
    for (auto &[k, v] : buffers)
    {
        std::string sub_name = get_name(get_name(prelimiter, name), k);
        named_parameters_dict.insert({sub_name, v});
    }
    if (!submodules.empty())
    {
        for (auto &[k, v] : submodules)
        {
            for (auto &[l, m] : v->named_parameters_dict)
            {
                std::string sub_name = get_name(name, l);
                if (!(hasEnding(sub_name, ".")))
                {
                    named_parameters_dict.insert({sub_name, m});
                }
            }
        }
    }
}

void Module::update(std::unordered_map<std::string, array> trained_weights)
{
    // Create references for all the known parameters
    this->named_parameters();

    for (auto &[k, v] : trained_weights)
    {
        if (!(named_parameters_dict.find(k) != named_parameters_dict.end()))
        {
            std::cout << "Named parameter does not contain the key: " << k << "\n";
        }
        else if (named_parameters_dict.at(k).shape() != v.shape())
        {
            std::cout << "There is a shape difference for : " << k << "->"
                        << named_parameters_dict.at(k).shape() << " and " << v.shape()
                        << std::endl;
        }
        else
        {
            named_parameters_dict.at(k) = v;
        }
    }
}

void Module::load_from_safetensors(const std::string &file, StreamOrDevice s)
{
    SafetensorsLoad loaded_weights = load_safetensors(file, s);
    update(loaded_weights.first);
}

void Module::load_from_gguf(const std::string &file, StreamOrDevice s)
{
    GGUFLoad loaded_weights = load_gguf(file, s);
    update(loaded_weights.first);
}

void Module::load_weights(
    const std::string &file,
    StreamOrDevice s)
{
    if (hasEnding(file, ".safetensors"))
    {
        std::cout << "Loading model from .safetensors file...\n";
        load_from_safetensors(file, s);
    }
    else if (hasEnding(file, ".gguf"))
    {
        load_from_gguf(file, s);
        std::cout << "Loading model from .gguf file...\n";
    }
    else
    {
        std::cout << "Model file format is not supported...\n";
    }
}

void Module::print_parameters()
{
    this->named_parameters();
    std::cout << "\n[\nparameters:\n";
    for (auto &[k, v] : named_parameters_dict)
    {
        std::cout << k << ":\n"
                    << v << "\n";
    }
    std::cout << "]\n";
}

} // namespace mlx::core::nn