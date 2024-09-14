#pragma once

#include <iostream>
#include <map>
#include <memory>
#include <sstream>
#include <unordered_map>
#include <vector>
#include "mlx/mlx.h"
#include "utils.h"
#include <string>

namespace mlx::core::nn
{
    class Module
    {
    public:
        std::unordered_map<std::string, array> parameters{};
        std::unordered_map<std::string, array> buffers{};
        std::unordered_map<std::string, std::shared_ptr<Module>> submodules{};
        std::unordered_map<std::string, array& > named_parameters_dict{};

        std::string name;
        StreamOrDevice device = metal::is_available() ? Device::gpu : Device::cpu;

        Module() {};
        Module(const Module &) = default;
        // Module& operator=(Module&&) noexcept = default;
        //  Module(Module&&) noexcept = default;

        virtual ~Module() = default;

        array forward(const array &input);

        array &register_parameter(std::string name, array wb);

        array &register_parameter(std::string name, array &&wb);
        
        bool hasEnding(const std::string &str, const std::string &suffix);
       
        array &register_buffer(std::string name, array &wb);
       
        template <typename T>
        void register_module(std::string sub_name, T &m);

        // template <typename T>
        // void register_module(std::string sub_name, std::shared_ptr<T> m);

        template <typename T>
        void register_module(std::string sub_name, std::shared_ptr<T> &m);

        template <typename T>
        void register_layer(std::string layers_name, std::vector<T> &layers);

        // template <typename T>
        // void register_layer(std::string layers_name, std::vector<std::shared_ptr<T>> layers);

        void named_parameters(std::string prelimiter = "");

        void update(std::unordered_map<std::string, array> trained_weights);

        void load_from_safetensors(const std::string &file, StreamOrDevice s);

        void load_from_gguf(const std::string &file, StreamOrDevice s);

        void load_weights(
            const std::string &file,
            StreamOrDevice s = metal::is_available() ? Device::gpu : Device::cpu);
        
        void print_parameters();
    };

    template <typename T>
    void Module::register_module(std::string sub_name, T &m)
    {
        // `register_component` allows you to register the component(in order) as
        // used by the NN
        if (!std::is_base_of<T, Module>::value)
        {
            // Error the code is not correct
        }

        // Check if the submodule exists before trying to access it
        if (!(submodules.find(sub_name) != submodules.end()))
        {
            // Add the parameters of the submodules to the named_parameters_dict
            //   m.name = sub_name;
            //   m.named_parameters();
            submodules.insert({sub_name, m});
            submodules.at(sub_name)->name = sub_name;
            submodules.at(sub_name)->named_parameters();
        }
        else
        {
            // Handle the case where the submodule doesn't exist
        }
    }

    // template <typename T>
    // void Module::register_module(std::string sub_name, std::shared_ptr<T> m)
    // {
    //     // `register_component` allows you to register the component(in order) as
    //     // used by the NN
    //     if (!std::is_base_of<T, Module>::value)
    //     {
    //         // Error the code is not correct
    //     }

    //     // Check if the submodule exists before trying to access it
    //     if (!(submodules.find(sub_name) != submodules.end()))
    //     {
    //         // Add the parameters of the submodules to the named_parameters_dict
    //         // m.name = sub_name;
    //         // m.named_parameters();
    //         submodules.insert({sub_name, m});
    //         submodules.at(sub_name)->name = sub_name;
    //         submodules.at(sub_name)->named_parameters();
    //     }
    //     else
    //     {
    //         // Handle the case where the submodule doesn't exist
    //     }
    // }

    template <typename T>
    void Module::register_module(std::string sub_name, std::shared_ptr<T> &m)
    {
        // `register_component` allows you to register the component(in order) as
        // used by the NN
        if (!std::is_base_of<T, Module>::value)
        {
            // Error the code is not correct
        }

        // Check if the submodule exists before trying to access it
        if (!(submodules.find(sub_name) != submodules.end()))
        {
            // Add the parameters of the submodules to the named_parameters_dict
            // m.name = sub_name;
            // m.named_parameters();
            submodules.insert({sub_name, m});
            submodules.at(sub_name)->name = sub_name;
            submodules.at(sub_name)->named_parameters();
        }
        else
        {
            // Handle the case where the submodule doesn't exist
        }
    }

    template <typename T>
    void Module::register_layer(std::string layers_name, std::vector<T> &layers)
    {
        // `register_component` allows you to register the layers(in order) as
        // used by the NN
        if (!std::is_base_of<T, Module>::value)
        {
            // Error the code is not correct
        }
        for (size_t i = 0; i < layers.size(); i++)
        {
            register_module(get_name(layers_name, i), layers[i]);
        }
    }

    // template <typename T>
    // void Module::register_layer(std::string layers_name, std::vector<std::shared_ptr<T>> layers)
    // {
    //     // `register_component` allows you to register the layers(in order) as
    //     // used by the NN
    //     if (!std::is_base_of<T, Module>::value)
    //     {
    //         // Error the code is not correct
    //     }
    //     for (size_t i = 0; i < layers.size(); i++)
    //     {
    //         register_module(get_name(layers_name, i), layers[i]);
    //     }
    // }

}; // namespace mlx::core::nn