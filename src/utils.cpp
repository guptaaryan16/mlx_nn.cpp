#include "mlx_nn/utils.h"

namespace mlx::core::nn
{
    template <typename T>
    std::string get_name(const std::string &prelimiter, const T &value)
    {
        std::ostringstream oss;
        if (prelimiter.empty())
        {
            oss << value;
        }
        else
        {
            oss << prelimiter << "." << value;
        }
        return oss.str();
    }

    // Specialization for two string parameters
    inline std::string get_name(const std::string &prelimiter, const std::string &value)
    {
        return get_name<std::string>(prelimiter, value);
    }

    // Specialization for three string parameters
    inline std::string get_name(const std::string &prelimiter, const std::string &value1, const std::string &value2)
    {
        std::ostringstream oss;
        if (prelimiter.empty())
        {
            oss << value1 << "." << value2;
        }
        else
        {
            oss << prelimiter << "." << value1 << "." << value2;
        }
        return oss.str();
    }

    // Explicit instantiations for common types
    template std::string get_name<std::string>(const std::string &, const std::string &);
    template std::string get_name<int>(const std::string &, const int &);
    template std::string get_name<size_t>(const std::string &, const size_t &);

} // namespace mlx::core::nn