// MIT License
//
// Copyright (c) 2022 Advanced Micro Devices, Inc. All rights reserved.
//
// Permission is hereby granted, free of charge, to any person obtaining a copy
// of this software and associated documentation files (the "Software"), to deal
// in the Software without restriction, including without limitation the rights
// to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
// copies of the Software, and to permit persons to whom the Software is
// furnished to do so, subject to the following conditions:
//
// The above copyright notice and this permission notice shall be included in all
// copies or substantial portions of the Software.
//
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
// IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
// FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
// AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
// LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
// SOFTWARE.

#ifndef SIMPLE_DISTRIBUTIONS_CPP_ARGUMENT_PARSING_HPP
#define SIMPLE_DISTRIBUTIONS_CPP_ARGUMENT_PARSING_HPP

#include <algorithm>
#include <charconv>
#include <iostream>
#include <optional>
#include <string_view>

// Needed for the 's' suffix of `std::string` literals.
using namespace std::string_literals;

/// \brief The random distribution kind selected on the command line.
enum class Distribution
{
    uniform_int,
    uniform_real,
    normal,
    poisson
};

/// \brief The set of arguments parsed from the command line.
struct CliArguments
{
    int          device_id_;
    Distribution distribution_;
    size_t       size_;
    bool         print_;
};

/// \brief Operator overload to simply print a \p CliArguments instance.
std::ostream& operator<<(std::ostream& os, const CliArguments& cli_args)
{
    // An immediately-invoked lambda expression selects the name of the distribution.
    const std::string_view distribution_name = [&]()
    {
        switch(cli_args.distribution_)
        {
            case Distribution::uniform_int: return "uniform_int";
            case Distribution::uniform_real: return "uniform_real";
            case Distribution::normal: return "normal";
            case Distribution::poisson: return "poisson";
            default: return "unknown";
        }
    }();

    // Printing the fields to the `std::ostream` object.
    return os << "Selected device id: " << cli_args.device_id_
              << "\nSelected distribution: " << distribution_name
              << "\nSelected size: " << cli_args.size_ << "\nPrinting results: " << std::boolalpha
              << cli_args.print_;
}

/// \brief Converts a \p std::string_view to integral type \p T.
// Throws an exception with an error message if the conversion is unsuccessful.
template<typename T>
T parse_integral_arg(const std::string_view arg_value)
{
    T value;
    // Try to convert the string_view to an integral type. If successful, the value is written to
    // the variable `value`
    const auto conversion_result
        = std::from_chars(arg_value.data(), arg_value.data() + arg_value.size(), value);
    // The default constructed `std::errc` stands for successful conversion.
    if(conversion_result.ec != std::errc{})
    {
        throw std::runtime_error(
            "Could not convert argument \""s.append(arg_value).append("\" to an integral value"));
    }
    return value;
}

/// \brief Parses an \p std::string_view to a \p Distribution.
/// Throws an exception with an error message if the conversion is unsuccessful.
Distribution parse_distribution_arg(const std::string_view distribution_arg)
{
    if(distribution_arg == "uniform_int")
    {
        return Distribution::uniform_int;
    }
    if(distribution_arg == "uniform_real")
    {
        return Distribution::uniform_real;
    }
    if(distribution_arg == "normal")
    {
        return Distribution::normal;
    }
    if(distribution_arg == "poisson")
    {
        return Distribution::poisson;
    }
    throw std::runtime_error(
        "Argument \""s.append(distribution_arg).append("\" is not a valid distribution"));
}

/// \brief Parses the array of command line arguments to parameters consumed by the rest
/// of the program. \p argc must be set to the size of the \p argv array. Each pointer in
/// the \p argv array must point to a valid null-terminated string containing the argument.
CliArguments parse_args(const int argc, const char** argv)
{
    // Pointers fulfill the random access iterator traits, thereby can be used with the
    // standard algorithms.
    const char** argv_end = argv + argc;

    // This local function searches for `arg_name` in the argument array and returns true if found.
    const auto find_argument = [&](const std::string_view arg_name)
    {
        const auto arg_name_it = std::find(argv, argv_end, arg_name);
        return arg_name_it != argv_end;
    };

    // This local function searches for `arg_name` in the argument array. If found, it returns a pointer
    // to the next argument -- that is assumed to be the provided value. Otherwise returns a null optional.
    // If the found argument is the last one, an exception with an error message is thrown.
    const auto find_argument_value
        = [&](const std::string_view arg_name) -> std::optional<std::string_view>
    {
        const auto arg_name_it = std::find(argv, argv_end, arg_name);
        if(arg_name_it == argv_end)
        {
            return std::nullopt;
        }
        // std::next returns the iterator copied and advanced by one
        const auto arg_value_it = std::next(arg_name_it);
        if(arg_value_it == argv_end)
        {
            throw std::runtime_error("Value for argument is not supplied: "s.append(arg_name));
        }
        return std::make_optional(*arg_value_it);
    };

    // The options below need provided values, thereby `find_argument_value` is used.
    const auto device_arg       = find_argument_value("--device").value_or("0");
    const auto distribution_arg = find_argument_value("--distribution").value_or("uniform_int");
    const auto size_arg         = find_argument_value("--size").value_or("10000000");

    // The option below is just a flag. Its existence is checked by `find_argument`.
    const bool print_arg = find_argument("--print");

    // Parse the arguments read to the corresponding type and return.
    return {parse_integral_arg<int>(device_arg),
            parse_distribution_arg(distribution_arg),
            parse_integral_arg<size_t>(size_arg),
            print_arg};
}

constexpr std::string_view cli_usage_message
    = "Usage: simple_distributions_cpp [--device <device_id>] [--distribution "
      "{uniform_int|uniform_real|normal|poisson}] [--size <size>] [--print]";

#endif // SIMPLE_DISTRIBUTIONS_CPP_ARGUMENT_PARSING_HPP
