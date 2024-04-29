// MIT License
//
// Copyright (c) 2022-2023 Advanced Micro Devices, Inc. All rights reserved.
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

#include "example_utils.hpp"

#include <thrust/generate.h>
#include <thrust/host_vector.h>
#include <thrust/random.h>
#include <thrust/remove.h>

// An anonymous namespace sets static linkage to its contents.
// This means that the contained function definitions will only be visible
// in the current compilation unit (i.e. cpp source file).
namespace
{

/// \brief Test whether the point is outside of the unit circle
template<typename T>
struct is_outside_circle
{
    template<typename Tuple>
    inline __host__ __device__ bool operator()(const Tuple& tuple) const
    {
        // Unpack the tuple into x and y coordinates
        const T x = thrust::get<0>(tuple);
        const T y = thrust::get<1>(tuple);

        return x * x + y * y > 1;
    }
};

} // namespace

int main()
{
    // Number of points to generate
    constexpr size_t N = 20;

    // Generate random points in the unit square on the host
    thrust::default_random_engine            rng;
    thrust::uniform_real_distribution<float> u01(0.0f, 1.0f);

    auto get_coordinate = [&rng, &u01] { return u01(rng); };

    // Vectors of coordinates of points
    thrust::host_vector<float> x(N);
    thrust::host_vector<float> y(N);
    thrust::generate(x.begin(), x.end(), get_coordinate);
    thrust::generate(y.begin(), y.end(), get_coordinate);

    // Print the initial points
    std::cout << std::fixed; // set fixed-point formatting of floats
    std::cout << "Generated " << N << " points" << std::endl
              << format_pairs(x.begin(), x.end(), y.begin(), y.end()) << std::endl
              << std::endl;

    // Remove points where x^2 + y^2 > 1 and determine new array sizes
    size_t new_size
        = thrust::remove_if(thrust::make_zip_iterator(thrust::make_tuple(x.begin(), y.begin())),
                            thrust::make_zip_iterator(thrust::make_tuple(x.end(), y.end())),
                            is_outside_circle<float>())
          - thrust::make_zip_iterator(thrust::make_tuple(x.begin(), y.begin()));

    // Resize the vectors (note: this does not free any memory)
    x.resize(new_size);
    y.resize(new_size);

    // Print the filtered points
    std::cout << "After removal of points outside of the unit circle, " << new_size
              << " points remain" << std::endl
              << format_pairs(x.begin(), x.end(), y.begin(), y.end()) << std::endl;
}
