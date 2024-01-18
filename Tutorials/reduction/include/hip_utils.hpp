// MIT License
//
// Copyright (c) 2024 Advanced Micro Devices, Inc. All rights reserved.
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

#pragma once

// HIP API
#include <hip/hip_runtime.h> // hipGetErrorString

// STL
#include <cstdlib> // std::exit
#include <iostream> // std::cerr, std::endl

namespace hip
{
void check(hipError_t err, const char* name)
{
    if(err != hipSuccess)
    {
        std::cerr << name << "(" << hipGetErrorString(err) << ")" << std::endl;
        std::exit(err);
    }
}

template<class T, size_t Size>
class static_array
{
public:
    using value_type      = T;
    using size_type       = size_t;
    using reference       = T&;
    using const_reference = const T&;

    [[nodiscard]] __device__ __host__ constexpr size_type size() const noexcept
    {
        return Size;
    }
    [[nodiscard]] __device__ __host__ constexpr size_type max_size() const noexcept
    {
        return Size;
    }
    [[nodiscard]] __device__ __host__ constexpr bool empty() const noexcept
    {
        return false;
    }
    [[nodiscard]] __device__ __host__ constexpr reference front() noexcept
    {
        return _elems[0];
    }
    [[nodiscard]] __device__ __host__ constexpr const_reference front() const noexcept
    {
        return _elems[0];
    }
    [[nodiscard]] __device__ __host__ constexpr reference back() noexcept
    {
        return _elems[Size - 1];
    }
    [[nodiscard]] __device__ __host__ constexpr const_reference back() const noexcept
    {
        return _elems[Size - 1];
    }

    T _elems[Size];
};

template<size_t I, class T, size_t Size>
[[nodiscard]] constexpr T& get(static_array<T, Size>& arr) noexcept
{
    static_assert(I < Size, "array index out of bounds");
    return arr._elems[I];
}

template<size_t I, class T, size_t Size>
[[nodiscard]] constexpr const T& get(const static_array<T, Size>& arr) noexcept
{
    static_assert(I < Size, "array index out of bounds");
    return arr._elems[I];
}

template<size_t I, class T, size_t Size>
[[nodiscard]] constexpr T&& get(static_array<T, Size>&& arr) noexcept
{
    static_assert(I < Size, "array index out of bounds");
    return arr._elems[I];
}

template<size_t I, class T, size_t Size>
[[nodiscard]] constexpr const T&& get(const static_array<T, Size>&& arr) noexcept
{
    static_assert(I < Size, "array index out of bounds");
    return arr._elems[I];
}
} // namespace hip

#define HIP_CHECK(call) ::hip::check((call), #call)
