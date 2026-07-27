// MIT License
//
// Copyright (c) 2026 firedoil
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

#ifndef KEM_CONFIG_H
#define KEM_CONFIG_H

#define ALGO_KYBER 1
#define ALGO_AIGIS_ENC 2

#ifndef ALGORITHM
    #define ALGORITHM ALGO_KYBER
#endif

#ifndef PARAM_MODE
    #if ALGORITHM == ALGO_KYBER
        #define PARAM_MODE 3
    #else
        #define PARAM_MODE 4
    #endif
#endif

#if ALGORITHM != ALGO_KYBER && ALGORITHM != ALGO_AIGIS_ENC
    #error "ALGORITHM must be ALGO_KYBER or ALGO_AIGIS_ENC"
#endif

#endif // KEM_CONFIG_H
