# MIT License
#
# Copyright (c) 2026 firedoil
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

GPU_RUNTIME ?= HIP
ROCM_PATH ?= /opt/rocm
HIPCXX ?= $(ROCM_PATH)/bin/hipcc

CXX_STD := c++17
ICXXFLAGS := -std=$(CXX_STD)
ICPPFLAGS :=
ILDFLAGS :=
ILDLIBS :=

ifneq ($(GPU_RUNTIME), HIP)
  $(error This example currently supports only the AMD HIP platform)
endif

ifneq ($(HIP_PLATFORM),)
  ifneq ($(HIP_PLATFORM), amd)
    $(error This example currently supports only the AMD HIP platform)
  endif
endif

ICXXFLAGS += $(CXXFLAGS)
ICPPFLAGS += $(CPPFLAGS)
ILDFLAGS += $(LDFLAGS)
ILDLIBS += $(LDLIBS)

DEFAULT_EXAMPLES := \
  applications_pqc_kyber768 \
  applications_pqc_aigis_enc3 \
  applications_pqc_mldsa65 \
  applications_pqc_aigis_sig2

KEM_EXAMPLES := \
  applications_pqc_kyber512 \
  applications_pqc_kyber768 \
  applications_pqc_kyber1024 \
  applications_pqc_aigis_enc1 \
  applications_pqc_aigis_enc2 \
  applications_pqc_aigis_enc3 \
  applications_pqc_aigis_enc4

SIG_EXAMPLES := \
  applications_pqc_mldsa44 \
  applications_pqc_mldsa65 \
  applications_pqc_mldsa87 \
  applications_pqc_aigis_sig1 \
  applications_pqc_aigis_sig2 \
  applications_pqc_aigis_sig3

ALL_EXAMPLES := $(KEM_EXAMPLES) $(SIG_EXAMPLES)

applications_pqc_kyber512: ALGORITHM=1
applications_pqc_kyber512: PARAM_MODE=2
applications_pqc_kyber768: ALGORITHM=1
applications_pqc_kyber768: PARAM_MODE=3
applications_pqc_kyber1024: ALGORITHM=1
applications_pqc_kyber1024: PARAM_MODE=4
applications_pqc_aigis_enc1: ALGORITHM=2
applications_pqc_aigis_enc1: PARAM_MODE=1
applications_pqc_aigis_enc2: ALGORITHM=2
applications_pqc_aigis_enc2: PARAM_MODE=2
applications_pqc_aigis_enc3: ALGORITHM=2
applications_pqc_aigis_enc3: PARAM_MODE=3
applications_pqc_aigis_enc4: ALGORITHM=2
applications_pqc_aigis_enc4: PARAM_MODE=4
applications_pqc_mldsa44: ALGORITHM=1
applications_pqc_mldsa44: PARAM_MODE=2
applications_pqc_mldsa65: ALGORITHM=1
applications_pqc_mldsa65: PARAM_MODE=3
applications_pqc_mldsa87: ALGORITHM=1
applications_pqc_mldsa87: PARAM_MODE=5
applications_pqc_aigis_sig1: ALGORITHM=2
applications_pqc_aigis_sig1: PARAM_MODE=1
applications_pqc_aigis_sig2: ALGORITHM=2
applications_pqc_aigis_sig2: PARAM_MODE=2
applications_pqc_aigis_sig3: ALGORITHM=2
applications_pqc_aigis_sig3: PARAM_MODE=3

all: $(DEFAULT_EXAMPLES)

all-variants: $(ALL_EXAMPLES)

$(KEM_EXAMPLES): kem_api/main.hip $(wildcard kem_api/*.h) $(wildcard kem_api/*.hpp)
	$(HIPCXX) $(ICXXFLAGS) $(ICPPFLAGS) $(ILDFLAGS) \
		-DALGORITHM=$(ALGORITHM) -DPARAM_MODE=$(PARAM_MODE) \
		-o $@ $< $(ILDLIBS)

$(SIG_EXAMPLES): sig_api/main.hip $(wildcard sig_api/*.h) $(wildcard sig_api/*.hpp)
	$(HIPCXX) $(ICXXFLAGS) $(ICPPFLAGS) $(ILDFLAGS) \
		-DALGORITHM=$(ALGORITHM) -DPARAM_MODE=$(PARAM_MODE) \
		-DBATCH_KEYGEN_SAMPLE_SPLIT_FAST=1 \
		-o $@ $< $(ILDLIBS)

test: $(DEFAULT_EXAMPLES)
	@set -e; for example in $(DEFAULT_EXAMPLES); do ./$$example --self-test; done

clean:
	$(RM) $(ALL_EXAMPLES)

.PHONY: all all-variants clean test
