#!/bin/bash
# MIT License
#
# Copyright (c) 2024 Advanced Micro Devices, Inc. All rights reserved.
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

set -e

GIT_TOP_LEVEL=$(git rev-parse --show-toplevel)

# Inputs and Defaults
PACKAGE_NAME="${1:-ROCm-SDK-Examples}"
PACKAGE_VERSION="${2:-6.2.0}"
PACKAGE_INSTALL_PREFIX="${3:-/opt/rocm/examples}"
BUILD_DIR="${4:-$GIT_TOP_LEVEL/build}"
DEB_DIR="${5:-$BUILD_DIR/deb}"
RPM_DIR="${6:-$BUILD_DIR/rpm}"

PACKAGE_CONTACT="ROCm Developer Support <rocm-dev.support@amd.com>"
PACKAGE_DESCRIPTION_SUMMARY="A collection of examples for the ROCm software stack"
PACKAGE_HOMEPAGE_URL="https://github.com/ROCm/ROCm-examples"

RPM_BUILD_DIR="$RPM_DIR/BUILD"
RPM_SOURCE_DIR="$RPM_DIR/SOURCES"
RPM_SPEC_DIR="$RPM_DIR/SPECS"
RPM_RPMS_DIR="$RPM_DIR/RPMS"
RPM_SRPM_DIR="$RPM_DIR/SRPMS"

# Directories to be included in the package
SOURCE_DIRS=(
    "AI"
    "Applications"
    "Common"
    "Dockerfiles"
    "External"
    "HIP-Basic"
    "Libraries"
    "LLVM_ASAN"
)

copy_sources() {
    local dest_dir=$1
    mkdir -p $dest_dir

    # Copy source files in root to package
    cp LICENSE.md CMakeLists.txt README.md $dest_dir

    # Copy source directories to package
    for dir in "${SOURCE_DIRS[@]}"; do
        rsync -a --exclude 'build' --exclude '.gitignore' --exclude '*.vcxproj**' --exclude '*.sln' --exclude 'bin' --exclude '*.o' --exclude '*.exe' $dir $dest_dir
    done
}

create_deb_package() {
    local package_dir=$1
    local deb_root="$BUILD_DIR/deb_tmp"
    local install_dir="$deb_root/$PACKAGE_INSTALL_PREFIX"
    mkdir -p "$deb_root/DEBIAN" "$install_dir"

    # Copy the sources to the install directory
    cp -r $package_dir/* $install_dir/

    # Create control file
    local control_file="$deb_root/DEBIAN/control"
    echo "Package: $PACKAGE_NAME" > $control_file
    echo "Version: $PACKAGE_VERSION" >> $control_file
    echo "Architecture: amd64" >> $control_file
    echo "Maintainer: $PACKAGE_CONTACT" >> $control_file
    echo "Description: $PACKAGE_DESCRIPTION_SUMMARY" >> $control_file
    echo "Homepage: $PACKAGE_HOMEPAGE_URL" >> $control_file
    echo "Depends: " >> $control_file
    echo "Section: devel" >> $control_file
    echo "Priority: optional" >> $control_file

    # Build DEB package
    fakeroot dpkg-deb --build $deb_root $DEB_DIR/${PACKAGE_NAME}_${PACKAGE_VERSION}_amd64.deb

    # Cleanup temporary deb package directory
    rm -rf $deb_root
}

create_rpm_package() {
    local package_dir=$1
    local spec_file="$RPM_SPEC_DIR/${PACKAGE_NAME}.spec"
    mkdir -p "$RPM_SOURCE_DIR" "$RPM_BUILD_DIR"

    # Create the spec file
    cat <<EOF >$spec_file
%define _build_id_links none
%global debug_package %{nil}
Name:           $PACKAGE_NAME
Version:        $PACKAGE_VERSION
Release:        1%{?dist}
Summary:        $PACKAGE_DESCRIPTION_SUMMARY
License:        MIT
URL:            $PACKAGE_HOMEPAGE_URL
Source0:        %{name}-%{version}.tar.gz
BuildArch:      x86_64

%description
$PACKAGE_DESCRIPTION_SUMMARY

%prep
%setup -q

%build

%install
mkdir -p %{buildroot}$PACKAGE_INSTALL_PREFIX
cp -r * %{buildroot}$PACKAGE_INSTALL_PREFIX

%files
$PACKAGE_INSTALL_PREFIX

%changelog
EOF

    # Create source tarball
    tar czf $RPM_SOURCE_DIR/${PACKAGE_NAME}-${PACKAGE_VERSION}.tar.gz -C $BUILD_DIR ${PACKAGE_NAME}-${PACKAGE_VERSION}

    # Build the RPM package
    rpmbuild --define "_topdir $RPM_DIR" -ba $spec_file

    # Move the generated RPM file to RPM_DIR and clean up
    find $RPM_RPMS_DIR -name "${PACKAGE_NAME}-${PACKAGE_VERSION}-*.rpm" -exec mv {} $RPM_DIR \;
    rm -rf $RPM_BUILD_DIR $RPM_SOURCE_DIR $RPM_SPEC_DIR $RPM_RPMS_DIR $RPM_SRPM_DIR
}


# Clean up previous build artifacts
rm -rf $BUILD_DIR
mkdir -p $DEB_DIR $RPM_BUILD_DIR $RPM_SOURCE_DIR $RPM_SPEC_DIR $RPM_RPMS_DIR $RPM_SRPM_DIR

pushd $GIT_TOP_LEVEL || exit

# Copy sources to build directory
copy_sources $BUILD_DIR/${PACKAGE_NAME}-${PACKAGE_VERSION}

# Create DEB package
create_deb_package $BUILD_DIR/${PACKAGE_NAME}-${PACKAGE_VERSION}

# Create RPM package
create_rpm_package $BUILD_DIR/${PACKAGE_NAME}-${PACKAGE_VERSION}

popd || exit
