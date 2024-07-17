#!/bin/bash
# MIT License
#
# Copyright (c) 2022-2024 Advanced Micro Devices, Inc. All rights reserved.
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

PACKAGE_NAME="ROCm-SDK-Examples"
PACKAGE_VERSION="6.2.0"
PACKAGE_VENDOR="Advanced Micro Devices, Inc."
PACKAGE_CONTACT="ROCm Developer Support <rocm-dev.support@amd.com>"
PACKAGE_DESCRIPTION_SUMMARY="A collection of examples for the ROCm software stack"
PACKAGE_INSTALL_PREFIX="/opt/rocm/examples"

BUILD_DIR="build"
DEB_DIR="$BUILD_DIR/deb"
RPM_DIR="$BUILD_DIR/rpm"
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

# Clean up previous build artifacts
rm -rf $BUILD_DIR
mkdir -p $DEB_DIR $RPM_BUILD_DIR $RPM_SOURCE_DIR $RPM_SPEC_DIR $RPM_RPMS_DIR $RPM_SRPM_DIR

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
    local control_file="$package_dir/DEBIAN/control"
    mkdir -p "$(dirname $control_file)"

    # Create control file
    echo "Package: $PACKAGE_NAME" > $control_file
    echo "Version: $PACKAGE_VERSION" >> $control_file
    echo "Architecture: amd64" >> $control_file
    echo "Maintainer: $PACKAGE_CONTACT" >> $control_file
    echo "Description: $PACKAGE_DESCRIPTION_SUMMARY" >> $control_file
    echo "Homepage: https://github.com/ROCm/ROCm-examples" >> $control_file
    echo "Depends: " >> $control_file
    echo "Section: devel" >> $control_file
    echo "Priority: optional" >> $control_file

    # Build DEB package
    fakeroot dpkg-deb --build $package_dir $DEB_DIR/${PACKAGE_NAME}_${PACKAGE_VERSION}_amd64.deb
}

create_rpm_package() {
    local package_dir=$1
    local spec_file="$RPM_SPEC_DIR/${PACKAGE_NAME}.spec"
    mkdir -p "$(dirname $spec_file)"

    # Create spec file
    echo "Name: $PACKAGE_NAME" > $spec_file
    echo "Version: $PACKAGE_VERSION" >> $spec_file
    echo "Release: 1" >> $spec_file
    echo "Summary: $PACKAGE_DESCRIPTION_SUMMARY" >> $spec_file
    echo "Group: Development/Tools" >> $spec_file
    echo "License: MIT" >> $spec_file
    echo "URL: https://github.com/ROCm/ROCm-examples" >> $spec_file
    echo "Source0: %{name}-%{version}.tar.gz" >> $spec_file
    echo "%description" >> $spec_file
    echo "$PACKAGE_DESCRIPTION_SUMMARY" >> $spec_file
    echo "%prep" >> $spec_file
    echo "%setup -q" >> $spec_file
    echo "%build" >> $spec_file
    echo "%install" >> $spec_file
    echo "mkdir -p %{buildroot}$PACKAGE_INSTALL_PREFIX" >> $spec_file
    echo "cp -a * %{buildroot}$PACKAGE_INSTALL_PREFIX" >> $spec_file
    echo "%files" >> $spec_file
    echo "$PACKAGE_INSTALL_PREFIX" >> $spec_file

    # Build RPM package
    tar -czf $RPM_SOURCE_DIR/${PACKAGE_NAME}-${PACKAGE_VERSION}.tar.gz -C $package_dir .
    rpmbuild -ba $spec_file --define "_topdir $RPM_DIR"
    mv $RPM_RPMS_DIR/x86_64/${PACKAGE_NAME}-${PACKAGE_VERSION}-1.x86_64.rpm $RPM_DIR/${PACKAGE_NAME}_${PACKAGE_VERSION}_amd64.rpm
}

# Copy sources to build directory
copy_sources $BUILD_DIR/$PACKAGE_NAME

# Create DEB package
create_deb_package $BUILD_DIR/$PACKAGE_NAME

# Create RPM package
create_rpm_package $BUILD_DIR/$PACKAGE_NAME
