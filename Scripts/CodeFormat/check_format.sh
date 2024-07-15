#!/usr/bin/env bash

SOURCE_COMMIT="$1"
if [ "$#" -gt 0 ]; then
    shift
fi

# If no source commit is given target the default branch
if [ "x$SOURCE_COMMIT" = "x" ]; then
    # If remote is not set use the remote of the current branch or fallback to "origin"
    if [ "x$REMOTE" = "x" ]; then
        BRANCH="$(git rev-parse --abbrev-ref HEAD)"
        REMOTE="$(git config --local --get "branch.$BRANCH.remote" || echo 'origin')"
    fi
    SOURCE_COMMIT="remotes/$REMOTE/HEAD"
fi

# Force colored diff output
DIFF_COLOR_SAVED="$(git config --local --get color.diff)"
if [ "x$DIFF_COLOR_SAVED" != "x" ]; then
    git config --local --replace-all "color.diff" "always"
else
    git config --local --add "color.diff" "always"
fi

scratch="$(mktemp -t check-format.XXXXXXXXXX)"
finish () {
    # Remove temporary file
    rm -rf "$scratch"
    # Restore setting
    if [ "x$DIFF_COLOR_SAVED" != "x" ]; then
        git config --local --replace-all "color.diff" "$DIFF_COLOR_SAVED"
    else
        git config --local --unset "color.diff"
    fi
}
# The trap will be invoked whenever the script exits, even due to a signal, this is a bash only
# feature
trap finish EXIT

CLANG_FORMAT="clang-format"

# Get all staged files with relevant extensions
FILES=$(git diff --cached --name-only --diff-filter=ACM | grep -E '\.(cc|cp|cpp|c\+\+|cxx|cu|cuh|hh|hpp|hxx|hip|vert|frag)$')
if [ -z "$FILES" ]; then
    echo "No files to format."
    exit 0
fi

echo "Running $CLANG_FORMAT --style=file -i $FILES"
$CLANG_FORMAT --style=file -i $FILES

# Check if any files were modified by clang-format
MODIFIED_FILES=$(git diff --name-only --diff-filter=M $FILES)
if [ -z "$MODIFIED_FILES" ]; then
    echo "No formatting changes needed."
    exit 0
fi

# Dump formatting diff and signal failure
printf \
"\033[31m==== FORMATTING VIOLATIONS DETECTED ====\033[0m
run '\033[33m%s --style=file -i %s\033[0m' to apply these formatting changes\n\n" \
"$CLANG_FORMAT" "$FILES"

git diff $FILES
exit 1
