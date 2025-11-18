#!/bin/bash

# This AWK script performs an indentation check on all Makefiles under the current working directory.
# 
# High-level pattern matching logic:
# 1. Detects the start of a recipe block by matching lines that starts with non-space character and contains : thats not followed by =.
# 2. Within a recipe block, lines that begin with spaces are INVALID.
# 3. If a non-indented line appears, it's the end of the recipe block.
# 4. Outside of recipe blocks, any line that starts with a TAB is INVALID (only recipe lines should begin with TAB).
# 
# Prints out invalid lines

fail=0

while IFS= read -r file; do
  awk '
    /^[^[:space:]]+:\s*($|[^=])/ {
      in_recipe=1
      next
    }
    in_recipe == 1 && /^[ ]+/ {
      print FILENAME ":" NR ": INVALID - Recipe line indented with spaces, should use TAB:" $0
      fail=1
      next
    }
    in_recipe == 1 && /^\t/ {
      next
    }
    in_recipe == 1 && /^[^[:space:]]/ {
      in_recipe=0
    }
    in_recipe == 0 && /^\t/ {
      print FILENAME ":" NR ": INVALID - Non-recipe line indented with TAB:" $0
      fail=1
      next
    }
  ' "$file"
done < <(find . -name Makefile)

exit $fail