

# ls -l upcoming_changes/**/908.*.rst
# ls -l docs/source/whats_new/upcoming_changes/**/909.*.rst

## Preview the Changes (Dry Run)
# find docs/source/whats_new/upcoming_changes/ -type f -name "908.*.rst" | while read -r filepath; do
#     dir=$(dirname "$filepath")
#     file=$(basename "$filepath")
#     echo "mv -- \"$filepath\" \"$dir/909.${file#908.}\""
# done

## Execute the Recursive Rename
# find docs/source/whats_new/upcoming_changes/ -type f -name "908.*.rst" | while read -r filepath; do
#     dir=$(dirname "$filepath")
#     file=$(basename "$filepath")
#     mv -- "$filepath" "$dir/909.${file#908.}"
# done
