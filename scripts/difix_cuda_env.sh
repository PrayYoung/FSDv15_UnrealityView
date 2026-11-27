#!/usr/bin/env bash
# Use the mounted host driver instead of a container forward-compatibility shim.
# CUDA error 804 occurs when that shim is unsupported by the rented GeForce GPU.
_difix_library_paths=()
IFS=: read -ra _difix_existing_paths <<< "${LD_LIBRARY_PATH:-}"
for _difix_path in "${_difix_existing_paths[@]}"; do
  case "$_difix_path" in
    */cuda*/compat|*/cuda*/compat/) ;;
    "") ;;
    *) _difix_library_paths+=("$_difix_path") ;;
  esac
done
export LD_LIBRARY_PATH="$(IFS=:; echo "${_difix_library_paths[*]}")"
unset _difix_library_paths _difix_existing_paths _difix_path
