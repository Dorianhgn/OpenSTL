#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat <<'EOF'
Usage:
  flatten_checkpoints.sh /path/to/checkpoints

Description:
  Finds all .ckpt files inside subfolders of the given checkpoints directory,
  moves them to the root of that directory, and renames them to:
    epoch=<EPOCH>-val_loss=<LOSS>.ckpt

Notes:
  - The script ignores top-level .ckpt files (e.g., last.ckpt).
  - If a target filename already exists, a __dupN suffix is appended.
  - Empty directories left behind are removed.
EOF
}

if [[ ${1:-} == "-h" || ${1:-} == "--help" || ${1:-} == "" ]]; then
  usage
  [[ ${1:-} == "" ]] && exit 2 || exit 0
fi

checkpoints_dir="${1%/}"

if [[ ! -d "$checkpoints_dir" ]]; then
  echo "ERROR: Not a directory: $checkpoints_dir" >&2
  exit 2
fi

moved=0
skipped=0

# Only consider .ckpt inside subfolders (mindepth=2 avoids root-level last.ckpt etc.)
while IFS= read -r -d '' ckpt_path; do
  rel="${ckpt_path#"$checkpoints_dir"/}"
  base="$(basename -- "$ckpt_path")"

  epoch=""
  loss=""

  # Typical broken pattern in this repo:
  #   checkpoints/epoch=epoch=94-val_loss=val/loss=0.4798.ckpt
  if [[ "$rel" =~ epoch=epoch=([0-9]+) ]]; then
    epoch="${BASH_REMATCH[1]}"
  elif [[ "$rel" =~ epoch=([0-9]+) ]]; then
    epoch="${BASH_REMATCH[1]}"
  fi

  if [[ "$base" =~ loss=([-0-9.eE+]+)\.ckpt$ ]]; then
    loss="${BASH_REMATCH[1]}"
  elif [[ "$rel" =~ val_loss=([-0-9.eE+]+) ]]; then
    loss="${BASH_REMATCH[1]}"
  fi

  if [[ -z "$epoch" || -z "$loss" ]]; then
    echo "WARN: Could not parse epoch/loss for: $rel (skipping)" >&2
    skipped=$((skipped + 1))
    continue
  fi

  target="$checkpoints_dir/epoch=${epoch}-val_loss=${loss}.ckpt"

  if [[ -e "$target" ]]; then
    i=1
    while [[ -e "${target%.ckpt}__dup${i}.ckpt" ]]; do
      i=$((i + 1))
    done
    target="${target%.ckpt}__dup${i}.ckpt"
  fi

  mv -- "$ckpt_path" "$target"
  moved=$((moved + 1))
done < <(find "$checkpoints_dir" -mindepth 2 -type f -name "*.ckpt" -print0)

# Clean up empty directories left behind.
find "$checkpoints_dir" -mindepth 1 -type d -empty -print0 | while IFS= read -r -d '' empty_dir; do
  rmdir -- "$empty_dir" || true
done

echo "Done. moved=$moved skipped=$skipped"
