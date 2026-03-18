#!/usr/bin/env bash

set -euo pipefail

# Public SEVIR source of truth:
# - AWS Open Data Registry: https://registry.opendata.aws/sevir/
# - Bucket: https://sevir.s3.amazonaws.com

AVAILABLE_MODALITIES=(vis ir069 ir107 vil lght)
DEFAULT_MODALITIES=(vil)

MODALITIES=("${DEFAULT_MODALITIES[@]}")
DO_PROCESS=1

show_help() {
	cat <<'EOF'
Usage: bash tools/prepare_data/download_sevir.sh [options]

Options:
  --modalities MOD1 MOD2 ...   Modalities to download (vis, ir069, ir107, vil, lght)
  --modalities=MOD1,MOD2,...   Comma-separated modality list
  --no-process                 Only download raw files, do not generate processed HDF5 files
  --help                       Show this help message

Default:
  --modalities vil
EOF
}

contains_item() {
	local needle="$1"
	shift
	for item in "$@"; do
		if [[ "$item" == "$needle" ]]; then
			return 0
		fi
	done
	return 1
}

parse_modalities_csv() {
	local csv="$1"
	IFS=',' read -r -a MODALITIES <<< "$csv"
}

while [[ $# -gt 0 ]]; do
	case "$1" in
		--help|-h)
			show_help
			exit 0
			;;
		--no-process)
			DO_PROCESS=0
			shift
			;;
		--modalities=*)
			parse_modalities_csv "${1#*=}"
			shift
			;;
		--modalities)
			shift
			MODALITIES=()
			while [[ $# -gt 0 && "$1" != --* ]]; do
				MODALITIES+=("$1")
				shift
			done
			;;
		*)
			echo "Unknown argument: $1" >&2
			show_help
			exit 1
			;;
	esac
done

if [[ ${#MODALITIES[@]} -eq 0 ]]; then
	echo "No modalities selected." >&2
	exit 1
fi

declare -A SEEN_MODALITY=()
FILTERED_MODALITIES=()
for modality in "${MODALITIES[@]}"; do
	if ! contains_item "$modality" "${AVAILABLE_MODALITIES[@]}"; then
		echo "Unsupported modality: $modality" >&2
		echo "Allowed: ${AVAILABLE_MODALITIES[*]}" >&2
		exit 1
	fi
	if [[ -z "${SEEN_MODALITY[$modality]+x}" ]]; then
		SEEN_MODALITY[$modality]=1
		FILTERED_MODALITIES+=("$modality")
	fi
done

MODALITIES=("${FILTERED_MODALITIES[@]}")

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "${SCRIPT_DIR}/../.." && pwd)"
DATA_DIR="${ROOT_DIR}/data/sevir"
BUCKET_URL="https://sevir.s3.amazonaws.com"

mkdir -p "${DATA_DIR}"

download_with_wget() {
	local url="$1"
	local dest="$2"

	mkdir -p "$(dirname "${dest}")"
	if [[ -f "${dest}" ]]; then
		echo "Skipping existing ${dest}"
		return 0
	fi

	if command -v wget >/dev/null 2>&1; then
		wget -c "${url}" -O "${dest}"
	elif command -v curl >/dev/null 2>&1; then
		curl -L --fail --retry 3 --retry-delay 5 -C - "${url}" -o "${dest}"
	else
		echo "Neither wget nor curl is available." >&2
		exit 1
	fi
}

echo "Downloading SEVIR catalog"
download_with_wget "${BUCKET_URL}/CATALOG.csv" "${DATA_DIR}/CATALOG.csv"

echo "Selected modalities: ${MODALITIES[*]}"
echo "Downloading SEVIR raw HDF5 files"
export SEVIR_BUCKET_URL="${BUCKET_URL}"
export SEVIR_DATA_DIR="${DATA_DIR}"
for modality in "${MODALITIES[@]}"; do
	while IFS= read -r key; do
		[[ -z "$key" ]] && continue
		dest="${DATA_DIR}/${key}"
		if [[ -f "$dest" ]]; then
			echo "Skipping existing ${key}"
			continue
		fi
		mkdir -p "$(dirname "$dest")"
		echo "Downloading ${key}"
		if command -v wget >/dev/null 2>&1; then
			wget -c "${BUCKET_URL}/${key}" -O "$dest"
		elif command -v curl >/dev/null 2>&1; then
			curl -L --fail --retry 3 --retry-delay 5 -C - "${BUCKET_URL}/${key}" -o "$dest"
		else
			echo "Neither wget nor curl is available." >&2
			exit 1
		fi
	done < <(python - "$BUCKET_URL" "$modality" <<'PY'
import sys
import urllib.parse
import urllib.request
import xml.etree.ElementTree as ET

bucket_url = sys.argv[1].rstrip("/")
modality = sys.argv[2]
prefix = f"data/{modality}/"

def list_keys(prefix: str):
	token = None
	while True:
		params = {"list-type": "2", "prefix": prefix, "max-keys": "1000"}
		if token:
			params["continuation-token"] = token
		url = f"{bucket_url}/?{urllib.parse.urlencode(params)}"
		with urllib.request.urlopen(url, timeout=120) as response:
			root = ET.fromstring(response.read())
		ns = {"s3": "http://s3.amazonaws.com/doc/2006-03-01/"}
		for content in root.findall("s3:Contents", ns):
			key = content.findtext("s3:Key", default="", namespaces=ns)
			if key and not key.endswith("/"):
				print(key)
		if root.findtext("s3:IsTruncated", default="false", namespaces=ns) != "true":
			break
		token = root.findtext("s3:NextContinuationToken", default="", namespaces=ns)
		if not token:
			break

list_keys(prefix)
PY
)
done

echo "Finished downloading raw SEVIR data"

# Convert raw SEVIR data into the compact training/testing H5 files consumed by OpenSTL.
mkdir -p "${DATA_DIR}/processed"

if [[ "$DO_PROCESS" -eq 1 ]]; then
	for modality in "${MODALITIES[@]}"; do
		case "$modality" in
			vis|ir069|ir107|vil)
				train_h5="${DATA_DIR}/processed/${modality}_training.h5"
				test_h5="${DATA_DIR}/processed/${modality}_testing.h5"
				if [[ -f "$train_h5" && -f "$test_h5" ]]; then
					echo "Skipping processing for ${modality} (already exists)"
					continue
				fi
				python "${ROOT_DIR}/tools/prepare_data/generate_sevir.py" \
					--sevir_data "${DATA_DIR}" \
					--data_name "$modality" \
					--output_dir "${DATA_DIR}/processed"
				;;
			lght)
				echo "Skipping processing for lght: no processed HDF5 pipeline is defined in generate_sevir.py"
				;;
		esac
	done
fi

echo "finished"

# Download and arrange them in the following structure:
# OpenSTL
# ©¸©¤©¤data
#    ©À©¤©¤ sevir
#    ©¦   ©À©¤©¤ ir069
#    ©¦   ©À©¤©¤ ir107
#    ©¦   ©À©¤©¤ vis
#    ©¦   ©À©¤©¤ vil
#    ©¦   ©À©¤©¤ lght
#    ©¦   ©À©¤©¤ processed
#    ©¦   ©¦   ©À©¤©¤ ir069_training.h5
#    ©¦   ©¦   ©À©¤©¤ ir069_testing.h5
#    ©¦   ©¦   ©À©¤©¤ ir107_training.h5
#    ©¦   ©¦   ©À©¤©¤ ir107_testing.h5
#    ©¦   ©¦   ©À©¤©¤ vis_training.h5
#    ©¦   ©¦   ©À©¤©¤ vis_testing.h5
#    ©¦   ©¦   ©À©¤©¤ vil_training.h5
#    ©¦   ©¦   ©À©¤©¤ vil_testing.h5
