#!/usr/bin/env bash
set -euo pipefail

work_dir="${1:?usage: build_yolo11_rknn_ci.sh WORK_DIR OUTPUT_DIR}"
output_dir="${2:?usage: build_yolo11_rknn_ci.sh WORK_DIR OUTPUT_DIR}"

model_zoo_ref="v2.3.0"
model_url="https://ftrg.zbox.filez.com/v2/delivery/data/95f00b0fc900458ba134f8b180b3f7a1/examples/yolo11/yolo11n.onnx"
model_sha256="62a3751662e06c678debb54b113c211d918f245ea6d3aea0b09fc418b8fc7705"
model_zoo_dir="${work_dir}/rknn_model_zoo"
onnx_path="${work_dir}/yolo11n.onnx"
rknn_path="${output_dir}/yolo11n-rk3588-int8.rknn"

mkdir -p "${work_dir}" "${output_dir}"

if [[ ! -d "${model_zoo_dir}/.git" ]]; then
  git clone --depth 1 --branch "${model_zoo_ref}" \
    https://github.com/airockchip/rknn_model_zoo.git "${model_zoo_dir}"
fi

curl --fail --location --retry 5 --retry-all-errors \
  --output "${onnx_path}" "${model_url}"
printf '%s  %s\n' "${model_sha256}" "${onnx_path}" | sha256sum --check --strict

(
  cd "${model_zoo_dir}/examples/yolo11/python"
  python convert.py "${onnx_path}" rk3588 i8 "${rknn_path}"
)

test -s "${rknn_path}"
rknn_sha256="$(sha256sum "${rknn_path}" | awk '{print $1}')"
rknn_size="$(stat --format='%s' "${rknn_path}")"

cat > "${output_dir}/yolo11n-rk3588-int8.manifest.json" <<EOF
{
  "schema_version": 1,
  "model": "Rockchip-optimized YOLO11n COCO",
  "model_family": "yolo11",
  "class_count": 80,
  "phone_class_id": 67,
  "input_shape": [1, 3, 640, 640],
  "target_platform": "rk3588",
  "quantization": "int8",
  "rknn_toolkit_version": "2.3.0",
  "rknn_model_zoo_ref": "${model_zoo_ref}",
  "onnx_url": "${model_url}",
  "onnx_sha256": "${model_sha256}",
  "rknn_filename": "$(basename "${rknn_path}")",
  "rknn_size_bytes": ${rknn_size},
  "rknn_sha256": "${rknn_sha256}"
}
EOF

sha256sum "${rknn_path}" "${output_dir}/yolo11n-rk3588-int8.manifest.json"
