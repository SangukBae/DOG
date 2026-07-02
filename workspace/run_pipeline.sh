#!/bin/bash
# 오토레이블링 + 뷰어 생성 한 번에 실행
# 사용법: bash run_pipeline.sh <sensor_csv> <video_mp4> [threshold]

set -e

SENSOR=$1
VIDEO=$2
THRESHOLD=${3:-0.7}

if [ -z "$SENSOR" ] || [ -z "$VIDEO" ]; then
  echo "사용법: bash run_pipeline.sh <sensor_csv> <video_mp4> [threshold]"
  echo "예시:   bash run_pipeline.sh /workspace/test_data/sensor_data_20260421_063519.csv /workspace/test_data/test1_H264.mp4 0.7"
  exit 1
fi

BASE=$(basename "$SENSOR" .csv)

# 입력이 test_data 하위폴더에 있으면 그 구조를 outputs 아래에 그대로 유지
# 예: /workspace/test_data/260521/B/xxx.csv → /workspace/outputs/260521/B/
TEST_ROOT=/workspace/test_data
SENSOR_DIR=$(cd "$(dirname "$SENSOR")" && pwd)
if [[ "$SENSOR_DIR" == "$TEST_ROOT" ]]; then
  REL=""
elif [[ "$SENSOR_DIR" == "$TEST_ROOT"/* ]]; then
  REL="${SENSOR_DIR#"$TEST_ROOT"/}"
else
  REL=""   # test_data 밖의 입력은 outputs 최상위에 저장
fi
OUTSUB="/workspace/outputs${REL:+/$REL}"
mkdir -p "$OUTSUB"

LABEL_OUT="$OUTSUB/${BASE}_labeled.csv"
SMOOTH_OUT="$OUTSUB/${BASE}_labeled_smoothed.csv"
VIEWER_OUT="$OUTSUB/${BASE}_viewer.html"

echo "=============================="
echo " 오토 레이블링 파이프라인"
echo "=============================="
echo " 센서 파일: $SENSOR"
echo " 영상 파일: $VIDEO"
echo " 신뢰도 임계치: $THRESHOLD"
echo "------------------------------"

echo "[1/3] 오토레이블링 실행 중..."
python /workspace/auto_label.py --input "$SENSOR" --threshold "$THRESHOLD" --output-dir "$OUTSUB"

echo "[2/3] 필터링 + 소리 처리 중..."
python /workspace/postprocess.py \
  --input         "$LABEL_OUT" \
  --min-duration  1.0 \
  --smooth-window 0.5 \
  --protect-conf  0.85 \
  --video         "$VIDEO"

# 후처리 성공 시 그 결과를, 실패 시 원본 라벨을 뷰어 입력으로 사용
if [ -f "$SMOOTH_OUT" ]; then VIEWER_LABEL="$SMOOTH_OUT"; else VIEWER_LABEL="$LABEL_OUT"; fi

echo "[3/3] 검수 뷰어 생성 중..."
python /workspace/make_viewer.py \
  --sensor "$SENSOR" \
  --label  "$VIEWER_LABEL" \
  --video  "$VIDEO" \
  --output "$VIEWER_OUT" \
  --threshold "$THRESHOLD"

echo "=============================="
echo " 완료!"
echo " 뷰어: $VIEWER_OUT"
echo " 로컬에서 HTML 파일을 브라우저로 열어 검수하세요"
echo "=============================="
