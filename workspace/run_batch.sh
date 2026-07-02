#!/bin/bash
# test_data/ 하위의 모든 CSV+MP4 쌍을 재귀 탐색해 오토라벨링 파이프라인 일괄 실행
# 사용법: bash run_batch.sh [root_dir=/workspace/test_data] [threshold=0.7]
#
# 출력은 outputs/ 가 평탄하므로 파일명 충돌을 막기 위해
# <날짜>_<피험자>_<원본이름> 형태(BASE)로 라벨/뷰어를 생성한다.

set -u

ROOT=${1:-/workspace/test_data}
THRESHOLD=${2:-0.7}
OUT=/workspace/outputs
mkdir -p "$OUT"

mapfile -t CSVS < <(find "$ROOT" -type f -name '*.csv' ! -name '*_labeled.csv' | sort)
total=${#CSVS[@]}
echo "대상 CSV: $total 개 (root=$ROOT, threshold=$THRESHOLD)"

ok=0; skip=0; fail=0; i=0
for csv in "${CSVS[@]}"; do
  i=$((i+1))
  dir=$(dirname "$csv")
  stem=$(basename "$csv" .csv)
  video="$dir/$stem.mp4"

  # test_data/<날짜>/<피험자>/<stem> → <날짜>_<피험자>_<stem> 로 고유화
  rel=${dir#$ROOT/}
  prefix=$(echo "$rel" | tr '/' '_')
  base="${prefix}_${stem}"

  label_out="$OUT/${base}_labeled.csv"
  viewer_out="$OUT/${base}_viewer.html"

  echo "------------------------------------------------------------"
  echo "[$i/$total] $rel/$stem"

  if [ ! -f "$video" ]; then
    echo "  ⚠ 짝 영상 없음: $video → 건너뜀"
    skip=$((skip+1)); continue
  fi
  if [ -f "$viewer_out" ]; then
    echo "  ✓ 이미 처리됨 → 건너뜀"
    skip=$((skip+1)); continue
  fi

  if python /workspace/auto_label.py --input "$csv" --threshold "$THRESHOLD" \
     && python /workspace/postprocess.py --input "$OUT/${stem}_labeled.csv" \
          --min-duration 1.0 --smooth-window 0.5 --protect-conf 0.85 --video "$video" \
     && cp -f "$OUT/${stem}_labeled_smoothed.csv" "$label_out" \
     && python /workspace/make_viewer.py \
          --sensor "$csv" --label "$label_out" --video "$video" \
          --output "$viewer_out" --threshold "$THRESHOLD"; then
    rm -f "$OUT/${stem}_labeled.csv" "$OUT/${stem}_labeled_smoothed.csv"   # 고유화 이름만 남김
    echo "  완료 → $viewer_out"
    ok=$((ok+1))
  else
    echo "  ✗ 실패: $rel/$stem"
    fail=$((fail+1))
  fi
done

echo "============================================================"
echo "완료: 성공 $ok / 건너뜀 $skip / 실패 $fail (전체 $total)"
