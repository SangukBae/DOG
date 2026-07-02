#!/bin/bash
# workspace 날짜폴더(기본) 또는 임의 root 하위의 모든 CSV+MP4 쌍을 재귀 탐색해 오토라벨링 파이프라인 일괄 실행
# 사용법: bash run_batch.sh [root_dir=/workspace] [threshold=0.7]
#
# 출력은 입력의 하위폴더 구조를 그대로 유지한다.
#   /workspace/260521/260521/B/20260521_124753.csv → outputs/260521/260521/B/20260521_124753_*.{csv,html}
#   /workspace/test_data/260521/B/20260521_124753.csv → outputs/260521/B/20260521_124753_*.{csv,html}

set -u

ROOT=${1:-/workspace}
THRESHOLD=${2:-0.7}
OUT=/workspace/outputs
WORK_ROOT=/workspace
TEST_ROOT=/workspace/test_data
DATE_RE='^[0-9]{6}$'
mkdir -p "$OUT"

collect_csvs() {
  if [ "$ROOT" = "$WORK_ROOT" ]; then
    find "$WORK_ROOT" -mindepth 2 -type f -name '*.csv' \
      ! -path "$OUT/*" \
      ! -path "$TEST_ROOT/*" \
      ! -name '*_labeled.csv' \
      ! -name '*_labeled_smoothed.csv' \
      ! -name '*_labeled_reviewed.csv' \
      | awk -F/ '$3 ~ /^[0-9]{6}$/ {print}'
  else
    find "$ROOT" -type f -name '*.csv' \
      ! -name '*_labeled.csv' \
      ! -name '*_labeled_smoothed.csv' \
      ! -name '*_labeled_reviewed.csv'
  fi
}

mapfile -t CSVS < <(collect_csvs | sort)
total=${#CSVS[@]}
echo "대상 CSV: $total 개 (root=$ROOT, threshold=$THRESHOLD)"

ok=0; skip=0; fail=0; i=0
for csv in "${CSVS[@]}"; do
  i=$((i+1))
  dir=$(dirname "$csv")
  stem=$(basename "$csv" .csv)
  video="$dir/$stem.mp4"

  # 입력 하위경로 → 동일 구조를 outputs 아래에 생성
  if [[ "$dir" == "$TEST_ROOT" ]]; then
    rel=""
  elif [[ "$dir" == "$TEST_ROOT"/* ]]; then
    rel=${dir#"$TEST_ROOT"/}
  elif [[ "$dir" == "$WORK_ROOT"/* ]]; then
    rel_cand=${dir#"$WORK_ROOT"/}
    top=${rel_cand%%/*}
    if [[ "$top" =~ $DATE_RE ]]; then
      rel="$rel_cand"
    elif [ "$dir" = "$ROOT" ]; then
      rel=""
    else
      rel=${dir#"$ROOT"/}
    fi
  elif [ "$dir" = "$ROOT" ]; then
    rel=""
  else
    rel=${dir#"$ROOT"/}
  fi
  outsub="$OUT${rel:+/$rel}"
  mkdir -p "$outsub"

  label_out="$outsub/${stem}_labeled.csv"
  smooth_out="$outsub/${stem}_labeled_smoothed.csv"
  viewer_out="$outsub/${stem}_viewer.html"

  echo "------------------------------------------------------------"
  echo "[$i/$total] ${rel:+$rel/}$stem"

  if [ ! -f "$video" ]; then
    echo "  ⚠ 짝 영상 없음: $video → 건너뜀"
    skip=$((skip+1)); continue
  fi
  if [ -f "$viewer_out" ]; then
    echo "  ✓ 이미 처리됨 → 건너뜀"
    skip=$((skip+1)); continue
  fi

  # 후처리 성공 시 smoothed 를, 실패 시 labeled 원본을 뷰어 입력으로 사용
  if python /workspace/auto_label.py --input "$csv" --threshold "$THRESHOLD" --output-dir "$outsub"; then
    python /workspace/postprocess.py --input "$label_out" \
      --min-duration 1.0 --smooth-window 0.5 --protect-conf 0.85 --video "$video" || true
    if [ -f "$smooth_out" ]; then viewer_label="$smooth_out"; else viewer_label="$label_out"; fi
    if python /workspace/make_viewer.py \
         --sensor "$csv" --label "$viewer_label" --video "$video" \
         --output "$viewer_out" --threshold "$THRESHOLD"; then
      echo "  완료 → $viewer_out"
      ok=$((ok+1))
    else
      echo "  ✗ 뷰어 생성 실패: ${rel:+$rel/}$stem"
      fail=$((fail+1))
    fi
  else
    echo "  ✗ 오토레이블링 실패: ${rel:+$rel/}$stem"
    fail=$((fail+1))
  fi
done

echo "============================================================"
echo "완료: 성공 $ok / 건너뜀 $skip / 실패 $fail (전체 $total)"
