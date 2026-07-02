"""
오토레이블링 스크립트
입력: sensor_data_*.csv (accel_x/y/z, gyro_x/y/z)
출력: *_labeled.csv (pred_label, confidence 컬럼 추가)
"""
import os
import sys
import pickle
import argparse
import warnings
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F

sys.path.append('/workspace')
from model import MSTCNPlusPlus, BiLSTM

PREPROCESSED_DIR  = '/workspace/preprocessed'
MODELS_DIR        = '/workspace/models'
OUTPUT_DIR        = '/workspace/outputs'
CONFIDENCE_THRESH = 0.7


# 우선순위가 높은 표준 weight 파일명 (mtime보다 먼저 본다)
PREFERRED_MODELS = ['mstcnpp_best.pt', 'best.pt']


def find_model_path():
    """models/ 폴더에서 사용할 weight 파일 선택.

    파일 mtime은 `touch`나 복사만으로도 바뀌어 엉뚱한 모델이 선택될 수 있으므로,
    먼저 표준 파일명(mstcnpp_best.pt 등)을 찾고, 없을 때만 최신 .pt로 폴백한다.
    """
    import glob
    models_dir = MODELS_DIR

    # 1. 표준 파일명 우선
    for name in PREFERRED_MODELS:
        cand = os.path.join(models_dir, name)
        if os.path.exists(cand):
            print(f"[모델 선택] {name} (표준 파일명)")
            return cand

    # 2. 폴백: 가장 최신 .pt 파일
    pt_files = sorted(
        glob.glob(os.path.join(models_dir, '*.pt')),
        key=os.path.getmtime,
        reverse=True
    )
    if not pt_files:
        raise FileNotFoundError(f"{models_dir} 에 .pt 파일이 없습니다.")

    selected = pt_files[0]
    print(f"[모델 자동 감지] {os.path.basename(selected)}")
    if len(pt_files) > 1:
        print(f"  (표준 파일명 없음 → 후보 {len(pt_files)}개 중 최신 파일 선택)")
    return selected

COL_MAP = {
    'accel_x': 'acc_x', 'accel_y': 'acc_y', 'accel_z': 'acc_z',
    'gyro_x':  'gyro_x','gyro_y':  'gyro_y','gyro_z':  'gyro_z',
}
COL_MAP_INV = {v: k for k, v in COL_MAP.items()}
FEATURES = ['acc_x', 'acc_y', 'acc_z', 'gyro_x', 'gyro_y', 'gyro_z']


def detect_sampling_rate(df):
    """timestamp 간격으로 샘플링 레이트 자동 감지"""
    interval_ms = df['timestamp'].diff().dropna().median()
    hz = round(1000 / interval_ms)
    return hz, interval_ms


def load_model(device):
    with open(os.path.join(PREPROCESSED_DIR, 'scaler.pkl'), 'rb') as f:
        scaler = pickle.load(f)
    with open(os.path.join(PREPROCESSED_DIR, 'label_encoder.pkl'), 'rb') as f:
        le = pickle.load(f)

    model_path = find_model_path()
    # weights_only=True 로 안전 로드 (pickle 임의코드 실행 차단 + 향후 torch 기본값 변경 대비).
    # 구버전 torch나 비표준 체크포인트는 미지원일 수 있어 폴백.
    try:
        ckpt = torch.load(model_path, map_location=device, weights_only=True)
    except Exception as e:
        print(f"⚠ weights_only 로드 실패 → 일반 로드로 폴백 ({e})")
        ckpt = torch.load(model_path, map_location=device, weights_only=False)
    model_name = ckpt.get('model_name', 'mstcnpp')
    model_hz   = ckpt.get('sampling_rate', 100)  # 없으면 100Hz 가정

    print(f"모델: {model_name}")

    if model_name == 'bilstm':
        cfg    = ckpt['config']
        extra  = cfg.get('extra', {})
        seq_len = cfg.get('seq_len', 150)
        model  = BiLSTM(
            in_channels  = cfg['in_channels'],
            num_classes  = cfg['num_classes'],
            num_f_maps   = cfg.get('num_f_maps', 64),
            kernel_size  = cfg.get('kernel_size', 5),
            dropout      = cfg.get('dropout', 0.5),
            hidden_size  = extra.get('hidden_size', 128),
            lstm_layers  = extra.get('lstm_layers', 2),
            conv_layers  = extra.get('conv_layers', 2),
        ).to(device)
        # BiLSTM은 50Hz 기준으로 학습
        model_hz = ckpt.get('sampling_rate', 50)

    else:
        seq_len = ckpt.get('seq_len', 1000)
        model   = MSTCNPlusPlus(
            num_stages  = ckpt['num_stages'],
            num_layers  = ckpt['num_layers'],
            num_f_maps  = ckpt['num_f_maps'],
            in_channels = ckpt['in_channels'],
            num_classes = ckpt['num_classes'],
            kernel_size = ckpt['kernel_size'],
            dropout     = ckpt['dropout'],
        ).to(device)

    model.load_state_dict(ckpt['model_state_dict'])
    model.eval()

    print(f"로드 완료 (epoch {ckpt['epoch']}, val_acc: {ckpt['val_acc']:.4f})")
    print(f"모델 학습 Hz: {model_hz}Hz, SEQ_LEN: {seq_len}")
    print(f"클래스: {list(le.classes_)}")
    return model, scaler, le, model_hz, seq_len, model_name


def infer_full(X, model, device):
    """전체 시퀀스를 한 번에 추론 (fully-convolutional MS-TCN++ 전용).
    윈도우 경계 아티팩트가 없다. 메모리 부족 시 호출부에서 윈도우 추론으로 폴백."""
    with torch.no_grad():
        x      = torch.tensor(X.T).unsqueeze(0).to(device)   # (1, C, T)
        output = model(x)[-1]
        probs  = F.softmax(output, dim=1).squeeze(0).T.cpu().numpy()
    return probs


def infer(X, model, device, seq_len, num_classes):
    total     = len(X)
    all_probs = np.zeros((total, num_classes), dtype=np.float32)
    count     = np.zeros(total, dtype=np.float32)
    with torch.no_grad():
        for start in range(0, total, seq_len // 2):
            end   = min(start + seq_len, total)
            chunk = X[start:end]
            if len(chunk) < seq_len:
                pad   = np.zeros((seq_len - len(chunk), chunk.shape[1]), dtype=np.float32)
                chunk = np.vstack([chunk, pad])
            x      = torch.tensor(chunk.T).unsqueeze(0).to(device)
            output = model(x)[-1]
            probs  = F.softmax(output, dim=1).squeeze(0).T.cpu().numpy()
            actual = min(end - start, total - start)
            all_probs[start:start+actual] += probs[:actual]
            count[start:start+actual]     += 1
    all_probs /= np.maximum(count[:, None], 1)
    return all_probs


def auto_label(input_path, threshold):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")

    model, scaler, le, model_hz, seq_len, model_name = load_model(device)
    num_classes = len(le.classes_)

    df = pd.read_csv(input_path)
    print(f"\n입력 파일: {input_path}")
    print(f"컬럼: {list(df.columns)}")
    print(f"프레임 수: {len(df):,}")

    # 샘플링 레이트 자동 감지
    csv_hz, interval_ms = detect_sampling_rate(df)
    print(f"CSV 샘플링 레이트: {csv_hz}Hz (간격 {interval_ms:.1f}ms)")

    # Hz 불일치 체크
    if csv_hz != model_hz:
        print(f"\n⚠ Hz 불일치 감지: CSV={csv_hz}Hz, 모델={model_hz}Hz")
        if csv_hz == 100 and model_hz == 50:
            print("  → 추론 후 100Hz CSV에 2행씩 같은 레이블 적용")
        elif csv_hz == 50 and model_hz == 100:
            print("  → 100Hz 모델에 50Hz 데이터 입력 (성능 저하 가능)")
    else:
        print(f"  → Hz 일치 ({csv_hz}Hz)")

    df = df.rename(columns=COL_MAP)
    missing = [c for c in FEATURES if c not in df.columns]
    if missing:
        raise ValueError(f"필요한 컬럼이 없습니다: {missing}\n현재 컬럼: {list(df.columns)}")

    # 모델이 50Hz인데 CSV가 100Hz면 다운샘플링해서 추론
    if csv_hz == 100 and model_hz == 50:
        df_infer = df.iloc[::2].reset_index(drop=True)
        print(f"  → 추론용 다운샘플링: {len(df)}행 → {len(df_infer)}행")
    else:
        df_infer = df

    X_df = pd.DataFrame(df_infer[FEATURES].values, columns=FEATURES).astype(np.float32)
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        X = scaler.transform(X_df)

    print("\n추론 중...")
    # MS-TCN++은 fully-convolutional이라 전체 시퀀스 1패스가 가능(윈도우 경계 아티팩트 없음).
    # 메모리 부족 시 윈도우 추론으로 폴백. BiLSTM은 학습 seq_len 분포를 따르도록 윈도우 유지.
    all_probs = None
    if model_name != 'bilstm':
        try:
            all_probs = infer_full(X, model, device)
            print(f"  전체 시퀀스 1패스 추론 ({len(X):,} 프레임)")
        except RuntimeError as e:
            if device.type == 'cuda':
                torch.cuda.empty_cache()
            print(f"  ⚠ 1패스 추론 실패({type(e).__name__}) → 윈도우 추론으로 폴백")
            all_probs = None
    if all_probs is None:
        all_probs = infer(X, model, device, seq_len, num_classes)
    confidence  = all_probs.max(axis=1)
    pred_idx    = all_probs.argmax(axis=1)
    pred_labels = le.inverse_transform(pred_idx).astype(object)

    # ── 신뢰도 임계값 적용: threshold 미만 예측은 Unlabeled 처리 ──
    low_conf_mask = confidence < threshold
    pred_labels[low_conf_mask] = 'Unlabeled'
    print(f"\n임계값 {threshold} 적용: {int(low_conf_mask.sum()):,}개 → Unlabeled")

    # ── 100Hz CSV + 50Hz 모델: 2행씩 같은 레이블 적용 (정보 손실 없음) ──
    if csv_hz == 100 and model_hz == 50:
        print(f"\n[Hz Adaptive 레이블링]")
        print(f"  추론 결과: {len(pred_labels):,}개 (50Hz 기준)")
        print(f"  원본 CSV:  {len(df):,}행 (100Hz)")
        print(f"  방식: 50Hz 레이블 1개 → 100Hz 행 2개에 동일 적용")

        labels_100hz = np.repeat(pred_labels, 2)[:len(df)]
        conf_100hz   = np.repeat(confidence,  2)[:len(df)]

        # 검증: 2행씩 동일한지 확인
        mismatch = 0
        for i in range(0, len(labels_100hz) - 1, 2):
            if labels_100hz[i] != labels_100hz[i+1]:
                mismatch += 1
        print(f"  검증: 2행씩 동일 여부 → 불일치 {mismatch}쌍 (0이어야 정상)")
        print(f"  원본 행수 유지: {len(df):,}행 ✓")

        df['pred_label'] = labels_100hz
        df['confidence'] = conf_100hz.round(4)

    elif csv_hz == 50 and model_hz == 50:
        print(f"\n[Hz 일치: 50Hz → 50Hz 직접 레이블링]")
        df['pred_label'] = pred_labels
        df['confidence'] = confidence.round(4)

    elif csv_hz == 100 and model_hz == 100:
        print(f"\n[Hz 일치: 100Hz → 100Hz 직접 레이블링]")
        df['pred_label'] = pred_labels
        df['confidence'] = confidence.round(4)

    else:
        print(f"\n⚠ 비표준 Hz 조합: CSV={csv_hz}Hz, 모델={model_hz}Hz → 그대로 적용")
        df['pred_label'] = pred_labels
        df['confidence'] = confidence.round(4)

    df = df.rename(columns=COL_MAP_INV)

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    out_name = os.path.splitext(os.path.basename(input_path))[0] + '_labeled.csv'
    out_path = os.path.join(OUTPUT_DIR, out_name)
    df.to_csv(out_path, index=False)

    unlabeled = (df['pred_label'] == 'Unlabeled').sum()
    print(f"\n결과 요약:")
    print(f"  CSV Hz        : {csv_hz}Hz")
    print(f"  모델 Hz       : {model_hz}Hz")
    print(f"  전체 프레임   : {len(df):,}")
    print(f"  자동 레이블   : {len(df)-unlabeled:,} ({(len(df)-unlabeled)/len(df)*100:.1f}%)")
    print(f"  Unlabeled        : {unlabeled:,} ({unlabeled/len(df)*100:.1f}%)")
    print(f"\n클래스별 분포:")
    for cls, cnt in df['pred_label'].value_counts().items():
        print(f"  {cls:12s}: {cnt:>6,}개 ({cnt/len(df)*100:.1f}%)")
    print(f"\n저장 완료: {out_path}")
    return out_path


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--input',     type=str,   required=True)
    parser.add_argument('--threshold', type=float, default=CONFIDENCE_THRESH)
    args = parser.parse_args()
    auto_label(args.input, args.threshold)
