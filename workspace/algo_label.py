"""
알고리즘 기반 보정 레이어 (하이브리드 Phase 1, 보수적 보정)

목걸이 IMU 신호처리로 '쉬운' 행동을 직접 판정해서, 알고리즘이 확신할 때만
딥러닝 라벨을 덮어쓴다. 애매하면 DL 라벨을 그대로 둔다(→ Eating/Drinking/
Sniffing 같은 미세 행동은 자연히 DL이 담당).

담당 영역(알고리즘):
  - Lying / Standing : 중력 방향(저역통과 accel)으로 목걸이 기울기
  - Walking / Running: |accel| 동적 강도 + 걸음 주파수(FFT)
  - Shaking          : 고주파 강한 주기 신호 + 자이로 에너지

자가 보정(self-calibration): DL이 높은 신뢰도로 단 라벨들의 신호 분포에서
임계값/중력 중심(centroid)을 자동 학습 → 목걸이 장착 방식이 달라도 적응.
보정 데이터가 부족한 클래스는 '기권(abstain)'하여 DL을 유지(보수적).
"""
import numpy as np

# 알고리즘이 담당하는 클래스 (이 5개만 덮어쓰기 후보)
DOMAIN = ['Lying', 'Standing', 'Walking', 'Running', 'Shaking']
# 자세(Lying/Standing) 교정을 허용할 DL 원본 라벨 (확신한 미세행동은 보존)
POSTURE_OVERRIDABLE = {'Lying', 'Standing', '미분류'}
GRAVITY = 9.81

# 자가 보정에 사용할 DL 신뢰도 하한
CALIB_CONF = 0.80
# 알고리즘이 덮어쓰기 위한 최소 확신도 / 윈도우 동의 비율
OVERRIDE_CONF = 0.60
OVERRIDE_AGREE = 0.50


def _valley_2means(x, iters=30):
    """1D 2-means로 분포를 두 모드(rest/active)로 나눈 경계값.
    움직임 분포에서 '정지 vs 활동' 골짜기를 라벨 없이 찾는다.
    활동 꼬리가 무거우므로 로그 공간에서 분할(rest 클러스터를 잘 분리)."""
    x = np.asarray(x, dtype=float)
    x = x[np.isfinite(x) & (x >= 0)]
    if x.size < 4:
        return 1.0
    lx = np.log(x + 1e-3)
    c0, c1 = np.percentile(lx, 15), np.percentile(lx, 85)
    if c1 - c0 < 1e-6:
        return float(np.exp((c0 + c1) / 2) - 1e-3)
    for _ in range(iters):
        mid = (c0 + c1) / 2
        a, b = lx[lx <= mid], lx[lx > mid]
        if a.size:
            c0 = a.mean()
        if b.size:
            c1 = b.mean()
    return float(np.exp((c0 + c1) / 2) - 1e-3)


def _unit(v):
    return v / (np.linalg.norm(v) + 1e-9)


def _kmeans2_unit(O, iters=30):
    """단위 중력 벡터 집합 O(N,3)를 코사인 기준 2개 군집으로. (c0, c1, assign)
    assign: True=군집1. 잘 분리된 두 자세를 라벨 없이 가른다."""
    seed0 = O[0]
    seed1 = O[int((O @ seed0).argmin())]   # seed0에서 가장 먼(코사인 최소) 점
    c0, c1 = _unit(seed0), _unit(seed1)
    assign = (O @ c1) > (O @ c0)
    for _ in range(iters):
        assign = (O @ c1) > (O @ c0)
        if (~assign).any():
            c0 = _unit(O[~assign].mean(axis=0))
        if assign.any():
            c1 = _unit(O[assign].mean(axis=0))
    return c0, c1, assign


def _longest_run(arr, val):
    """arr(시간순)에서 값 val이 연속으로 나오는 최장 길이."""
    best = cur = 0
    for x in arr:
        if x == val:
            cur += 1
            best = max(best, cur)
        else:
            cur = 0
    return best


def _windows(n, win, stride):
    starts = list(range(0, max(1, n - win + 1), stride))
    if starts and starts[-1] != n - win and n >= win:
        starts.append(n - win)
    if not starts:
        starts = [0]
    return [(s, min(s + win, n)) for s in starts]


def _win_features(accel, gyro, amag, hz, s, e):
    """한 윈도우의 신호 피처."""
    grav = accel[s:e].mean(axis=0)              # 저역통과 ≈ 중력 방향
    gmag = np.linalg.norm(grav) + 1e-9
    orient = grav / gmag                        # 단위 중력 벡터
    dyn = amag[s:e] - amag[s:e].mean()          # 동적 가속도(중력 제거)
    motion = float(dyn.std())                   # 움직임 강도
    gyro_e = float(np.linalg.norm(gyro[s:e], axis=1).mean())  # 회전 에너지

    # 주파수: 동적 가속도의 FFT 피크 (0.5~12Hz 대역)
    L = e - s
    if L >= 8:
        P = np.abs(np.fft.rfft(dyn * np.hanning(L)))
        freqs = np.fft.rfftfreq(L, d=1.0 / hz)
        band = (freqs >= 0.5) & (freqs <= 12.0)
        if band.any():
            Pb = P[band]; fb = freqs[band]
            k = int(Pb.argmax())
            peak_freq = float(fb[k])
            peak_ratio = float(Pb[k] / (Pb.sum() + 1e-9))  # 주기성 강도
        else:
            peak_freq, peak_ratio = 0.0, 0.0
    else:
        peak_freq, peak_ratio = 0.0, 0.0

    return dict(orient=orient, motion=motion, gyro_e=gyro_e,
                peak_freq=peak_freq, peak_ratio=peak_ratio)


def _calibrate(feats, dl_label, dl_conf, swap_posture=False, verbose=True):
    """라벨 비의존 자가보정. 자세 centroid는 DL 라벨이 아니라 정지 프레임
    중력방향의 무감독 군집화로 얻는다(=DL이 confidently 틀려도 오염 안 됨).
    임계값은 신호 분포에서 직접 추정. 근거 부족하면 None(→기권)."""
    feats = np.array(feats, dtype=object)
    conf = np.array(dl_conf)
    lab = np.array(dl_label)
    hi = conf >= CALIB_CONF

    cal = {}

    # 정적 vs 이동 경계: 움직임 분포 자체를 로그 2-means로 갈라 '골짜기'를 임계값으로
    motion = np.array([f['motion'] for f in feats])
    cal['static_th'] = float(np.clip(_valley_2means(motion), 0.4, 2.0))
    cal['walk_run_fb'] = 3.0  # 개 보행~2Hz/구보 3~4Hz: 물리 기본값 고정

    # ── 자세 centroid (라벨 비의존) ─────────────────────────────────────
    # 정지 프레임의 중력방향을 k=2 무감독 군집화 → 깨끗한 두 자세 중심.
    # 이름(Lying/Standing)은 DL이 아니라 물리 단서로: 정지 연속구간(bout)이
    # 더 긴 군집 = Lying (개는 누우면 오래 가만히 있고, Standing은 더 짧음).
    cal['c_lying'] = None
    cal['c_standing'] = None
    orient = np.array([f['orient'] for f in feats])
    smask = motion < cal['static_th']
    if smask.sum() >= 20:
        O = orient[smask]
        c0, c1, assign = _kmeans2_unit(O)
        sep = 1.0 - float(np.dot(c0, c1))           # 두 자세 분리도
        if sep >= 0.05:
            full = np.full(len(feats), -1)          # 시간순 군집 라벨
            full[np.where(smask)[0]] = assign.astype(int)
            run0, run1 = _longest_run(full, 0), _longest_run(full, 1)

            # 군집별 DL 자세 다수표 (고신뢰 우선)
            def posture_majority(cid):
                for sel in ((full == cid) & hi, full == cid):
                    sub = lab[sel]
                    nL = int((sub == 'Lying').sum()); nS = int((sub == 'Standing').sum())
                    if nL + nS >= 10:
                        return 'Lying' if nL > nS else 'Standing'
                return None
            m0, m1 = posture_majority(0), posture_majority(1)

            # 두 군집의 DL 이름이 서로 다르면(자기일관적) DL naming 신뢰,
            # 같으면(DL이 두 자세를 혼동) bout 길이로 폴백.
            if m0 and m1 and m0 != m1:
                lying_is_1 = (m1 == 'Lying'); method = 'DL다수표'
            else:
                lying_is_1 = run1 >= run0; method = '정지구간길이'
            lying_is_1 ^= bool(swap_posture)
            cal['c_lying'], cal['c_standing'] = (c1, c0) if lying_is_1 else (c0, c1)
            if verbose:
                print(f"  [algo] 자세 군집(무감독): sep={sep:.2f}, 이름결정={method}"
                      f"{' (swap)' if swap_posture else ''} "
                      f"(c0:{m0}/bout{run0}, c1:{m1}/bout{run1}) → Lying={'c1' if lying_is_1 else 'c0'}")
                print(f"         c_lying={np.round(cal['c_lying'],2)}, "
                      f"c_standing={np.round(cal['c_standing'],2)}")
        elif verbose:
            print(f"  [algo] 자세 군집 분리 안 됨(sep={sep:.2f}) → 자세 보정 기권(DL 유지)")

    # Shaking: 고신뢰 Shaking 윈도우에서 자이로 에너지/주파수 하한, 없으면 기본
    ms = hi & (lab == 'Shaking')
    if ms.sum() >= 5:
        idx = np.where(ms)[0]
        cal['shake_gyro_th'] = float(np.percentile([feats[i]['gyro_e'] for i in idx], 25))
        cal['shake_freq_th'] = float(np.percentile([feats[i]['peak_freq'] for i in idx], 25))
        cal['shake_enabled'] = True
    else:
        cal['shake_gyro_th'] = 2.5
        cal['shake_freq_th'] = 3.5
        cal['shake_enabled'] = False  # 보정 근거 없으면 Shaking 판정 비활성(보수적)
    return cal


def _decide(f, cal):
    """윈도우 피처 → (라벨, 확신도). 판정 불가면 (None, 0)."""
    motion, gyro_e = f['motion'], f['gyro_e']
    pf, pr = f['peak_freq'], f['peak_ratio']

    # 이동(보행/구보) 여부: 주기적 신호 + 충분한 강도
    is_locomotion = (motion >= cal['static_th'] and 0.8 <= pf <= 6.0 and pr >= 0.10)

    # 1) Shaking: 고주파 + 강한 주기성 + 큰 회전 (보정된 경우만)
    if cal['shake_enabled'] and pf >= cal['shake_freq_th'] and pr >= 0.15 \
       and gyro_e >= cal['shake_gyro_th'] and motion >= cal['static_th']:
        conf = min(1.0, 0.5 + pr)
        return 'Shaking', conf

    # 2) 이동이 아니면 자세(Lying/Standing) 판정. 서서 두리번거려 motion이 다소
    #    커도 '주기적 이동'만 아니면 자세로 본다(머리 움직임 허용). 회전이 크면 제외.
    #    centroid는 무감독 군집화로 깨끗하므로 '더 가까운 centroid'를 신뢰.
    if not is_locomotion and gyro_e < cal['shake_gyro_th']:
        cl, cs = cal['c_lying'], cal['c_standing']
        if cl is not None and cs is not None:
            sim_l = float(np.dot(f['orient'], cl))
            sim_s = float(np.dot(f['orient'], cs))
            best = max(sim_l, sim_s)
            margin = abs(sim_l - sim_s)
            if best >= 0.85 and margin >= 0.02:  # 한 자세에 명확히 정렬 + 더 가까운 쪽 존재
                lab = 'Lying' if sim_l > sim_s else 'Standing'
                conf = min(1.0, max(0.0, (best - 0.6) / 0.4))
                return lab, conf
        return None, 0.0  # 자세 보정 근거 부족/모호 → 기권

    # 3) 이동(Walking/Running): 목걸이 IMU + 무정답 상태에서 walk/run 주파수
    #    분리는 신뢰도가 낮아 기본 비활성. DL이 담당. (실험용 플래그로만 활성)
    if cal.get('allow_locomotion') and is_locomotion:
        lab = 'Running' if pf >= cal['walk_run_fb'] else 'Walking'
        conf = min(1.0, 0.45 + pr + min(motion, 5.0) / 20.0)
        return lab, conf

    return None, 0.0


def algorithmic_correct(df, hz, win_sec=1.0, stride_sec=0.5, verbose=True,
                        allow_locomotion=False, swap_posture=False):
    """DL 결과(df.pred_label/confidence)에 알고리즘 보정을 적용해 라벨 배열 반환.
    df 에는 accel_x/y/z, gyro_x/y/z 원본 컬럼이 있어야 한다."""
    need = ['accel_x', 'accel_y', 'accel_z', 'gyro_x', 'gyro_y', 'gyro_z']
    alt = {'accel_x': 'acc_x', 'accel_y': 'acc_y', 'accel_z': 'acc_z'}
    cols = {}
    for c in need:
        if c in df.columns:
            cols[c] = c
        elif alt.get(c) in df.columns:
            cols[c] = alt[c]
        else:
            if verbose:
                print(f"  [algo] 센서 컬럼 {c} 없음 → 알고리즘 보정 스킵")
            return df['pred_label'].astype(object).values.copy(), {}

    accel = df[[cols['accel_x'], cols['accel_y'], cols['accel_z']]].values.astype(float)
    gyro = df[[cols['gyro_x'], cols['gyro_y'], cols['gyro_z']]].values.astype(float)
    amag = np.linalg.norm(accel, axis=1)
    n = len(df)
    dl_lab = df['pred_label'].astype(object).values
    dl_conf = df['confidence'].astype(float).values if 'confidence' in df.columns else np.ones(n)

    win = max(8, int(round(win_sec * hz)))
    stride = max(1, int(round(stride_sec * hz)))
    wins = _windows(n, win, stride)

    feats = [_win_features(accel, gyro, amag, hz, s, e) for s, e in wins]
    win_dl_lab = [dl_lab[s] for s, e in wins]
    win_dl_conf = [float(dl_conf[s:e].mean()) for s, e in wins]
    cal = _calibrate(feats, win_dl_lab, win_dl_conf,
                     swap_posture=swap_posture, verbose=verbose)
    cal['allow_locomotion'] = allow_locomotion

    # 윈도우 판정 → 프레임별 투표 누적
    K = len(DOMAIN)
    idx = {c: i for i, c in enumerate(DOMAIN)}
    vote_conf = np.zeros((n, K))   # 클래스별 누적 확신도
    vote_cnt = np.zeros((n, K))    # 클래스별 동의 윈도우 수
    cover = np.zeros(n)            # 프레임을 덮은 윈도우 수
    for (s, e), f in zip(wins, feats):
        cover[s:e] += 1
        lab, conf = _decide(f, cal)
        if lab is not None and conf >= OVERRIDE_CONF:
            k = idx[lab]
            vote_conf[s:e, k] += conf
            vote_cnt[s:e, k] += 1

    out = dl_lab.copy()
    changed = 0
    per_class = {c: 0 for c in DOMAIN}
    for i in range(n):
        if cover[i] == 0 or vote_cnt[i].sum() == 0:
            continue
        k = int(vote_cnt[i].argmax())
        agree = vote_cnt[i, k] / cover[i]
        mean_conf = vote_conf[i, k] / max(vote_cnt[i, k], 1)
        if agree >= OVERRIDE_AGREE and mean_conf >= OVERRIDE_CONF:
            new = DOMAIN[k]
            # 자세 교정(Lying/Standing)은 DL이 '자세 or 미분류'로 본 프레임에만.
            # DL이 확신한 미세행동(Eating/Drinking/Sniffing 등)은 보존 → 머리 숙인
            # 정적 행동이 자세로 잘못 덮이는 것 방지.
            if new in ('Lying', 'Standing') and out[i] not in POSTURE_OVERRIDABLE:
                continue
            if out[i] != new:
                changed += 1
                per_class[new] += 1
            out[i] = new

    if verbose:
        cl = 'O' if cal.get('c_lying') is not None else 'X'
        cs = 'O' if cal.get('c_standing') is not None else 'X'
        print(f"  [algo] 보정 윈도우 {len(wins)}개, static_th={cal['static_th']:.2f}, "
              f"자세centroid(L/S)={cl}/{cs}, shaking={'on' if cal['shake_enabled'] else 'off'}, "
              f"locomotion={'on' if allow_locomotion else 'off(→DL)'}")
        print(f"  [algo] DL→알고리즘 변경 프레임: {changed:,} ({changed/n*100:.1f}%)  "
              f"세부: {{ {', '.join(f'{c}:{v}' for c, v in per_class.items() if v)} }}")
    return out, cal
