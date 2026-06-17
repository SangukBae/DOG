"""
MS-TCN++ 모델
- Stage 1: Dual Dilated Layer (DDL) 사용 (예측 생성)
- Stage 2~N: 일반 Dilated Residual Layer (정제)
- 논문: MS-TCN++, Li et al. 2020
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


# ── 일반 Dilated Residual Layer (정제 스테이지용) ─────────────────────────────
class DilatedResidualLayer(nn.Module):
    def __init__(self, dilation, in_channels, out_channels, kernel_size):
        super().__init__()
        padding = dilation * (kernel_size - 1) // 2
        self.conv = nn.Conv1d(in_channels, out_channels, kernel_size,
                              padding=padding, dilation=dilation)
        self.norm = nn.LayerNorm(out_channels)
        self.res_conv = nn.Conv1d(in_channels, out_channels, 1)

    def forward(self, x):
        out = F.relu(self.conv(x))
        out = self.norm(out.transpose(1, 2)).transpose(1, 2)
        return out + self.res_conv(x)


# ── Dual Dilated Layer (예측 생성 스테이지용) ─────────────────────────────────
class DualDilatedLayer(nn.Module):
    """
    한 레이어에서 두 개의 dilation을 병렬로 적용
    Conv A: dilation = 2^l      (점점 멀리)
    Conv B: dilation = 2^(L-l)  (점점 가까이)
    → concat → 1x1 conv로 합침
    """
    def __init__(self, dilation_a, dilation_b, in_channels, out_channels, kernel_size):
        super().__init__()
        padding_a = dilation_a * (kernel_size - 1) // 2
        padding_b = dilation_b * (kernel_size - 1) // 2

        self.conv_a = nn.Conv1d(in_channels, out_channels, kernel_size,
                                padding=padding_a, dilation=dilation_a)
        self.conv_b = nn.Conv1d(in_channels, out_channels, kernel_size,
                                padding=padding_b, dilation=dilation_b)

        # concat 후 2*out_channels → out_channels
        self.conv_merge = nn.Conv1d(out_channels * 2, out_channels, 1)
        self.norm = nn.LayerNorm(out_channels)
        self.res_conv = nn.Conv1d(in_channels, out_channels, 1)

    def forward(self, x):
        a = self.conv_a(x)
        b = self.conv_b(x)
        out = F.relu(torch.cat([a, b], dim=1))  # (B, 2C, T)
        out = self.conv_merge(out)               # (B, C, T)
        out = self.norm(out.transpose(1, 2)).transpose(1, 2)
        return out + self.res_conv(x)


# ── 예측 생성 스테이지 (DDL 사용) ─────────────────────────────────────────────
class PredictionStage(nn.Module):
    def __init__(self, num_layers, num_f_maps, in_channels, num_classes, kernel_size, dropout):
        super().__init__()
        self.conv_in = nn.Conv1d(in_channels, num_f_maps, 1)
        self.layers = nn.ModuleList([
            DualDilatedLayer(
                dilation_a=2 ** i,
                dilation_b=2 ** (num_layers - i),
                in_channels=num_f_maps,
                out_channels=num_f_maps,
                kernel_size=kernel_size
            )
            for i in range(num_layers)
        ])
        self.dropout = nn.Dropout(dropout)
        self.conv_out = nn.Conv1d(num_f_maps, num_classes, 1)

    def forward(self, x):
        out = self.conv_in(x)
        for layer in self.layers:
            out = layer(out)
        out = self.dropout(out)
        return self.conv_out(out)  # (B, num_classes, T)


# ── 정제 스테이지 (일반 Dilated Conv 사용) ────────────────────────────────────
class RefinementStage(nn.Module):
    def __init__(self, num_layers, num_f_maps, in_channels, num_classes, kernel_size, dropout):
        super().__init__()
        self.conv_in = nn.Conv1d(in_channels, num_f_maps, 1)
        self.layers = nn.ModuleList([
            DilatedResidualLayer(2 ** i, num_f_maps, num_f_maps, kernel_size)
            for i in range(num_layers)
        ])
        self.dropout = nn.Dropout(dropout)
        self.conv_out = nn.Conv1d(num_f_maps, num_classes, 1)

    def forward(self, x):
        out = self.conv_in(x)
        for layer in self.layers:
            out = layer(out)
        out = self.dropout(out)
        return self.conv_out(out)  # (B, num_classes, T)


# ── MS-TCN++ ──────────────────────────────────────────────────────────────────
class MSTCNPlusPlus(nn.Module):
    """
    Stage 1: PredictionStage (DDL)
    Stage 2~N: RefinementStage (일반 dilated conv)
    스테이지 간 입력: 확률값만 전달 (특징벡터 X)
    """
    def __init__(self, num_stages, num_layers, num_f_maps,
                 in_channels, num_classes, kernel_size, dropout):
        super().__init__()
        assert num_stages >= 2, "num_stages는 최소 2 이상이어야 합니다"

        # Stage 1: 예측 생성 (DDL)
        self.prediction_stage = PredictionStage(
            num_layers, num_f_maps, in_channels, num_classes, kernel_size, dropout
        )

        # Stage 2~N: 정제 (일반 dilated conv)
        # 입력: num_classes (확률값), 출력: num_classes
        self.refinement_stages = nn.ModuleList([
            RefinementStage(
                num_layers, num_f_maps, num_classes, num_classes, kernel_size, dropout
            )
            for _ in range(num_stages - 1)
        ])

    def forward(self, x):
        # Stage 1
        out = self.prediction_stage(x)
        outputs = [out]

        # Stage 2~N: 확률값만 다음 스테이지로 전달
        for stage in self.refinement_stages:
            out = stage(F.softmax(out, dim=1))
            outputs.append(out)

        return outputs  # list of (B, num_classes, T), 마지막이 최종 예측


# ── BiLSTM ────────────────────────────────────────────────────────────────────
class BiLSTM(nn.Module):
    """
    Conv feature extractor + Bidirectional LSTM + FC
    conv_feat: Conv → ReLU → Conv → ReLU (BatchNorm 없음)
    """
    def __init__(self, in_channels, num_classes, num_f_maps=64,
                 kernel_size=5, dropout=0.5,
                 hidden_size=128, lstm_layers=2, conv_layers=2, **kwargs):
        super().__init__()

        # Conv feature extractor: Conv → ReLU → Dropout → Conv → ReLU → Dropout
        conv = []
        ch = in_channels
        for _ in range(conv_layers):
            conv += [
                nn.Conv1d(ch, num_f_maps, kernel_size, padding=kernel_size//2),
                nn.ReLU(),
                nn.Dropout(dropout),
            ]
            ch = num_f_maps
        self.conv_feat = nn.Sequential(*conv)

        # BiLSTM
        self.lstm = nn.LSTM(
            input_size=num_f_maps,
            hidden_size=hidden_size,
            num_layers=lstm_layers,
            batch_first=True,
            bidirectional=True,
            dropout=dropout if lstm_layers > 1 else 0,
        )

        self.fc = nn.Linear(hidden_size * 2, num_classes)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        # x: (B, C, T)
        feat = self.conv_feat(x)           # (B, F, T)
        feat = feat.permute(0, 2, 1)       # (B, T, F)
        out, _ = self.lstm(feat)           # (B, T, 2H)
        out = self.dropout(out)
        out = self.fc(out)                 # (B, T, num_classes)
        out = out.permute(0, 2, 1)         # (B, num_classes, T)
        return [out]
