import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class RoundSTE(torch.autograd.Function):
    """
    라운딩 연산을 위한 STE(Straight-Through Estimator)입니다.
    순전파 시에는 입력 텐서를 반올림합니다.
    역전파 시에는 항등 함수처럼 동작합니다.
    """
    @staticmethod
    def forward(ctx, x):
        return torch.round(x)

    @staticmethod
    def backward(ctx, grad_output):
        return grad_output

class ClampSTE(torch.autograd.Function):
    """
    클램핑 연산을 위한 STE(Straight-Through Estimator)입니다.
    논문의 설명에 따라 Clip 함수를 "우회(bypass)"하도록 합니다.
    순전파 시에는 입력을 특정 범위로 클램핑합니다.
    역전파 시에는 항등 함수처럼 동작하여 기울기를 그대로 통과시킵니다.
    """
    @staticmethod
    def forward(ctx, x, min_val, max_val):
        return torch.clamp(x, min_val, max_val)

    @staticmethod
    def backward(ctx, grad_output):
        # 클램핑 연산이 있었던 위치의 기울기를 항등 함수처럼 그대로 통과시킵니다.
        return grad_output, None, None

class BitLinear(nn.Module):
    """
    BitNet b1.58 방법론에 기반한 BitLinear 레이어입니다.
    모든 가중치는 삼진(-1, 0, 1) 값을 가지며 활성화 값은 8비트로 양자화됩니다.
    이 구현은 입력 'x'가 이미 정규화되었다고 (예: RMSNorm 사용) 가정합니다.
    """
    def __init__(self, in_features, out_features, activation_bits=8, eps=1e-5):
        super(BitLinear, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.activation_bits = activation_bits
        # Qb는 양자화 경계값이며, 8비트의 경우 2^(8-1) = 128입니다.
        # 값은 [-Qb+eps, Qb-eps] 범위로 양자화됩니다.
        self.Qb = 2**(self.activation_bits - 1)
        self.eps = eps

        # 잠재 가중치, 표준 nn.Linear 가중치처럼 초기화됩니다.
        self.weight = nn.Parameter(torch.Tensor(out_features, in_features))
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))

    def forward(self, x):
        # 입력 x는 (batch_size, ..., in_features) 형태를 가질 것으로 예상됩니다.
        # x는 이미 정규화되었다고 (예: RMSNorm 사용) 가정합니다.

        # 1. 활성화 양자화 (토큰별 absmax 양자화)
        # scale_x는 논문의 gamma_a (입력 활성화의 토큰별 절대 최대값)에 해당합니다.
        scale_x = torch.max(torch.abs(x), dim=-1, keepdim=True)[0] + self.eps
        
        # x를 대략 [-Qb, Qb] 범위로 스케일링한 후 반올림합니다.
        x_rounded = RoundSTE.apply(x * self.Qb / scale_x)
        # STE를 사용하여 클램핑합니다.
        x_for_matmul = ClampSTE.apply(x_rounded, -self.Qb + self.eps, self.Qb - self.eps)


        # 2. 가중치 삼진화 (absmean 양자화)
        # scale_w는 논문2의 식3에서 gamma_w (가중치의 평균 절대값)에 해당합니다.
        # 논문1의 식12에서 beta와 유사합니다.
        scale_w = torch.mean(torch.abs(self.weight)) + self.eps
        
        # w_scaled_for_ternary는 논문2의 식1에서 W / gamma_w 입니다.
        w_scaled_for_ternary = self.weight / scale_w
        
        # 라운딩에 STE를 사용하여 가중치를 {-1, 0, 1}로 삼진화합니다.
        # w_ternary_candidate는 논문2의 식1에서 W_tilde에 해당합니다.
        w_rounded_ternary = RoundSTE.apply(w_scaled_for_ternary)
        # STE를 사용하여 클램핑합니다.
        w_for_matmul = ClampSTE.apply(w_rounded_ternary, -1.0, 1.0)
        
        # 3. 양자화된 활성화와 삼진화된 가중치를 사용한 선형 연산
        out_quant = F.linear(x_for_matmul, w_for_matmul)

        # 4. 역양자화
        output = out_quant * (scale_w * scale_x / self.Qb)
        
        return output

    def extra_repr(self):
        return f'in_features={self.in_features}, out_features={self.out_features}, activation_bits={self.activation_bits}'