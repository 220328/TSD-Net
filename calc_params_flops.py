# -*- coding: utf-8 -*-

"""
Calculate Params / MACs / FLOPs for DCTNet

Run:
    python calc_params_flops.py

Dependencies:
    pip install thop fvcore
"""

import torch
import torch.nn as nn

from thop import profile, clever_format


# ============================================================
# 1. Model import
# ============================================================

# 根据你的 DCTNet.py 中实际类名二选一
try:
    from methods.DCTNet_clean_for_complexity import DCTNet as Model
    print("Load model class: DCTNet")
except ImportError:
    from methods.DCTNet import Network as Model
    print("Load model class: Network")


# ============================================================
# 2. Settings
# ============================================================

# 必须和论文测试时的输入尺寸一致
INPUT_H = 384
INPUT_W = 384

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


# ============================================================
# 3. Wrapper
# ============================================================

class ModelWrapper(nn.Module):
    """
    用于解决：
    1. 模型存在 test_forward()
    2. 模型输出 tuple/list/dict
    3. FLOPs 工具只希望得到一个 Tensor 输出
    """

    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, x):

        # 优先使用测试阶段 forward
        if hasattr(self.model, "test_forward"):
            out = self.model.test_forward(x)
        else:
            out = self.model(x)

        # ----------------------------------------------------
        # tuple / list
        # ----------------------------------------------------
        if isinstance(out, (tuple, list)):

            for item in out:
                if torch.is_tensor(item):
                    return item

            raise RuntimeError(
                "Model output is tuple/list, but no Tensor was found."
            )

        # ----------------------------------------------------
        # dict
        # ----------------------------------------------------
        if isinstance(out, dict):

            for key, value in out.items():
                if torch.is_tensor(value):
                    return value

            raise RuntimeError(
                "Model output is dict, but no Tensor was found."
            )

        return out


# ============================================================
# 4. Count parameters manually
# ============================================================

def count_parameters(model):

    total_params = sum(
        p.numel()
        for p in model.parameters()
    )

    trainable_params = sum(
        p.numel()
        for p in model.parameters()
        if p.requires_grad
    )

    return total_params, trainable_params


# ============================================================
# 5. Main
# ============================================================

def main():

    print("\n" + "=" * 70)
    print("Model Complexity Analysis")
    print("=" * 70)

    print(f"Device     : {DEVICE}")
    print(f"Input size : 1 x 3 x {INPUT_H} x {INPUT_W}")

    # --------------------------------------------------------
    # Build model
    # --------------------------------------------------------

    model = Model()

    model = model.to(DEVICE)
    model.eval()

    wrapper = ModelWrapper(model)
    wrapper = wrapper.to(DEVICE)
    wrapper.eval()

    x = torch.randn(
        1,
        3,
        INPUT_H,
        INPUT_W
    ).to(DEVICE)

    # ========================================================
    # Parameter count
    # ========================================================

    total_params, trainable_params = count_parameters(model)

    print("\n[Parameters]")

    print(
        "Total Params     : {:.4f} M".format(
            total_params / 1e6
        )
    )

    print(
        "Trainable Params : {:.4f} M".format(
            trainable_params / 1e6
        )
    )

    # ========================================================
    # THOP
    # ========================================================

    print("\n" + "-" * 70)
    print("[THOP]")
    print("-" * 70)

    with torch.no_grad():

        macs, params = profile(
            wrapper,
            inputs=(x,),
            verbose=False
        )

    # THOP 返回的是 MACs
    flops = 2 * macs

    macs_str, params_str = clever_format(
        [macs, params],
        "%.4f"
    )

    print(f"THOP Params : {params_str}")

    print(
        "MACs        : {:.4f} G".format(
            macs / 1e9
        )
    )

    print(
        "FLOPs*      : {:.4f} G".format(
            flops / 1e9
        )
    )

    print(
        "* FLOPs = 2 × MACs"
    )

    # ========================================================
    # fvcore
    # ========================================================

    print("\n" + "-" * 70)
    print("[FVCore cross-check]")
    print("-" * 70)

    try:

        from fvcore.nn import FlopCountAnalysis

        with torch.no_grad():

            flop_analyzer = FlopCountAnalysis(
                wrapper,
                x
            )

            # 防止大量 warning 干扰
            flop_analyzer.unsupported_ops_warnings(False)
            flop_analyzer.uncalled_modules_warnings(False)

            fvcore_flops = flop_analyzer.total()

        print(
            "FVCore FLOPs : {:.4f} G".format(
                fvcore_flops / 1e9
            )
        )

        unsupported = flop_analyzer.unsupported_ops()

        if len(unsupported) > 0:

            print("\nUnsupported operators:")

            for op, count in unsupported.items():
                print(
                    "  {:40s}: {}".format(
                        str(op),
                        count
                    )
                )

        else:
            print(
                "\nNo unsupported operators detected."
            )

    except Exception as e:

        print(
            "FVCore calculation failed:"
        )

        print(e)

    # ========================================================
    # Final
    # ========================================================

    print("\n" + "=" * 70)
    print("Recommended values for paper")
    print("=" * 70)

    print(
        "Params : {:.2f} M".format(
            total_params / 1e6
        )
    )

    print(
        "MACs   : {:.2f} G".format(
            macs / 1e9
        )
    )

    print(
        "FLOPs  : {:.2f} G  (if 1 MAC = 2 FLOPs)".format(
            flops / 1e9
        )
    )

    print("=" * 70)


if __name__ == "__main__":
    main()