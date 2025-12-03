import argparse
import os
from datetime import datetime

import paddle

from . import APIConfig
from .base import APITestBase


class APITestGPUCustomDump(APITestBase):
    """
    在 GPU 与自定义设备（如 XPU / 第三方定制卡）上运行同一 API case，
    计算前向 + 反向结果，并将结果以 npz 形式落盘。
    """

    def __init__(
        self,
        api_config,
        dump_dir="report/gpu_custom_dump",
        test_amp=False,
        gpu_id=0,
        custom_device_type=None,
        custom_device_id=0,
    ):
        super().__init__(api_config)
        self.dump_dir = dump_dir
        self.test_amp = test_amp
        self.gpu_id = gpu_id
        self.custom_device_type = custom_device_type
        self.custom_device_id = custom_device_id

    # -------------------- 设备与落盘相关工具函数 --------------------
    def _ensure_dirs(self, path):
        os.makedirs(path, exist_ok=True)

    def _to_tensor_list(self, x):
        """将输出 / 梯度统一转换成 Tensor 列表，便于直接序列化保存。"""
        if x is None:
            return None
        if isinstance(x, paddle.Tensor):
            return [x]
        if isinstance(x, (list, tuple)):
            tensors = [t for t in x if isinstance(t, paddle.Tensor)]
            return tensors or None
        return None

    def _dump_results(self, tag, output, grads):
        """
        将指定设备的前向 / 反向结果直接保存为 Tensor 列表（使用 paddle.save）：
          <dump_dir>/<sanitized_api_name>/{tag}_forward.pdtensor
          <dump_dir>/<sanitized_api_name>/{tag}_grad.pdtensor
        """
        api_name = self.api_config.config.replace("/", "_").replace(" ", "_")
        dump_path = os.path.join(self.dump_dir, api_name)
        self._ensure_dirs(dump_path)

        out_list = self._to_tensor_list(output)
        grad_list = self._to_tensor_list(grads)

        if out_list is not None:
            paddle.save(out_list, os.path.join(dump_path, f"{tag}_forward.pdtensor"))
        if grad_list is not None:
            paddle.save(grad_list, os.path.join(dump_path, f"{tag}_grad.pdtensor"))

    def _run_on_device(self, device_str):
        """
        在指定设备上运行一次前向 + 反向，返回 (output, grads)。
        device_str 形如：'gpu:0', 'xpu:0', 'iluvatar_gpu:0' 等。
        """
        import paddle

        try:
            paddle.set_device(device_str)
        except Exception as e:
            print(f"[device set error] {device_str} -> {e}", flush=True)
            return None, None

        if not self.gen_paddle_input():
            print(f"[gen_paddle_input failed] device={device_str}", flush=True)
            return None, None

        # 前向
        try:
            if self.test_amp:
                with paddle.amp.auto_cast():
                    output = self.paddle_api(*tuple(self.paddle_args), **self.paddle_kwargs)
            else:
                output = self.paddle_api(*tuple(self.paddle_args), **self.paddle_kwargs)
        except Exception as err:
            print(f"[forward error] device={device_str}  {self.api_config.config}\n{err}", flush=True)
            return None, None

        # 反向
        out_grads = None
        if self.need_check_grad():
            inputs_list = self.get_paddle_input_list()
            try:
                result_outputs, result_outputs_grads = self.gen_paddle_output_and_output_grad(output)
            except Exception as grad_prepare_err:
                print(
                    f"[backward prepare error] device={device_str}  {self.api_config.config}\n{grad_prepare_err}",
                    flush=True,
                )
                return output, None

            if inputs_list and result_outputs and result_outputs_grads:
                try:
                    out_grads = paddle.grad(
                        result_outputs,
                        inputs_list,
                        grad_outputs=result_outputs_grads,
                        allow_unused=True,
                    )
                except Exception as grad_err:
                    print(
                        f"[backward error] device={device_str}  {self.api_config.config}\n{grad_err}",
                        flush=True,
                    )
                    out_grads = None
            else:
                print(
                    f"[backward skip] device={device_str} no valid inputs or outputs for gradient computation",
                    flush=True,
                )

        return output, out_grads

    # -------------------- 主流程：GPU vs Custom 设备 --------------------
    def test(self):
        # 1. 是否跳过
        if self.need_skip():
            print("[Skip]", self.api_config.config, flush=True)
            return

        # 2. 解析 Paddle API 信息 & 生成 numpy 输入
        if not self.ana_paddle_api_info():
            print("[ana_paddle_api_info failed]", self.api_config.config, flush=True)
            return

        try:
            if not self.gen_numpy_input():
                print("[gen_numpy_input failed]", self.api_config.config, flush=True)
                return
        except Exception as err:
            print("[numpy error]", self.api_config.config, "\n", str(err), flush=True)
            return

        # 3. 确定 GPU / 自定义设备字符串
        gpu_device_str = f"gpu:{self.gpu_id}"

        if self.custom_device_type is None:
            # 自动探测：优先 XPU，再尝试自定义设备
            try:
                if paddle.device.is_compiled_with_xpu():
                    self.custom_device_type = "xpu"
                else:
                    custom_types = paddle.device.get_all_custom_device_type()
                    if custom_types:
                        self.custom_device_type = custom_types[0]
                    else:
                        print(
                            "[no custom device available] "
                            "compiled_without_xpu and no custom_device_type found.",
                            self.api_config.config,
                            flush=True,
                        )
                        return
            except Exception as e:
                print(f"[detect custom device error] {e}", flush=True)
                return

        custom_device_str = (
            f"{self.custom_device_type}:{self.custom_device_id}"
            if self.custom_device_type != "xpu"
            else f"xpu:{self.custom_device_id}"
        )

        print(
            f"{datetime.now()} [Begin] {self.api_config.config}\n"
            f"  GPU device   : {gpu_device_str}\n"
            f"  Custom device: {custom_device_str}",
            flush=True,
        )

        # 4. GPU 上运行
        gpu_out, gpu_grads = self._run_on_device(gpu_device_str)
        if gpu_out is None:
            print("[gpu execution failed]", self.api_config.config, flush=True)
        else:
            self._dump_results("gpu", gpu_out, gpu_grads)
            print("[gpu dump done]", self.api_config.config, flush=True)

        # 5. 自定义设备 / XPU 上运行
        custom_out, custom_grads = self._run_on_device(custom_device_str)
        if custom_out is None:
            print(f"[{custom_device_str} execution failed]", self.api_config.config, flush=True)
        else:
            tag = self.custom_device_type if self.custom_device_type is not None else "custom"
            self._dump_results(tag, custom_out, custom_grads)
            print(f"[{tag} dump done]", self.api_config.config, flush=True)


def parse_bool(v):
    if isinstance(v, bool):
        return v
    s = str(v).lower()
    if s in {"true", "1", "yes", "y"}:
        return True
    if s in {"false", "0", "no", "n"}:
        return False
    raise argparse.ArgumentTypeError(f"Invalid bool value: {v}")


def main():
    parser = argparse.ArgumentParser(
        description="在 GPU / 自定义设备 上运行 API case，并将前向 + 反向结果以 npz 落盘。"
    )
    parser.add_argument(
        "--api_config",
        type=str,
        required=True,
        help="单条 API 配置（与 engine 中的 api_config 字符串格式一致）",
    )
    parser.add_argument(
        "--dump_dir",
        type=str,
        default="report/gpu_custom_dump",
        help="结果保存目录（npz 文件会按 API 配置分子目录存放）",
    )
    parser.add_argument(
        "--test_amp",
        type=parse_bool,
        default=False,
        help="是否在前向中启用 AMP 自动混合精度",
    )
    parser.add_argument(
        "--gpu_id",
        type=int,
        default=0,
        help="使用的 GPU 设备号（形如 gpu:<gpu_id>）",
    )
    parser.add_argument(
        "--custom_device_type",
        type=str,
        default=None,
        help="自定义设备类型名称，例如 'xpu'、'iluvatar_gpu' 等；"
        "留空则自动探测：优先 XPU，再尝试 paddle 自定义设备。",
    )
    parser.add_argument(
        "--custom_device_id",
        type=int,
        default=0,
        help="自定义设备 ID，如 xpu:0 / iluvatar_gpu:0 中的 0",
    )

    args = parser.parse_args()

    print(f"Options: {vars(args)}", flush=True)

    try:
        api_config = APIConfig(args.api_config.strip())
    except Exception as err:
        print(f"[config parse error] {args.api_config} {str(err)}", flush=True)
        return

    case = APITestGPUCustomDump(
        api_config,
        dump_dir=args.dump_dir,
        test_amp=args.test_amp,
        gpu_id=args.gpu_id,
        custom_device_type=args.custom_device_type,
        custom_device_id=args.custom_device_id,
    )
    try:
        case.test()
    finally:
        case.clear_tensor()
        del case
        del api_config


if __name__ == "__main__":
    main()


