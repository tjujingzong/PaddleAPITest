import argparse
import os
import hashlib
from datetime import datetime
from engineV2 import detect_device_type

import paddle

from . import APIConfig
from .base import APITestBase


class APITestGPUCustomDump(APITestBase):
    def __init__(
        self,
        api_config,
        dump_dir="gpu_custom_dump",
        test_amp=False,
    ):
        super().__init__(api_config)
        self.dump_dir = dump_dir
        self.test_amp = test_amp

    def _ensure_dirs(self, path):
        os.makedirs(path, exist_ok=True)

    def _to_tensor_list(self, x):
        if x is None:
            return None
        if isinstance(x, paddle.Tensor):
            return [x]
        if isinstance(x, (list, tuple)):
            tensors = [t for t in x if isinstance(t, paddle.Tensor)]
            return tensors or None
        return None

    def _dump_results(self, tag, output, grads):
        api_name = self.api_config.config.replace("/", "_").replace(" ", "_")
        dump_path = os.path.join(self.dump_dir, api_name)
        self._ensure_dirs(dump_path)

        out_list = self._to_tensor_list(output)
        grad_list = self._to_tensor_list(grads)

        key = f"{tag}-{api_name}"
        sha16 = hashlib.sha256(key.encode("utf-8")).hexdigest()[:16]
        file_prefix = f"{tag}-{api_name}-{sha16}"

        forward_path = None
        grad_path = None

        if out_list is not None:
            forward_path = os.path.join(dump_path, f"{file_prefix}_forward.pdtensor")
            paddle.save(out_list, forward_path)

        if grad_list is not None:
            grad_path = os.path.join(dump_path, f"{file_prefix}_grad.pdtensor")
            paddle.save(grad_list, grad_path)

        return forward_path, grad_path

    def _run_on_device(self, device_str):
        try:
            paddle.set_device(device_str)
        except Exception as e:
            print(f"[device set error] {device_str} -> {e}", flush=True)
            return None, None

        if not self.gen_paddle_input():
            print(f"[gen_paddle_input failed] device={device_str}", flush=True)
            return None, None

        try:
            if self.test_amp:
                with paddle.amp.auto_cast():
                    output = self.paddle_api(*tuple(self.paddle_args), **self.paddle_kwargs)
            else:
                output = self.paddle_api(*tuple(self.paddle_args), **self.paddle_kwargs)
        except Exception as err:
            print(f"[forward error] device={device_str}  {self.api_config.config}\n{err}", flush=True)
            return None, None

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

    def test(self):
        if self.need_skip():
            print("[Skip]", self.api_config.config, flush=True)
            return

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

        device_type = detect_device_type()
        try:
            if paddle.device.is_compiled_with_cuda():
                device_type = "gpu"
            elif paddle.device.is_compiled_with_xpu():
                device_type = "xpu"
            else:
                custom_types = paddle.device.get_all_custom_device_type()
                if custom_types:
                    device_type = custom_types[0]
        except Exception as e:
            print(f"[detect device error] {e}", flush=True)
            return

        if device_type is None:
            print("[no available device]", self.api_config.config, flush=True)
            return

        device_str = f"{device_type}:0"

        print(
            f"{datetime.now()} [Begin] {self.api_config.config}\n"
            f"  Device: {device_str}",
            flush=True,
        )

        out, grads = self._run_on_device(device_str)
        if out is None:
            print(f"[{device_str} execution failed]", self.api_config.config, flush=True)
        else:
            forward_path, grad_path = self._dump_results(device_type, out, grads)
            print(f"[{device_type} dump done]", self.api_config.config, flush=True)

            if forward_path is not None:
                try:
                    loaded_forward = paddle.load(forward_path)
                    print(f"[loaded forward] {forward_path}")
                    for i, t in enumerate(loaded_forward):
                        arr = t.numpy().flatten()
                        print(
                            f"  forward[{i}] shape={t.shape}, dtype={t.dtype}, "
                            f"first_values={arr[:10]}"
                        )
                except Exception as e:
                    print(f"[load forward error] {forward_path} -> {e}", flush=True)

            if grad_path is not None:
                try:
                    loaded_grads = paddle.load(grad_path)
                    print(f"[loaded grad] {grad_path}")
                    for i, t in enumerate(loaded_grads):
                        arr = t.numpy().flatten()
                        print(
                            f"  grad[{i}] shape={t.shape}, dtype={t.dtype}, "
                            f"first_values={arr[:10]}"
                        )
                except Exception as e:
                    print(f"[load grad error] {grad_path} -> {e}", flush=True)


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
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--api_config",
        type=str,
        required=True,
    )
    parser.add_argument(
        "--dump_dir",
        type=str,
        default="report/gpu_custom_dump",
    )
    parser.add_argument(
        "--test_amp",
        type=parse_bool,
        default=False,
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
    )
    try:
        case.test()
    finally:
        case.clear_tensor()
        del case
        del api_config


if __name__ == "__main__":
    main()
