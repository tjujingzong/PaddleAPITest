import hashlib
import json
import os
import subprocess
import tempfile
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np
import paddle

from .api_config import APIConfig
from .api_config.log_writer import write_to_log
from .paddle_device_vs_cpu import APITestCustomDeviceVSCPU

# ========== 两阶段流水线配置参数 ==========
# 预下载进程池大小
PRE_DOWNLOAD_WORKERS = 4
# GPU 计算进程池轮询等待时间（秒）
POLL_INTERVAL = 1.0
# 最大等待时间（秒），超过此时间仍未下载完成则报错
MAX_WAIT_TIME = 3600
# 本地 cache 目录
CACHE_DIR = Path(tempfile.gettempdir()) / "paddle_device_vs_gpu_cache"
# ==========================================


# ========== 全局文件状态管理 ==========
# 直接通过文件系统检查文件是否存在，无需复杂的共享状态


def _get_local_cache_path(filename):
    """获取本地 cache 路径"""
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    return CACHE_DIR / filename


def _download_worker(args):
    """预下载 worker 函数（在独立进程中运行）"""
    filename, bos_path, bcecmd_path, bos_conf_path = args
    
    # 确保路径都是字符串
    bcecmd_path = str(bcecmd_path)
    bos_conf_path = str(bos_conf_path)
    
    # 检查是否已下载
    local_path = _get_local_cache_path(filename)
    if local_path.exists():
        # 使用文件系统标记（因为多进程间 Manager 可能不共享）
        print(f"[pre-download] File already exists: {local_path}", flush=True)
        return filename, True
    
    # 构建 BOS 路径
    cleaned = bos_path.strip().lstrip("/").rstrip("/")
    remote_path = f"bos:/{cleaned}/{filename}"
    
    # 执行下载
    cmd = [
        bcecmd_path,
        "--conf-path",
        bos_conf_path,
        "bos",
        "cp",
        remote_path,
        str(local_path),
    ]
    
    try:
        # 打印完整命令以便调试
        cmd_str = " ".join(cmd)
        print(f"[pre-download] Downloading {filename}...", flush=True)
        print(f"[pre-download] Command: {cmd_str}", flush=True)
        print(f"[pre-download] Target path: {local_path}", flush=True)
        
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=300)
        
        # 打印命令输出以便调试
        if result.stdout:
            print(f"[pre-download] stdout: {result.stdout[:200]}", flush=True)
        if result.stderr:
            print(f"[pre-download] stderr: {result.stderr[:200]}", flush=True)
        
        if result.returncode == 0:
            # 验证文件确实存在
            if local_path.exists():
                print(f"[pre-download] Successfully downloaded: {filename} -> {local_path}", flush=True)
                return filename, True
            else:
                print(f"[pre-download] Download command succeeded but file not found: {local_path}", flush=True)
                return filename, False
        else:
            print(
                f"[pre-download] Failed to download {filename}, returncode: {result.returncode}, stderr: {result.stderr}",
                flush=True,
            )
            return filename, False
    except Exception as e:
        print(f"[pre-download] Error downloading {filename}: {e}", flush=True)
        import traceback
        print(f"[pre-download] Traceback: {traceback.format_exc()}", flush=True)
        return filename, False


def start_pre_download_pool(api_config_file, target_device_type, random_seed, 
                            bos_path, bcecmd_path, bos_conf_path):
    """启动预下载进程池
    
    Args:
        api_config_file: API 配置文件路径
        target_device_type: 目标设备类型
        random_seed: 随机种子
        bos_path: BOS 路径
        bcecmd_path: bcecmd 路径
        bos_conf_path: bcecmd 配置文件路径
    """
    if not api_config_file or not os.path.exists(api_config_file):
        print(f"[pre-download] No valid api_config_file provided, skip pre-download", flush=True)
        return
    
    # 确保路径都是字符串
    bcecmd_path = str(bcecmd_path)
    bos_conf_path = str(bos_conf_path)
    
    print(f"[pre-download] Starting pre-download pool with {PRE_DOWNLOAD_WORKERS} workers", flush=True)
    print(f"[pre-download] Cache directory: {CACHE_DIR}", flush=True)
    print(f"[pre-download] BOS path: {bos_path}", flush=True)
    print(f"[pre-download] bcecmd path: {bcecmd_path}", flush=True)
    print(f"[pre-download] bcecmd conf path: {bos_conf_path}", flush=True)
    
    # 读取配置文件，计算所有需要的文件名
    filenames = []
    try:
        with open(api_config_file, "r") as f:
            config_lines = [line.strip() for line in f if line.strip()]
        
        for config_line in config_lines:
            try:
                api_config = APIConfig(config_line)
                # 计算文件名（需要模拟 APITestPaddleDeviceVSGPU 的逻辑）
                config_str = json.dumps(
                    {
                        "api_name": api_config.api_name,
                        "args": [str(arg) for arg in api_config.args],
                        "kwargs": {k: str(v) for k, v in api_config.kwargs.items()},
                    },
                    sort_keys=True,
                )
                config_hash = hashlib.md5(config_str.encode()).hexdigest()[:16]
                filename = f"{target_device_type}-{random_seed}-{config_hash}.pdtensor"
                filenames.append(filename)
            except Exception as e:
                print(f"[pre-download] Failed to parse config line: {config_line[:100]}, error: {e}", flush=True)
                continue
        
        print(f"[pre-download] Found {len(filenames)} files to download", flush=True)
        
        # 去重：确保每个文件只下载一次（多个进程不会下载同一个文件）
        unique_filenames = list(set(filenames))
        if len(unique_filenames) != len(filenames):
            print(f"[pre-download] Removed {len(filenames) - len(unique_filenames)} duplicate filenames", flush=True)
        
        # 准备下载任务参数（每个文件一个任务，PRE_DOWNLOAD_WORKERS 个进程并行下载不同文件）
        download_tasks = [
            (filename, bos_path, bcecmd_path, bos_conf_path)
            for filename in unique_filenames
        ]
        
        print(f"[pre-download] Prepared {len(download_tasks)} download tasks", flush=True)
        
        # 启动进程池执行下载
        with ProcessPoolExecutor(max_workers=PRE_DOWNLOAD_WORKERS) as executor:
            futures = {executor.submit(_download_worker, task): task[0] for task in download_tasks}
            
            completed = 0
            failed = 0
            for future in as_completed(futures):
                filename = futures[future]
                try:
                    _, success = future.result()
                    if success:
                        completed += 1
                    else:
                        failed += 1
                except Exception as e:
                    print(f"[pre-download] Exception for {filename}: {e}", flush=True)
                    failed += 1
            
            print(f"[pre-download] Pre-download completed: {completed} succeeded, {failed} failed", flush=True)
    
    except Exception as e:
        print(f"[pre-download] Error in pre-download pool: {e}", flush=True)


class APITestPaddleDeviceVSGPU(APITestCustomDeviceVSCPU):
    def __init__(self, api_config, **kwargs):
        # 继承 CustomDevice vs CPU 的基本功能
        super().__init__(api_config, **kwargs)

        # 新增参数
        self.operation_mode = kwargs.get("operation_mode", None)
        self.bos_path = kwargs.get("bos_path", "")
        self.target_device_type = kwargs.get("target_device_type", "")
        self.random_seed = kwargs.get("random_seed", 0)
        self.atol = kwargs.get("atol", 1e-2)
        self.rtol = kwargs.get("rtol", 1e-2)
        self.bcecmd_path = Path(kwargs.get("bcecmd_path", "./bcecmd")).resolve()
        self.bos_conf_path = kwargs.get("bos_conf_path", "./conf")

        # 设置随机种子确保一致性
        if self.random_seed != 0:
            np.random.seed(self.random_seed)
            paddle.seed(self.random_seed)

    def _get_config_hash(self):
        """生成API配置的哈希值，用于文件名"""
        config_str = json.dumps(
            {
                "api_name": self.api_config.api_name,
                "args": [str(arg) for arg in self.api_config.args],
                "kwargs": {k: str(v) for k, v in self.api_config.kwargs.items()},
            },
            sort_keys=True,
        )
        return hashlib.md5(config_str.encode()).hexdigest()[:16]

    def _get_local_device_type(self):
        """获取当前设备的类型，优先复用 engineV2 的检测逻辑。"""
        from engineV2 import detect_device_type
        return detect_device_type()

    def _get_filename(self, device_type=None):
        """生成PDTensor文件名"""
        if device_type is None:
            device_type = self._get_local_device_type()
        return f"{device_type}-{self.random_seed}-{self._get_config_hash()}.pdtensor"

    def _save_tensor_locally(self, output, grads=None):
        """保存结果到本地PDTensor文件"""
        # 保存到临时文件
        temp_dir = tempfile.gettempdir()
        local_path = Path(temp_dir) / self._get_filename()

        # 使用paddle.save保存张量数据
        save_data = {"output": output}
        if grads is not None:
            save_data["grads"] = grads

        paddle.save(save_data, str(local_path))
        print(f"[upload] Saved pdtensor file: {local_path}", flush=True)
        return local_path

    def _build_bos_path(self, filename: str) -> str:
        cleaned = self.bos_path.strip().lstrip("/").rstrip("/")
        return f"bos:/{cleaned}/{filename}"

    def _bcecmd_cp(self, src: str, dst: str, action: str):
        """使用指定的 bcecmd 命令执行 cp 操作"""
        cmd = [
            str(self.bcecmd_path),
            "--conf-path",
            self.bos_conf_path,
            "bos",
            "cp",
            src,
            dst,
        ]
        print(f"[{action}] Running command: {' '.join(cmd)}", flush=True)
        return subprocess.run(cmd, capture_output=True, text=True, timeout=300)

    def _upload_to_bos(self, local_path):
        """使用 bcecmd 上传文件到 BOS"""
        if not self.bos_path:
            print(f"[upload] No bos_path specified, skip upload", flush=True)
            return

        remote_path = self._build_bos_path(local_path.name)
        try:
            result = self._bcecmd_cp(str(local_path), remote_path, "upload")
            if result.returncode == 0:
                print(f"[upload] Upload succeeded: {remote_path}", flush=True)
                local_path.unlink(missing_ok=True)
            else:
                print(
                    f"[upload] Upload failed: {remote_path}, stderr: {result.stderr}",
                    flush=True,
                )
        except Exception as e:
            print(f"[upload] Upload failed: {e}", flush=True)

    def _download_from_bos(self, filename):
        """使用 bcecmd 从 BOS 下载文件"""
        if not self.bos_path:
            print(f"[download] No bos_path specified, skip download", flush=True)
            return None

        temp_dir = tempfile.gettempdir()
        local_path = Path(temp_dir) / filename

        if local_path.exists():
            print(f"[download] File already exists locally: {local_path}", flush=True)
            return local_path

        remote_path = self._build_bos_path(filename)
        try:
            result = self._bcecmd_cp(remote_path, str(local_path), "download")
            if result.returncode == 0:
                print(f"[download] Download succeeded: {local_path}", flush=True)
                return local_path
            else:
                print(
                    f"[download] Download failed: {remote_path}, stderr: {result.stderr}",
                    flush=True,
                )
                return None
        except Exception as e:
            print(f"[download] Download failed: {e}", flush=True)
            return None

    def _run_paddle(self, device_type: str):
        """在指定设备上运行 Paddle（统一 GPU / XPU / 自定义设备逻辑）。"""
        try:
            paddle_device_type = device_type
            if device_type == "gpu":
                # engineV2.py sets CUDA_VISIBLE_DEVICES, so paddle will use the correct GPU.
                paddle.set_device("gpu")
            elif device_type == "xpu":
                paddle.set_device(f"xpu:{self.xpu_device_id}")
            elif device_type == self.custom_device_type and self.check_custom_device_available():
                paddle.set_device(f"{self.custom_device_type}:{self.custom_device_id}")
            elif device_type == "cpu":
                paddle.set_device("cpu")
            else:
                print(f"[error] No custom device available", flush=True)
                return None, None

            if not self.ana_paddle_api_info():
                print("ana_paddle_api_info failed", flush=True)
                return None, None

            if not self.gen_numpy_input():
                print("gen_numpy_input failed", flush=True)
                return None, None

            if not self.gen_paddle_input():
                print("gen_paddle_input failed", flush=True)
                return None, None

            paddle_output = self.paddle_api(
                *tuple(self.paddle_args), **self.paddle_kwargs
            )

            paddle_grads = None
            if self.need_check_grad():
                inputs_list = self.get_paddle_input_list()
                result_outputs, result_outputs_grads = (
                    self.gen_paddle_output_and_output_grad(paddle_output)
                )
                if inputs_list and result_outputs and result_outputs_grads:
                    paddle_grads = paddle.grad(
                        outputs=result_outputs,
                        inputs=inputs_list,
                        grad_outputs=result_outputs_grads,
                        allow_unused=True,
                    )

            return paddle_output, paddle_grads

        except Exception as e:
            print(
                f"[paddle {paddle_device_type} error] {self.api_config.config}: {e}",
                flush=True,
            )
            write_to_log("paddle_error", self.api_config.config)
            return None, None

    def _compare_with_downloaded(self, local_output, local_grads, downloaded_tensor):
        """与下载的结果进行对比"""
        try:
            print(
                f"[compare] Comparing results for {self.api_config.config}", flush=True
            )

            # 加载下载的数据
            remote_data = paddle.load(str(downloaded_tensor))
            remote_output = remote_data["output"]

            # 对比Forward输出（直接使用Paddle对比）
            try:
                if isinstance(local_output, paddle.Tensor) and isinstance(
                    remote_output, paddle.Tensor
                ):
                    # 使用Paddle的对比方法
                    np.testing.assert_allclose(
                        local_output.numpy(),
                        remote_output.numpy(),
                        atol=self.atol,
                        rtol=self.rtol,
                        equal_nan=True,
                    )
                elif isinstance(local_output, (list, tuple)) and isinstance(
                    remote_output, (list, tuple)
                ):
                    # 列表或元组对比
                    for i, (local_item, remote_item) in enumerate(
                        zip(local_output, remote_output)
                    ):
                        if isinstance(local_item, paddle.Tensor) and isinstance(
                            remote_item, paddle.Tensor
                        ):
                            np.testing.assert_allclose(
                                local_item.numpy(),
                                remote_item.numpy(),
                                atol=self.atol,
                                rtol=self.rtol,
                                equal_nan=True,
                            )
                            print(
                                f"[compare] Forward output[{i}] comparison passed",
                                flush=True,
                            )
                else:
                    # 其他情况，尝试转换为numpy对比
                    local_np = (
                        local_output.numpy()
                        if isinstance(local_output, paddle.Tensor)
                        else np.array(local_output)
                    )
                    remote_np = (
                        remote_output.numpy()
                        if isinstance(remote_output, paddle.Tensor)
                        else np.array(remote_output)
                    )
                    np.testing.assert_allclose(
                        local_np,
                        remote_np,
                        atol=self.atol,
                        rtol=self.rtol,
                        equal_nan=True,
                    )

                print(
                    f"[compare] Forward accuracy check passed for {self.api_config.config}",
                    flush=True,
                )
            except Exception as e:
                print(
                    f"[compare] Forward accuracy check failed for {self.api_config.config}, error: {e}",
                    flush=True,
                )
                write_to_log("accuracy_error", self.api_config.config)
                return False

            # 对比Backward梯度（如果存在且Forward通过）
            if local_grads is not None and "grads" in remote_data:
                remote_grads = remote_data["grads"]

                try:
                    if isinstance(local_grads, (list, tuple)) and isinstance(
                        remote_grads, (list, tuple)
                    ):
                        for i, (local_grad, remote_grad) in enumerate(
                            zip(local_grads, remote_grads)
                        ):
                            if isinstance(local_grad, paddle.Tensor) and isinstance(
                                remote_grad, paddle.Tensor
                            ):
                                np.testing.assert_allclose(
                                    local_grad.numpy(),
                                    remote_grad.numpy(),
                                    atol=self.atol,
                                    rtol=self.rtol,
                                    equal_nan=True,
                                )
                                print(
                                    f"[compare] Backward gradient[{i}] comparison passed",
                                    flush=True,
                                )
                    elif isinstance(local_grads, paddle.Tensor) and isinstance(
                        remote_grads, paddle.Tensor
                    ):
                        np.testing.assert_allclose(
                            local_grads.numpy(),
                            remote_grads.numpy(),
                            atol=self.atol,
                            rtol=self.rtol,
                            equal_nan=True,
                        )

                    print(
                        f"[compare] Backward gradient check passed for {self.api_config.config}",
                        flush=True,
                    )
                except Exception as e:
                    print(
                        f"[compare] Backward gradient check failed for {self.api_config.config}, error: {e}",
                        flush=True,
                    )
                    return False

            print(
                f"[compare] Accuracy check passed for {self.api_config.config}",
                flush=True,
            )
            write_to_log("pass", self.api_config.config)
            return True

        except Exception as e:
            print(
                f"[compare] Comparison failed for {self.api_config.config}, error: {e}",
                flush=True,
            )
            write_to_log("accuracy_error", self.api_config.config)
            return False

    def test(self):
        """Main test function"""
        if self.operation_mode == "upload":
            self._test_upload_mode()
        elif self.operation_mode == "download":
            self._test_download_mode()
        else:
            print(
                "[error] operation_mode 不能为空，请指定 --operation_mode=upload 或 download",
                flush=True,
            )
            return

    def _test_upload_mode(self):
        """Upload模式：执行测试并上传结果"""
        print(f"[upload] Starting upload mode for {self.api_config.config}", flush=True)

        local_device_type = self._get_local_device_type()
        output, grads = self._run_paddle(local_device_type)

        if output is None:
            print(f"[upload] Execution failed for {self.api_config.config}", flush=True)
            return

        # 保存结果到本地PDTensor
        local_path = self._save_tensor_locally(output, grads)

        # 异步上传到BOS
        self._upload_to_bos(local_path)

        print(
            f"[upload] Upload mode completed for {self.api_config.config}", flush=True
        )

    def _test_download_mode(self):
        """Download模式：从本地 cache 查找文件并验证（不再主动下载）"""
        print(
            f"[download] Starting download mode for {self.api_config.config}",
            flush=True,
        )

        # 确定要查找的文件名
        target_filename = self._get_filename(self.target_device_type)
        
        # 从本地 cache 目录查找文件
        local_cache_path = _get_local_cache_path(target_filename)
        
        # 等待文件准备好（轮询等待）
        start_time = time.time()
        while not local_cache_path.exists():
            elapsed = time.time() - start_time
            if elapsed > MAX_WAIT_TIME:
                print(
                    f"[download] Timeout waiting for file {target_filename} after {MAX_WAIT_TIME}s",
                    flush=True,
                )
                return
            
            # 直接检查文件是否存在（文件系统是进程间共享的）
            
            print(
                f"[download] Waiting for file {target_filename} (elapsed: {elapsed:.1f}s)...",
                flush=True,
            )
            time.sleep(POLL_INTERVAL)
        
        print(
            f"[download] File ready: {local_cache_path}",
            flush=True,
        )

        # 在本地设备上执行测试
        local_device_type = self._get_local_device_type()
        local_output, local_grads = self._run_paddle(local_device_type)

        if local_output is None:
            print(
                f"[download] Local execution failed for {self.api_config.config}",
                flush=True,
            )
            return

        # 与下载的结果进行对比
        try:
            success = self._compare_with_downloaded(
                local_output, local_grads, local_cache_path
            )
        finally:
            # 精度比较完成后删除 cache 文件以节省空间
            # 文件已经在 _compare_with_downloaded 中加载到内存，可以安全删除
            try:
                if local_cache_path.exists():
                    local_cache_path.unlink(missing_ok=True)
                    print(
                        f"[download] Cleaned up cache file: {local_cache_path}",
                        flush=True,
                    )
            except Exception as e:
                print(
                    f"[download] Failed to delete cache file {local_cache_path}: {e}",
                    flush=True,
                )

        print(
            f"[download] Download mode completed for {self.api_config.config}",
            flush=True,
        )
