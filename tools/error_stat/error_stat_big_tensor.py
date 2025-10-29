# test_log 一键整理小工具：engineV2 big tensor 修改版
# @author: cangtianhuang
# @date: 2025-06-08
# 整理效果：pass + error + invalid

import argparse
import re
from pathlib import Path

# error logs
ERROR_FILES = [
    "api_config_accuracy_error.txt",
    "api_config_crash.txt",
    "api_config_paddle_error.txt",
    "api_config_torch_error.txt",
    "api_config_paddle_to_torch_failed.txt",
    "api_config_timeout.txt",
    "api_config_skip.txt",
]

# skip error info
SKIP_ERROR_INFO = [
    "CUDA out of memory",
    "Out of memory error",
    "[torch error]",
    "[Skip]",
    "[numpy error]",
    "[paddle_to_torch]",
    "(Cannot allocate memory)",
    "(InvalidArgument)",
    "(NotFound)",
    "(ResourceExhausted)",
    "(Unimplemented)",
    "Invalid TensorConfig",
    "Unable to allocate",
    "Cannot take a larger sample",
    "output type diff error",
    "should be",
    "must equal",
    "received:",
    "incorrect shape",
    "variance is of incorrect shape",
    "The data type of input Variable",
    "should satisfy",
]


def error_state(input_path, output_path):
    input_path = Path(input_path)
    output_path = Path(output_path)
    if not input_path.exists():
        print(f"{input_path} not exists", flush=True)
        return
    output_path.mkdir(parents=True, exist_ok=True)

    # get all test blocks
    logs = []
    in_test_block = False
    current_content = []

    log_path = input_path / "log_inorder.log"
    try:
        with log_path.open("r") as f:
            input_text = f.read()
    except Exception as err:
        print(f"Error reading {log_path}: {err}", flush=True)
        return

    for line in input_text.split("\n"):
        if "gpu_resources.cc" in line or "Waiting for available memory" in line:
            continue

        if "test begin" in line:
            if in_test_block and current_content:
                logs.append("\n".join(current_content))
            in_test_block = True
            current_content = [line]
            continue

        if "Worker PID" in line:
            if in_test_block and current_content:
                logs.append("\n".join(current_content))
            in_test_block = False
            current_content = []
            continue

        if in_test_block:
            current_content.append(line)

    if current_content:
        logs.append("\n".join(current_content))
    print(f"Found {len(logs)} logs", flush=True)

    def get_sort_key(content):
        lines = content.split("\n")
        match = re.search(r"test begin: (.*)$", lines[0])
        if match:
            return match.group(1).strip()
        return ""

    # get all pass api and config
    pass_file = input_path / "api_config_pass.txt"
    pass_apis = set()
    pass_configs = set()
    if pass_file.exists():
        try:
            with open(pass_file, "r") as f:
                for line in f:
                    line = line.strip()
                    if line:
                        pass_api = line.split("(", 1)[0]
                        pass_apis.add(pass_api)
                        pass_configs.add(line)
        except Exception as err:
            print(f"Error reading {pass_file}: {err}", flush=True)
            return
    print(f"Read {len(pass_apis)} pass apis", flush=True)
    print(f"Read {len(pass_configs)} pass api configs", flush=True)

    # classify logs
    pass_logs = {}
    error_logs = {}
    invalid_logs = {}
    for content in logs:
        key = get_sort_key(content)
        if not key:
            continue
        if any(info in content for info in SKIP_ERROR_INFO):
            invalid_logs[key] = content
        else:
            lines = content.split("\n")
            if len(lines) == 1 or len(lines) == 2 and not lines[1].startswith("["):
                invalid_logs[key] = content
            elif key in pass_configs:
                pass_logs[key] = content
            else:
                error_logs[key] = content
    print(f"Read {len(pass_logs)} pass logs", flush=True)
    print(f"Read {len(error_logs)} error logs", flush=True)
    if invalid_logs:
        print(f"Read {len(invalid_logs)} invalid logs", flush=True)

    # write pass_log.log
    pass_log = output_path / "pass_log.log"
    try:
        with open(pass_log, "w") as f:
            for key in sorted(pass_logs.keys()):
                content = pass_logs[key]
                f.write(content + "\n\n")
    except Exception as err:
        print(f"Error writing {pass_log}: {err}", flush=True)
        return
    print(f"Write {len(pass_logs)} pass logs", flush=True)

    # write pass_api.txt
    API_output_path = output_path / "pass_api.txt"
    try:
        with open(API_output_path, "w") as f:
            f.writelines(f"{line}\n" for line in sorted(pass_apis))
    except Exception as err:
        print(f"Error writing {API_output_path}: {err}", flush=True)
        return
    print(f"Write {len(pass_apis)} pass apis", flush=True)

    # write pass_config.txt
    CONFIG_output_path = output_path / "pass_config.txt"
    try:
        with open(CONFIG_output_path, "w") as f:
            f.writelines(f"{line}\n" for line in sorted(pass_configs))
    except Exception as err:
        print(f"Error writing {CONFIG_output_path}: {err}", flush=True)
        return
    print(f"Write {len(pass_configs)} pass api configs", flush=True)

    # get all error api and config
    error_apis = set()
    error_configs = set()
    for file_name in ERROR_FILES:
        FILE_PATH = input_path / file_name
        if not FILE_PATH.exists():
            continue
        try:
            with open(FILE_PATH, "r") as f:
                for line in f:
                    line = line.strip()
                    if line:
                        if line not in error_logs:
                            if line not in invalid_logs:
                                invalid_logs[line] = ""
                            continue
                        error_name = line.split("(", 1)[0]
                        error_apis.add(error_name)
                        error_configs.add(line)
        except Exception as err:
            print(f"Error reading {file_name}: {err}", flush=True)
            return
    print(f"Read {len(error_apis)} error apis", flush=True)
    print(f"Read {len(error_configs)} error api configs", flush=True)

    # write error_api.txt
    API_output_path = output_path / "error_api.txt"
    try:
        with open(API_output_path, "w") as f:
            f.writelines(f"{line}\n" for line in sorted(error_apis))
    except Exception as err:
        print(f"Error writing {API_output_path}: {err}", flush=True)
        return
    print(f"Write {len(error_apis)} error apis", flush=True)

    # write error_config.txt
    CONFIG_output_path = output_path / "error_config.txt"
    try:
        with open(CONFIG_output_path, "w") as f:
            f.writelines(f"{line}\n" for line in sorted(error_configs))
    except Exception as err:
        print(f"Error writing {CONFIG_output_path}: {err}", flush=True)
        return
    print(f"Write {len(error_configs)} error api configs", flush=True)

    # write error_log.log
    error_log = output_path / "error_log.log"
    count = 0
    try:
        with open(error_log, "w") as f:
            for key in sorted(error_logs.keys()):
                if key not in error_configs:
                    continue
                content = error_logs[key]
                f.write(content + "\n\n")
                count += 1
    except Exception as err:
        print(f"Error writing {error_log}: {err}", flush=True)
        return
    print(f"Write {count} error logs", flush=True)

    # write invalid_config.txt
    if invalid_logs:
        CONFIG_output_path = output_path / "invalid_config.txt"
        try:
            with open(CONFIG_output_path, "w") as f:
                f.writelines(f"{line}\n" for line in sorted(invalid_logs.keys()))
        except Exception as err:
            print(f"Error writing {CONFIG_output_path}: {err}", flush=True)
            return
        print(f"Write {len(invalid_logs)} invalid api configs", flush=True)


def main():
    default_input_log_path = "tester/api_config/test_log_big_tensor"
    default_output_log_path = default_input_log_path

    parser = argparse.ArgumentParser(
        description="test_log 一键整理小工具: engineV2 big tensor",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
使用示例:
  python %(prog)s --input xxx --output xxx # 指定输入输出路径
  python %(prog)s -i xxx # 仅指定输入路径，默认输出到输入目录
        """,
    )
    parser.add_argument(
        "--input", "-i", type=str, default=default_input_log_path, help="输入路径"
    )
    parser.add_argument(
        "--output", "-o", type=str, default=default_output_log_path, help="输出路径"
    )
    args = parser.parse_args()
    if args.output is None:
        args.output = args.input
    error_state(args.input, args.output)


if __name__ == "__main__":
    main()
