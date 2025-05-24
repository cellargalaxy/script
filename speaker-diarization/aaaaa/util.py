from pathlib import Path
import subprocess
import logging
import shlex
import os
import platform
import torch
import GPUtil


def get_logger(name='main', fmt='%(asctime)s %(levelname)-8s %(message)s'):
    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)
    if not logger.handlers:
        handler = logging.StreamHandler()
        formatter = logging.Formatter(
            fmt=fmt,
            datefmt='%Y-%m-%d %H:%M:%S'
        )
        handler.setFormatter(formatter)
        logger.addHandler(handler)
    return logger


logger = get_logger()


def mkdir(file_path):
    file_path = Path(file_path)
    file_path.parent.mkdir(parents=True, exist_ok=True)


cmd_logger = logging.getLogger("simple_logger")
handler_exec = logging.StreamHandler()
formatter_exec = logging.Formatter('%(message)s')
handler_exec.setFormatter(formatter_exec)
cmd_logger.addHandler(handler_exec)
cmd_logger.setLevel(logging.INFO)
cmd_logger.propagate = False


def popen_cmd(cmd):
    exec_cmd = "cd {} && {}".format(os.getcwd(), shlex.join(cmd))
    logger.info("执行命令: %s", exec_cmd)
    process = subprocess.Popen(
        exec_cmd,
        shell=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1
    )
    with process.stdout:
        for line in iter(process.stdout.readline, ''):
            cmd_logger.info("%s", line)
    return_code = process.wait()
    logging.info("执行命令，退出：%s", return_code)
    return return_code


def run_cmd(cmd):
    exec_cmd = "cd {} && {}".format(os.getcwd(), shlex.join(cmd))
    logger.info("执行命令: %s", exec_cmd)
    result = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    return result.stdout, result.returncode


def get_device_info():
    info = {}
    info['CPU'] = platform.processor() or platform.uname().processor
    gpus = GPUtil.getGPUs()
    if gpus:
        gpu = gpus[0]
        info['GPU'] = gpu.name
        info['CUDA Available'] = torch.cuda.is_available()
        info['CUDA Version'] = torch.version.cuda
        info['GPU Memory (MB)'] = f"{gpu.memoryTotal} MB"
    else:
        info['GPU'] = 'No GPU detected'
        info['CUDA Available'] = False
        info['CUDA Version'] = None
        info['GPU Memory (MB)'] = 'N/A'
    return info


def print_device_info():
    device_info = get_device_info()
    for key, value in device_info.items():
        logger.info(f"{key}: {value}")


def get_device_type():
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    return device


def set_device_type_to_manager(manager):
    device = get_device_type()
    manager['device'] = device


def get_file_ext(file_path):
    ext = Path(file_path).suffix[1:]
    return ext
