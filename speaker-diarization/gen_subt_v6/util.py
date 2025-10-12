from pathlib import Path
import subprocess
import logging
import shlex
import os
import platform
import torch
import GPUtil
import gc
import shutil
from inputimeout import inputimeout
import json


def get_logger(name='main', fmt='%(asctime)s %(levelname)-5s %(filename)s:%(lineno)d - %(message)s'):
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


def mkdir(path):
    obj = Path(path)
    _, ext = os.path.splitext(path)
    if ext:
        obj.parent.mkdir(parents=True, exist_ok=True)
    else:
        obj.mkdir(parents=True, exist_ok=True)


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
            logger.debug("%s", line)
    return_code = process.wait()
    logger.info("执行命令，退出：%s", return_code)
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


def get_device_type():
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    return device


def print_device_info():
    path_env = os.environ.get('PATH')
    logger.info(f"PATH: {path_env}")
    device_type = get_device_type()
    logger.info(f"device_type: {device_type}")
    device_info = get_device_info()
    for key, value in device_info.items():
        logger.info(f"{key}: {value}")


def get_compute_type():
    device = get_device_type()
    if device == 'cuda':
        return 'float16'
    else:
        return 'int8'


def get_file_ext(file_path):
    """
    '/aaa/bbb/ccc' -> ''
    '/aaa/bbb/ccc.txt' -> 'txt'
    :param file_path:
    :return:
    """
    ext = Path(file_path).suffix[1:]
    return ext


def get_file_name(file_path):
    """
    '/aaa/bbb/ccc' -> 'ccc'
    '/aaa/bbb/ccc.txt' -> 'ccc'
    :param file_path:
    :return:
    """
    name = os.path.splitext(os.path.basename(file_path))[0]
    return name


def get_file_basename(file_path):
    """
    '/aaa/bbb/ccc' -> 'ccc'
    '/aaa/bbb/ccc.txt' -> 'ccc.txt'
    :param file_path:
    :return:
    """
    name = os.path.basename(file_path)
    return name


def get_parent_dir(file_path):
    """
    '/aaa/bbb/ccc' -> 'bbb'
    '/aaa/bbb/ccc.txt' -> 'bbb'
    :param file_path:
    :return:
    """
    parent = os.path.dirname(file_path)
    parent_dir = os.path.basename(parent)
    return parent_dir


def get_ancestor_dir(file_path):
    """
    '/aaa/bbb/ccc' -> '/aaa/bbb'
    '/aaa/bbb/ccc.txt' -> '/aaa/bbb'
    :param file_path:
    :return:
    """
    file_dir = os.path.dirname(file_path)
    return file_dir


def json_dumps(obj):
    return json.dumps(obj, ensure_ascii=False)


def json_loads(content):
    return json.loads(content)


def save_as_json(obj, save_path):
    save_file(json.dumps(obj, ensure_ascii=False, indent=2), save_path)


def save_file(content, file_path):
    mkdir(file_path)
    with open(file_path, 'w', encoding='utf-8') as file:
        file.write(content)


def read_file(file_path, default_value=''):
    if not path_isfile(file_path):
        return default_value
    with open(file_path, 'r') as file:
        return file.read()


def read_file_to_obj(file_path, default_value=''):
    content = read_file(file_path, default_value=default_value)
    obj = json_loads(content)
    return obj


def path_exist(path):
    return os.path.exists(path)


def path_isfile(path):
    if not path_exist(path):
        return False
    return os.path.isfile(path)


def exec_gc():
    torch.cuda.empty_cache()
    torch.cuda.ipc_collect()
    gc.collect()


def copy_file(from_path, to_path):
    mkdir(to_path)
    shutil.copy(from_path, to_path)


def delete_path(path):
    path = Path(path)
    if path.is_file():
        path.unlink()
    elif path.is_dir():
        shutil.rmtree(path)


def listdir(path):
    file_names = sorted(os.listdir(path))
    return file_names


def get_home_dir():
    home_path = Path.home()
    return home_path


def get_script_path():
    script_dir = os.path.dirname(os.path.abspath(__file__))
    return script_dir


def in_notebook() -> bool:
    try:
        import google.colab
        return True
    except ImportError:
        pass
    try:
        from IPython import get_ipython
        shell = get_ipython().__class__.__name__
        return shell == 'ZMQInteractiveShell'
    except (NameError, ImportError):
        pass
    return False


def input_timeout(prompt, timeout, default=None):
    if in_notebook():
        return default
    try:
        text = inputimeout(prompt=prompt, timeout=timeout)
    except Exception:
        return default
    else:
        return text
