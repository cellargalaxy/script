from pathlib import Path
import subprocess
import logging
import os
import platform
import shutil
import json
import sys
import copy
import gc
import math
from collections import Counter
import re


def get_logger(name='main', fmt='%(asctime)s %(levelname)-5s %(filename)s:%(lineno)d - %(message)s'):
    logger = logging.getLogger(name)
    if logger.handlers:
        return logger
    logger.setLevel(logging.INFO)
    formatter = logging.Formatter(fmt=fmt, datefmt='%Y-%m-%d %H:%M:%S')

    console_handler = logging.StreamHandler()
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)

    log_file = f'log/{name}.log'
    log_path = Path(log_file)
    log_path.parent.mkdir(parents=True, exist_ok=True)
    file_handler = logging.FileHandler(log_file, mode='a', encoding='utf-8')
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)

    return logger


def flush_logger():
    for logger_name in logging.Logger.manager.loggerDict:
        logger = logging.getLogger(logger_name)
        for handler in logger.handlers:
            try:
                handler.flush()
            except Exception:
                pass


logger = get_logger()


def mkdir(path):
    obj = Path(path)
    _, ext = os.path.splitext(path)
    if ext:
        obj.parent.mkdir(parents=True, exist_ok=True)
    else:
        obj.mkdir(parents=True, exist_ok=True)


def popen_cmd(cmd):
    process = subprocess.Popen(
        cmd,
        shell=False,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
        cwd=os.getcwd(),
    )
    with process.stdout:
        for line in iter(process.stdout.readline, ''):
            logger.debug("%s", line)
    return_code = process.wait()
    logger.info("执行命令，退出：%s", return_code)
    return return_code


def run_cmd(cmd):
    result = subprocess.run(
        cmd,
        shell=False,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        bufsize=1,
        cwd=os.getcwd(),
    )
    return result.stdout, result.returncode


def get_sys_info():
    info = {}
    info['Python版本'] = sys.version
    info['Python路径'] = sys.executable
    info['操作系统'] = f"{platform.system()} {platform.release()} ({platform.version()})"
    info['环境变量'] = os.environ.get('PATH')
    info['CPU'] = f"{platform.processor() or platform.uname().processor} ({platform.machine()})"

    try:
        import torch
        info['PyTorch版本'] = torch.__version__
        info['CUDA是否可用'] = torch.cuda.is_available()
        info['CUDA版本'] = torch.version.cuda
        for i in range(torch.cuda.device_count()):
            props = torch.cuda.get_device_properties(i)
            total_memory_gb = round(props.total_memory / (1024 ** 3), 2)
            info['GPU'] = f"{props.name} 总显存 {total_memory_gb} GB"
            break
    except ImportError as e:
        logger.error("未安装依赖torch", exc_info=True)
        info['PyTorch版本'] = "torch未安装"
        info['CUDA是否可用'] = False
        info['CUDA版本'] = 'N/A'
        info['GPU'] = 'N/A'

    return info


def print_sys_info():
    sys_info = get_sys_info()
    for key, value in sys_info.items():
        logger.info(f"{key}: {value}")


def get_device_type():
    device = 'cpu'
    try:
        import torch
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
    except ImportError as e:
        logger.error("未安装依赖torch", exc_info=True)
    return device


def get_compute_type():
    device = get_device_type()
    if device == 'cuda':
        return 'float16'
    else:
        return 'int8'


def exec_gc():
    try:
        import torch
        torch.cuda.empty_cache()
        torch.cuda.ipc_collect()
    except ImportError as e:
        logger.error("未安装依赖torch", exc_info=True)
    gc.collect()


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
    return json.dumps(obj, ensure_ascii=False, sort_keys=True)


def json_loads(content):
    return json.loads(content)


def save_file(content, file_path):
    mkdir(file_path)
    with open(file_path, 'w', encoding='utf-8') as file:
        file.write(content)


def save_as_json(obj, save_path):
    save_file(json.dumps(obj, ensure_ascii=False, sort_keys=True, indent=2), save_path)


def path_exist(path):
    return os.path.exists(path)


def path_isfile(path):
    if not path_exist(path):
        return False
    return os.path.isfile(path)


def read_file(file_path, default_value=''):
    if not path_isfile(file_path):
        return default_value
    with open(file_path, 'r') as file:
        return file.read()


def read_file_to_obj(file_path, default_value=''):
    content = read_file(file_path, default_value=default_value)
    obj = json_loads(content)
    return obj


def move_file(from_path, to_path):
    mkdir(to_path)
    shutil.move(from_path, to_path)


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


def safe_file_name(name: str) -> str:
    # Windows 非法字符
    return re.sub(r'[\\/:*?"<>|]', '_', name)


def truncate_path(path, max_length=255):
    if not path:
        return path
    ancestor_dir = get_ancestor_dir(path)
    file_name = get_file_name(path)
    file_name = safe_file_name(file_name)
    file_ext = get_file_ext(path)
    file_name = truncate_string(file_name, max_length - len(file_ext.encode('utf-8')) - 4)  # 减4留点余地
    file_basename = file_name
    if file_ext:
        file_basename = f"{file_name}.{file_ext}"
    path = os.path.join(ancestor_dir, file_basename)
    return path


def truncate_string(text, max_length):
    if not text:
        return text
    data = text.encode('utf-8')
    if len(data) <= max_length:
        return text
    data = data[:max_length]
    try:
        text = data.decode('utf-8', 'ignore')  # 使用ignore无视被中间截断的字
        return text
    except UnicodeDecodeError:
        max_char = math.floor(max_length / 3)
        return text[:max_char]


def input_timeout(prompt, timeout, default=None):
    try:
        from inputimeout import inputimeout
        text = inputimeout(prompt=prompt, timeout=timeout)
    except ImportError:
        logger.error("未安装依赖inputimeout", exc_info=True)
        return default
    except Exception as e:
        return default
    else:
        return text


def deepcopy_obj(obj):
    return copy.deepcopy(obj)


def get_list_most(array):
    if not array:
        return None
    counts = Counter(array)
    max_count = max(counts.values())
    most = [element for element, count in counts.items() if count == max_count]
    most.sort()
    return most[0]
