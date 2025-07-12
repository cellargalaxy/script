import os

bin_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "bin")
os.environ["PATH"] = bin_dir + os.pathsep + os.environ.get("PATH", "")

os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'
os.environ['http_proxy'] = 'http://192.168.123.5:10808'
os.environ['https_proxy'] = 'http://192.168.123.5:10808'
os.environ['no_proxy'] = 'localhost,127.0.0.1,::1,192.168.123.5,mirrors.ustc.edu.cn'

from spleeter.separator import Separator

# 初始化 2 stems 模型
separator = Separator('spleeter:4stems')

# 分离音频（会自动输出 vocals 和 accompaniment）
separator.separate_to_file('../demo_jpn.wav', 'spleeter_demo/')
