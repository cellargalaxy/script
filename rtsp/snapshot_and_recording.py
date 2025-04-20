import traceback
import threading
import subprocess
import time
import logging

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s %(levelname)s %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S',
)


def task_actuator(task_func):
    while True:
        try:
            task_func()
        except Exception as e:
            logging.error("任务执行器，执行任务异常，任务名称：%s，异常：%s", threading.current_thread().name, e)
            traceback.print_exc()
        time.sleep(1)


def task_guardian(task_name, task_func):
    def func():
        task_actuator(task_func)

    while True:
        try:
            thread = threading.Thread(target=func, name=task_name)
            thread.start()
            thread.join()
            logging.info("任务守护者，任务执行器[%s]退出，重新启动", task_name)
        except BaseException as e:
            logging.error("任务守护者，捕获严重错误: %s", e)
            traceback.print_exc()
        time.sleep(1)


def thread_guardian(task_name, task_func):
    def func():
        task_guardian(task_name, task_func)

    thread = threading.Thread(target=func, daemon=True)
    thread.start()


def run_command(cmd):
    result = subprocess.run(cmd, shell=True, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    logging.info("执行命令，输出：", result)


def run_snapshot_and_recording():
    run_command("sh snapshot_and_recording.sh")


def run_clean():
    run_command("sh clean.sh")


thread_guardian("snapshot_and_recording", run_snapshot_and_recording)
thread_guardian("clean", run_clean)

while True:
    logging.info("运行中")
    time.sleep(10)
