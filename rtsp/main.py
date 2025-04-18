import traceback
import threading
import subprocess
import time


def task_actuator(task_func):
    while True:
        try:
            task_func()
        except Exception as e:
            print(f"任务执行器，执行任务异常，任务名称：{threading.current_thread().name}，异常：{e}")
            traceback.print_exc()
        time.sleep(1)


def task_guardian(task_name, task_func):
    def actuator():
        task_actuator(task_func)

    while True:
        try:
            thread = threading.Thread(target=actuator, name=task_name)
            thread.start()
            thread.join()
            print(f"任务守护者，任务执行器{task_name}退出，重新启动")
        except BaseException as e:
            print(f"任务守护者，捕获严重错误: {e}")
            traceback.print_exc()
        time.sleep(1)


def thread_guardian(task_name, task_func):
    def actuator():
        task_guardian(task_name, task_func)

    thread = threading.Thread(target=actuator, daemon=True)
    thread.start()


def run_command(cmd):
    result = subprocess.run(cmd, shell=True, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    print(f"执行命令，输出: {result}")


while True:
    print(f"执行main")
    time.sleep(100000)
