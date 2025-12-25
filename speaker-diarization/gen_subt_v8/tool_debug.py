import matplotlib.pyplot as plt

def plot_with_time(data, interval_ms):
    """
    根据浮点数组和时间间隔画图

    :param data: list 或 numpy array，浮点数组
    :param interval_ms: int 或 float，每个数据点对应的时间间隔（毫秒）
    """
    # 生成时间轴（单位：分钟）
    time_minutes = [
        i * interval_ms / 1000 / 60
        for i in range(len(data))
    ]

    # 画图
    plt.figure(figsize=(10, 4))
    plt.plot(time_minutes, data)
    plt.xlabel("Time (minutes)")
    plt.ylabel("Value")
    plt.title("Value vs Time")
    plt.grid(True)
    plt.tight_layout()
    plt.show()
