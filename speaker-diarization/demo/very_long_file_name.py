import math


def truncate_string(s, max_bytes):
    """
    根据最大字节数截断字符串，同时确保不会截断 UTF-8 编码的多字节字符。
    :param s: 待截断的 Python 字符串 (str)。
    :param max_bytes: 允许的最大字节长度 (int)。
    :return: 截断后的字符串 (str)。
    """
    if not s:
        return ""
    # 1. 尝试使用默认的 UTF-8 编码
    encoded_s = s.encode('utf-8')
    # 2. 如果字符串的字节长度未超限，直接返回原字符串
    if len(encoded_s) <= max_bytes:
        return s
    # 3. 截取最大允许的字节部分
    truncated_bytes = encoded_s[:max_bytes]
    # 4. 尝试将截取的字节解码回字符串
    #    'ignore' 错误处理机制：
    #    - 如果截取的字节末尾是一个不完整的 UTF-8 字符，
    #      `decode('utf-8', 'ignore')` 会自动忽略末尾不完整的字节。
    #      例如：b'hello\xe4\xbd' (不完整的'你') -> 'hello'
    #      这保证了返回的字符串是有效的、不包含乱码的。
    try:
        truncated_s = truncated_bytes.decode('utf-8', 'ignore')
        return truncated_s
    except UnicodeDecodeError:
        # 计算最大允许的字符数 (向上取整确保安全)
        # 255 字节 / 3 字节/字符 = 85 个字符
        max_chars = math.floor(max_bytes / 3)
        # 返回按字符数截断后的结果
        return s[:max_chars]


# 示例：一个包含中英文的字符串
example_string = "HelloPython!你好，世界。这是用来测试截断的超长字符串，超过255字节后后面的内容会被安全地忽略。"

# 设定最大字节长度
MAX_BYTES = 18
print(f"最大允许字节数: {MAX_BYTES}")

# 调用安全截断函数
result = safe_truncate_string(example_string, MAX_BYTES)

print("-" * 30)
print(f"原字符串 (字节数: {len(example_string.encode('utf-8'))}):\n{example_string}")
print("-" * 30)
print(f"截断后字符串 (字节数: {len(result.encode('utf-8'))}):\n{result}")
print(f"截断后字符串长度 (字符数): {len(result)}")
