"""
写一个python脚本，只用标准库，无外部依赖
1. 脚本启动后，手动在终端输入两个参数
1.1 文件夹路径，必填
1.2 语言类型，必填，可选范围有JA/ZH/EN
2. 根据文件夹路径，获得文件夹名称
3. 遍历文件夹下的文件，获得文件路径与文件名称
4. 以「{文件路径}|{文件夹名称}|{语言类型}|{文件名称}」格式，每个文件一行，输出到一个文本文件中
5. 输出的文本文件与文件夹同级别，文件名称为「{文件夹名称}.list」
6. 最后点击任意键退出脚本

例如文件夹路径是d:/demo
文件夹下有两个文件，分别是aaa.mp3与bbb.wav
输出的文本文件是d:/demo.list，内容为
```
d:/demo/aaa.mp3|demo|JA|aaa
d:/demo/bbb.wav|demo|JA|bbb
```
"""

import os


def get_user_input():
    folder_path = input("请输入文件夹路径: ").strip()
    while not folder_path or not os.path.isdir(folder_path):
        print("文件夹路径无效，请重新输入。")
        folder_path = input("请输入文件夹路径: ").strip()

    lang_type = input("请输入语言类型 (JA/ZH/EN): ").strip().upper()
    while lang_type not in ('JA', 'ZH', 'EN'):
        print("语言类型无效，请重新输入 (JA/ZH/EN)。")
        lang_type = input("请输入语言类型 (JA/ZH/EN): ").strip().upper()
    return folder_path, lang_type


def main():
    folder_path, lang_type = get_user_input()
    folder_name = os.path.basename(os.path.normpath(folder_path))
    output_path = os.path.join(os.path.dirname(folder_path), f"{folder_name}.list")

    lines = []
    for entry in os.listdir(folder_path):
        full_path = os.path.join(folder_path, entry)
        if os.path.isfile(full_path):
            file_name = os.path.splitext(entry)[0]
            lines.append(f"{full_path}|{folder_name}|{lang_type}|{file_name}")

    with open(output_path, 'w', encoding='utf-8') as f:
        for line in lines:
            f.write(line + "\n")

    print(f"已生成列表文件: {output_path}")
    input("按任意键退出...")


if __name__ == "__main__":
    main()
