$pythonScript = @"
"""
写一个python脚本，只用标准库，无外部依赖
1. input输入一个文件夹路径，例如d:/demo
1.1 文件夹下可能会有多个以「data_」开头的子文件夹，例如data_raw/data_train/data_test等等
2. 在文件夹下，输出一个csv文件，文件名称与文件夹相同，例如d:/demo/demo.csv，如果该csv文件已存在，则覆盖原数据
3. csv文件有两列
3.1 第一列表头是name，存储的是文件名称
3.2 第二列表头是sort，存储的是文件所在的子文件夹名称，同一个文件可能会在多个子文件夹下，多个子文件夹名称使用英文分号分隔
4. 遍历demo下面的全部子文件夹，获得文件名称与子文件夹名称，填写到csv中
4.1 同一个文件名称只有一行数据，第二列使用英文分号分隔记录多个子文件夹名称
4.2 csv文件以第一列升序排序
4.3 对第二列的子文件夹名称进行去除并排序
4.4 保存csv文件
4.5 遍历过程中打印执行情况
5. 最后点击任意键退出脚本
"""

import os
import csv


def main():
    folder = input("请输入文件夹路径: ").strip()
    if not os.path.isdir(folder):
        print(f"路径不存在: {folder}")
        os.system("pause")
        return

    folder_name = os.path.basename(os.path.normpath(folder))
    csv_path = os.path.join(folder, f"{folder_name}.csv")

    print(f"正在遍历文件夹: {folder}")
    file_to_folders = {}
    # 只处理以"data_"开头的子文件夹
    subfolders = [
        f for f in os.listdir(folder)
        if os.path.isdir(os.path.join(folder, f)) and f.startswith("data_")
    ]
    print(f"发现以'data_'开头的子文件夹: {subfolders}")

    for subfolder in subfolders:
        subfolder_path = os.path.join(folder, subfolder)
        for root, _, files in os.walk(subfolder_path):
            for file in files:
                if file not in file_to_folders:
                    file_to_folders[file] = set()
                file_to_folders[file].add(subfolder)
                print(f"文件: {file} 出现在子文件夹: {subfolder}")

    # 处理和排序
    rows = []
    for file, folders in file_to_folders.items():
        sorted_folders = sorted(set(folders))
        rows.append([file, ";".join(sorted_folders)])

    # 按第一列（文件名）升序排序
    rows.sort(key=lambda x: x[0])

    print(f"正在写入CSV文件: {csv_path}")
    with open(csv_path, "w", newline='', encoding="utf-8") as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(["name", "sort"])
        writer.writerows(rows)

    print(f"CSV文件已生成: {csv_path}")
    input("执行完毕，按任意键退出...")


if __name__ == "__main__":
    main()

"@
$scriptPath = "$env:TEMP\temp_script.py"
Set-Content -Path $scriptPath -Value $pythonScript
python $scriptPath
Remove-Item $scriptPath  # 清理临时文件