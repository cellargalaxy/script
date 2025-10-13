import os
import shutil


def main():
    base_path = input("请输入要处理的文件夹路径：").strip('"').strip("'")

    if not os.path.isdir(base_path):
        print(f"❌ 路径不存在或不是文件夹：{base_path}")
        input("按任意键退出...")
        return

    print(f"开始处理文件夹：{base_path}")
    moved_files = []

    # 递归遍历
    for root, dirs, files in os.walk(base_path, topdown=False):
        for filename in files:
            src_path = os.path.join(root, filename)

            # 计算相对路径，用子目录名拼接新文件名
            rel_path = os.path.relpath(src_path, base_path)
            parts = rel_path.split(os.sep)
            if len(parts) > 1:
                # 除最后一个以外都是目录名
                new_name = "_".join(parts[:-1] + [parts[-1]])
            else:
                new_name = parts[-1]

            dst_path = os.path.join(base_path, new_name)

            # 如果目标重名，添加编号避免覆盖
            if os.path.exists(dst_path):
                name, ext = os.path.splitext(new_name)
                count = 1
                while os.path.exists(dst_path):
                    dst_path = os.path.join(base_path, f"{name}_{count}{ext}")
                    count += 1

            # 移动文件
            shutil.move(src_path, dst_path)
            moved_files.append((src_path, dst_path))
            print(f"✅ 移动: {src_path} -> {dst_path}")

    # 删除空文件夹
    for root, dirs, files in os.walk(base_path, topdown=False):
        for d in dirs:
            dir_path = os.path.join(root, d)
            try:
                os.rmdir(dir_path)
                print(f"🗑️ 删除空文件夹: {dir_path}")
            except OSError:
                # 文件夹非空，跳过
                pass

    print("\n处理完成，共移动 {} 个文件。".format(len(moved_files)))
    input("按任意键退出...")


if __name__ == "__main__":
    main()
