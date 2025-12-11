import os
import glob
from tqdm import tqdm
from PIL import Image, UnidentifiedImageError


def check_png_files(folder_path):
    # 初始化统计
    total = 0
    valid = 0
    corrupted = 0
    fixed = 0
    corrupted_details = []

    # 获取所有PNG文件
    png_files = glob.glob(os.path.join(folder_path, "*.png"))
    total = len(png_files)
    print(f"开始检查PNG文件，共发现 {total} 个PNG文件...")

    # 逐个检查
    for file_path in tqdm(png_files, desc="Checking PNG files"):
        try:
            # 1. 基础校验：文件存在且非空
            if not os.path.exists(file_path):
                corrupted_details.append((file_path, f"文件不存在"))
                corrupted += 1
                continue
            if os.path.getsize(file_path) == 0:
                corrupted_details.append((file_path, f"空文件（大小为0）"))
                corrupted += 1
                continue

            # 2. 尝试打开并验证图片
            with Image.open(file_path) as img:
                # 关键：校验图片是否真的是PNG格式
                if img.format != "PNG":
                    corrupted_details.append((file_path, f"格式错误（实际为{img.format}）"))
                    corrupted += 1
                    continue
                # 验证图片完整性（读取像素，避免只校验头）
                img.load()
                valid += 1

        except UnidentifiedImageError:
            corrupted_details.append((file_path, "无法识别的图片格式（非PNG/损坏）"))
            corrupted += 1
        except PermissionError:
            corrupted_details.append((file_path, "权限不足，无法读取文件"))
            corrupted += 1
        except Exception as e:
            # 捕获所有异常，避免脚本中断
            corrupted_details.append((file_path, f"General error: {str(e)}"))
            corrupted += 1

    # 生成报告
    print("\n===== PNG文件检查报告 =====")
    print(f"总检查文件数: {total}")
    print(f"有效文件数: {valid}")
    print(f"损坏文件数: {corrupted}")
    print(f"修复成功文件数: {fixed}")

    # 打印损坏文件详情（只显示前50个，避免日志过长）
    if corrupted_details:
        print("\n===== 损坏文件详情（前50个） =====")
        for idx, (path, err) in enumerate(corrupted_details[:50], 1):
            print(f"{idx}. 路径: {path}")
            print(f"   错误: {err}\n")


if __name__ == "__main__":
    # 替换为你的数据集路径
    target_folder = r"D:\pythonProject\Watermarking\DIV2K\valid"
    check_png_files(target_folder)
