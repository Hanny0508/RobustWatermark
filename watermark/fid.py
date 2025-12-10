from pytorch_fid.fid_score import calculate_fid_given_paths  # 导入计算 FID 的函数

# 设置图像文件夹的路径
image_path1 = '/home/nas928/huangshuxin/lab/DIV2K/ours/container'
image_path2 = '/home/nas928/huangshuxin/lab/DIV2K/ours/vae5/attack'

# 调用函数计算 FID
fid_value = calculate_fid_given_paths([image_path1, image_path2], batch_size=5, device='cuda', dims=2048)

print(f'The FID score between the two image folders is: {fid_value}')
