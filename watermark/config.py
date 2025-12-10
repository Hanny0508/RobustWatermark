# Super parameters
clamp = 2.0
channels_in = 3
log10_lr = -4.5
lr = 10 ** log10_lr
weight_decay = 1e-5
init_scale = 0.01

# Model:
device_ids = [0]

# Train:
batch_size = 2
cropsize = 256
betas = (0.5, 0.999)
weight_step = 200
gamma = 0.5

# Val:
cropsize_val = 256
batchsize_val = 2
shuffle_val = False
val_freq = 10

# GAN
gan_lr = 1e-4
gan_weight = 0.1

# Dataset
TRAIN_PATH = r'/Users/hanny/Downloads/DIV2k/train'
VAL_PATH = r'/Users/hanny/Downloads/DIV2k/valid'
format_train = 'png'
format_val = 'png'

# Display and logging:
loss_display_cutoff = 2.0
loss_names = ['train loss', 'val loss', 'lr', 'attack method']
silent = False
live_visualization = False
progress_bar = True

# Saving checkpoints:

MODEL_PATH = r'/Users/hanny/PycharmProjects/pythonProject/2024-2025_2nd/Self/TheEncryptionAndDecryptionOfImageWatermarks/Code/watermark/model'
checkpoint_on_error = True
SAVE_freq = 50

IMAGE_PATH = r'/Users/hanny/PycharmProjects/pythonProject/2024-2025_2nd/Self/TheEncryptionAndDecryptionOfImageWatermarks/Code/watermark/image'
IMAGE_PATH_host = r'/Users/hanny/PycharmProjects/pythonProject/2024-2025_2nd/Self/TheEncryptionAndDecryptionOfImageWatermarks/Code/watermark/image/host'
IMAGE_PATH_secret = r'/Users/hanny/PycharmProjects/pythonProject/2024-2025_2nd/Self/TheEncryptionAndDecryptionOfImageWatermarks/Code/watermark/image/secret'
IMAGE_PATH_container = r'/Users/hanny/PycharmProjects/pythonProject/2024-2025_2nd/Self/TheEncryptionAndDecryptionOfImageWatermarks/Code/watermark/image/container'
IMAGE_PATH_extracted = r'/Users/hanny/PycharmProjects/pythonProject/2024-2025_2nd/Self/TheEncryptionAndDecryptionOfImageWatermarks/Code/watermark/image/extracted'
