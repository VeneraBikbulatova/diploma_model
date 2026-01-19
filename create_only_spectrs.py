import librosa
import numpy as np
import cv2
import os
from tqdm import tqdm

def create_yolo_labels(spectrogram, label, output_txt_path):
    """Создает YOLO-метки для спектрограммы"""
    height, width = spectrogram.shape
    
    # Пример: метка для всего изображения (адаптируйте под ваши нужды)
    class_id = 0  # ID класса
    x_center = 0.5  # Центр по X (нормализованный)
    y_center = 0.5  # Центр по Y (нормализованный)
    box_width = 1.0  # Ширина bbox (на всё изображение)
    box_height = 1.0  # Высота bbox (на всё изображение)
    
    with open(output_txt_path, 'w') as f:
        f.write(f"{class_id} {x_center} {y_center} {box_width} {box_height}")

def save_spectrogram_as_png(spectrogram, output_img_path):
    """Нормализует и сохраняет спектрограмму как PNG"""
    # Нормализация к 0-255
    img_normalized = 255 * (spectrogram - spectrogram.min()) / (spectrogram.max() - spectrogram.min())
    img_uint8 = img_normalized.astype(np.uint8)
    
    # Конвертация в BGR (для OpenCV)
    img_bgr = cv2.cvtColor(img_uint8, cv2.COLOR_GRAY2BGR)
    
    # Сохранение
    cv2.imwrite(output_img_path, img_bgr)

def process_audio_to_yolo(audio_path, output_dir, label, sr=22050, n_mels=128):
    """Основная функция обработки"""
    y, sr = librosa.load(audio_path, sr=sr)
    base_name = os.path.splitext(os.path.basename(audio_path))[0]
    segment_length = int(1.0 * sr)  # 1 секунда
    
    os.makedirs(os.path.join(output_dir, "images"), exist_ok=True)
    os.makedirs(os.path.join(output_dir, "labels"), exist_ok=True)
    
    for i in range(0, len(y), segment_length):
        segment = y[i:i+segment_length]
        if len(segment) < segment_length:
            continue
            
        # Генерация спектрограммы
        S = librosa.feature.melspectrogram(y=segment, sr=sr, n_mels=n_mels)
        S_db = librosa.power_to_db(S, ref=np.max)
        
        # Имена файлов
        img_name = f"{base_name}_seg{i}.png"
        label_name = f"{base_name}_seg{i}.txt"
        
        img_path = os.path.join(output_dir, "images", img_name)
        label_path = os.path.join(output_dir, "labels", label_name)
        
        # Сохранение
        save_spectrogram_as_png(S_db, img_path)
        create_yolo_labels(S_db, label, label_path)

# Пример использования
input_audio_dir = "lmd_full/b/segments"
output_yolo_dir = "lmd_full/b/spectrs"
label = 1  # Ваш класс

audio_files = [os.path.join(input_audio_dir, f) for f in os.listdir(input_audio_dir) if f.endswith(".wav")]

for audio_file in tqdm(audio_files, desc="Processing audio files"):
    process_audio_to_yolo(audio_file, output_yolo_dir, label)