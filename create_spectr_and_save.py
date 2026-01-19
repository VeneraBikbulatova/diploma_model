import numpy as np
import librosa
import os
import argparse
from multiprocessing import Pool
from functools import partial

DEFAULT_TARGET_SR = 22050
DEFAULT_N_MELS = 128
DEFAULT_SEGMENT_DURATION = 1.0
DEFAULT_FIXED_STEPS = 44

def augment_audio_segment(segment, sr=22050):
    augmented = []
    noise = 0.005 * np.random.randn(len(segment))
    augmented.append(segment + noise)
    augmented.append(librosa.effects.pitch_shift(segment, sr=sr, n_steps=2))
    augmented.append(librosa.effects.pitch_shift(segment, sr=sr, n_steps=-2))
    augmented.append(librosa.effects.time_stretch(segment, rate=0.8))
    augmented.append(librosa.effects.time_stretch(segment, rate=1.2))
    return augmented

def process_audio_file(file_path, label, output_dir, target_sr, n_mels, segment_duration, fixed_time_steps, augment=True):
    try:
        y, sr = librosa.load(file_path, sr=target_sr)
    except Exception as e:
        print(f"Ошибка загрузки {file_path}: {e}")
        return 0
    
    segment_samples = int(segment_duration * sr)
    all_spectrograms = []
    all_labels = []
    
    for start in range(0, len(y), segment_samples):
        end = start + segment_samples
        if end > len(y):
            break
        segment = y[start:end]
        
        S = librosa.feature.melspectrogram(y=segment, sr=sr, n_mels=n_mels)
        S_db = librosa.power_to_db(S, ref=np.max)
        
        # Жёстко задаём размер 128x128
        if S_db.shape[1] < fixed_time_steps:
            # Игнорируем сегменты короче 128 шагов
            continue
        else:
            # Обрезаем до 128 шагов
            S_db = S_db[:, :fixed_time_steps]
        
        # Проверяем, что размер действительно 128x128
        assert S_db.shape == (n_mels, fixed_time_steps), \
            f"Некорректный размер спектрограммы: {S_db.shape}, ожидается ({n_mels}, {fixed_time_steps})"
        
        all_spectrograms.append(S_db)
        all_labels.append(label)
        
        if augment and label == 1:
            for aug_segment in augment_audio_segment(segment, sr):
                S_aug = librosa.feature.melspectrogram(y=aug_segment, sr=sr, n_mels=n_mels)
                S_db_aug = librosa.power_to_db(S_aug, ref=np.max)
                
                # То же самое для аугментированных данных
                if S_db_aug.shape[1] < fixed_time_steps:
                    continue
                else:
                    S_db_aug = S_db_aug[:, :fixed_time_steps]
                
                assert S_db_aug.shape == (n_mels, fixed_time_steps)
                all_spectrograms.append(S_db_aug)
                all_labels.append(label)
    
    if all_spectrograms:
        base_name = os.path.splitext(os.path.basename(file_path))[0]
        np.save(os.path.join(output_dir, f"{base_name}_spectrograms.npy"), np.array(all_spectrograms))
        np.save(os.path.join(output_dir, f"{base_name}_labels.npy"), np.array(all_labels))
    
    return len(all_spectrograms)

def process_folder(folder_path, label, output_dir, target_sr, n_mels, segment_duration, fixed_time_steps, augment=True):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    audio_files = [os.path.join(folder_path, f) for f in os.listdir(folder_path) if f.endswith(('.wav', '.mp3', '.flac'))]
    print(f"Найдено {len(audio_files)} файлов в {folder_path}")
    
    with Pool() as pool:
        func = partial(
            process_audio_file,
            label=label,
            output_dir=output_dir,
            target_sr=target_sr,
            n_mels=n_mels,
            segment_duration=segment_duration,
            fixed_time_steps=fixed_time_steps,
            augment=augment
        )
        results = pool.map(func, audio_files)
    
    print(f"Обработано {sum(results)} сегментов (только 128x128).")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Преобразование аудио в мел-спектрограммы 128x128.')
    parser.add_argument('--input_dir', type=str, required=True, help='Папка с аудиофайлами')
    parser.add_argument('--output_dir', type=str, required=True, help='Папка для сохранения спектрограмм')
    parser.add_argument('--label', type=int, required=True, help='Метка класса (0 или 1)')
    parser.add_argument('--target_sr', type=int, default=DEFAULT_TARGET_SR, help='Частота дискретизации')
    parser.add_argument('--n_mels', type=int, default=DEFAULT_N_MELS, help='Количество полос Мела')
    parser.add_argument('--segment_duration', type=float, default=DEFAULT_SEGMENT_DURATION, help='Длительность сегмента (сек)')
    parser.add_argument('--no_augment', action='store_false', help='Отключить аугментацию')
    
    args = parser.parse_args()
    
    process_folder(
        folder_path=args.input_dir,
        label=args.label,
        output_dir=args.output_dir,
        target_sr=args.target_sr,
        n_mels=args.n_mels,
        segment_duration=args.segment_duration,
        fixed_time_steps=DEFAULT_FIXED_STEPS,
        augment=args.no_augment
    )