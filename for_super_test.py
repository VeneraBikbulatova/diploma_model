import librosa
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.models import load_model
import soundfile as sf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input
from tensorflow.keras.layers import InputLayer
from tensorflow.keras.models import load_model

def debug_predict(file_path, model, segment_duration=3.0, target_time_steps=44):
    # Загрузка аудио
    y, sr = librosa.load(file_path, sr=22050, mono=True)
    
    y_trimmed = librosa.effects.split(y, top_db=25, frame_length=2048, hop_length=512)  # Делим на ненулевые участки
    y_trimmed = np.concatenate([y[start:end] for (start, end) in y_trimmed])
    
    # 2. Медианный фильтр для подавления шумов
    y_clean = librosa.effects.preemphasis(y_trimmed, coef=0.97)
    
    # 3. Спектральное шумоподавление
    S = librosa.stft(y_clean)
    S_mag = np.abs(S)
    mask = librosa.util.softmask(S_mag, 0.2 * S_mag, power=2)
    S_clean = S * mask
    y_processed = librosa.istft(S_clean)
    
    plt.figure(figsize=(15, 5))
    plt.subplot(2, 1, 1)
    plt.plot(y)
    plt.title("Original waveform")
    plt.subplot(2, 1, 2)
    plt.plot(y_processed)
    plt.title("Processed waveform")
    plt.show()
    
    print(f"Длина после загрузки: {len(y)/sr:.2f} сек")
    print(f"Длина после обрезки: {len(y_trimmed)/sr:.2f} сек")
    print(f"Длина после обработки: {len(y_processed)/sr:.2f} сек")
    
    # Остальной код остается без изменений...
    segment_samples = int(segment_duration * sr)
    predictions = []
    
    for i, start in enumerate(range(0, len(y_processed), segment_samples)):
        segment = y_processed[start:start+segment_samples]
        print(f"Сегмент {i}: длина {len(segment)/sr:.2f} сек (требуется {segment_duration/2:.2f} сек)")
        if len(segment) < segment_samples//2:
            print("Пропуск сегмента - слишком короткий")
            continue
            
        hop_length = max(1, len(segment) // (target_time_steps - 1))
        S = librosa.feature.melspectrogram(y=segment, sr=sr, n_mels=128,
                                         hop_length=512, n_fft=2048)
        S_db = librosa.power_to_db(S, ref=np.max)
        
        print(f"Форма S_db перед изменением размера: {S_db.shape}")
        if S_db.shape[1] != target_time_steps:
            S_db = librosa.util.fix_length(S_db, size=target_time_steps, axis=1)
        print(f"Форма S_db после изменения размера: {S_db.shape}")
        
        if i == 0:
            plt.figure(figsize=(15, 5))
            librosa.display.specshow(S_db, sr=sr, x_axis='time', y_axis='mel')
            plt.colorbar(format='%+2.0f dB')
            plt.title(f"Обработанная спектрограмма\n(Форма: {S_db.shape})")
            plt.show()

        S_norm = (S_db - S_db.min()) / (S_db.max() - S_db.min() + 1e-10)
        input_data = S_norm[np.newaxis, ..., np.newaxis]
        print(f"Форма input_data перед предсказанием: {input_data.shape}")
        
        pred = model.predict(input_data, verbose=0)[0][0]
        predictions.append(pred)
        
        print(f"Сегмент {i}: p={pred:.4f}")
        
        if 0.1 < pred < 0.9:
            print(f"Сегмент {i}: p={pred:.4f} (сомнительный)")
        elif pred >= 0.9:
            print(f"Сегмент {i}: p={pred:.4f} (явная скрипка)")
    avg_pred = np.mean(predictions) if predictions else 0.0
    return predictions, avg_pred


def fix_input_layer(config):
    if 'batch_shape' in config:
        config.pop('batch_shape')  # Удаляем проблемный параметр
    return InputLayer.from_config(config)

# Пример использования
if __name__ == "__main__":
    # file_path = 'lettersabc/a/balanced_instruments/wav/muse_score/Скрипка_1.wav'
    # file_path = 'lettersabc/a/balanced_instruments/wav/muse_score/Кларнет_1.wav'
    # file_path = 'lettersabc/a/balanced_instruments/wav/muse_score/Банджо_1.wav'
    # file_path = 'whispered-dreams-20240601-041419.wav'
    file_path = 'Violin_for_test1.wav'
    # file_path = 'krasivaya-i-krasivaya-fortepiannaya-muzyika-40481.wav'
    # file_path = 'rahmaninov._prelyudiya_do-diez_minor.wav'
    # file_path = '526047d4315ba6b.wav'
    
    try:
        model = tf.keras.models.load_model('super_C/best_model.h5', compile=False)
        tf.saved_model.save(model, 'fixed_model')
        print("Модель успешно пересохранена в новом формате!")
                
        # с_model = load_model('best_model.h5')
        # tf.saved_model.save(с_model, 'best_model_saved')
        # model = load_model('super_C/best_model.h5')
        print("Модель успешно загружена")
        
        # Проверка файла
        info = sf.info(file_path)
        print(f"\nИнформация о файле:")
        print(f"Формат: {'стерео' if info.channels > 1 else 'моно'}")
        print(f"Частота: {info.samplerate} Hz")
        print(f"Длительность: {info.duration:.2f} сек")
        
        print("\nОбработка файла...")
        predictions, avg_pred = debug_predict(file_path, model)
        
        print("\nРезультаты:")
        print(f"Всего сегментов: {len(predictions)}")
        print(f"Средняя вероятность: {avg_pred:.4f}")
        print(f"Итоговый класс: {'violin' if avg_pred > 0.5 else 'other'}")
        
        # Визуализация распределения вероятностей
        plt.figure(figsize=(10, 5))
        plt.hist(predictions, bins=20, range=(0, 1))
        plt.title("Распределение вероятностей по сегментам")
        plt.xlabel("Вероятность скрипки")
        plt.ylabel("Количество сегментов")
        plt.show()
        
    except Exception as e:
        print(f"Ошибка: {str(e)}")