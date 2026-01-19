import os
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
import librosa
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.models import load_model
import soundfile as sf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input
from tensorflow.keras.layers import InputLayer
from matplotlib.backends.backend_pdf import PdfPages
import pandas as pd

def debug_predict(file_path, model, pdf, segment_duration=3.0, target_time_steps=36):
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
    
    # Создание фигуры для waveform
    fig_wave = plt.figure(figsize=(15, 5))
    plt.subplot(2, 1, 1)
    plt.plot(y)
    plt.title("Original waveform")
    plt.subplot(2, 1, 2)
    plt.plot(y_processed)
    plt.title("Processed waveform")
    pdf.savefig(fig_wave)
    plt.close(fig_wave)
    
    # Создание текстовых данных
    text_data = [
        f"Длина после загрузки: {len(y)/sr:.2f} сек",
        f"Длина после обрезки: {len(y_trimmed)/sr:.2f} сек",
        f"Длина после обработки: {len(y_processed)/sr:.2f} сек"
    ]
    
    # Остальной код остается без изменений...
    segment_samples = int(segment_duration * sr)
    predictions = []
    segment_info = []
    
    for i, start in enumerate(range(0, len(y_processed), segment_samples)):
        segment = y_processed[start:start+segment_samples]
        segment_length = len(segment)/sr
        text_data.append(f"Сегмент {i}: длина {segment_length:.2f} сек (требуется {segment_duration/2:.2f} сек)")
        
        if len(segment) < segment_samples//2:
            text_data.append("Пропуск сегмента - слишком короткий")
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
            fig_spec = plt.figure(figsize=(15, 5))
            librosa.display.specshow(S_db, sr=sr, x_axis='time', y_axis='mel')
            plt.colorbar(format='%+2.0f dB')
            plt.title(f"Обработанная спектрограмма\n(Форма: {S_db.shape})")
            pdf.savefig(fig_spec)
            plt.close(fig_spec)

        S_norm = (S_db - S_db.min()) / (S_db.max() - S_db.min() + 1e-10)
        input_data = S_norm[np.newaxis, ..., np.newaxis]
        print(f"Форма input_data перед предсказанием: {input_data.shape}")

        pred = model.predict(input_data, verbose=0)[0][0]
        predictions.append(pred)
        
        segment_info.append({
            'Сегмент': i,
            'Длина (сек)': segment_length,
            'Вероятность': pred,
            'Класс': 'скрипка' if pred > 0.5 else 'другое',
            'Примечание': 'сомнительный' if 0.1 < pred < 0.9 else ('явная скрипка' if pred >= 0.9 else '')
        })
        
        text_data.append(f"Сегмент {i}: p={pred:.4f}")
        
        if 0.1 < pred < 0.9:
            text_data.append(f"Сегмент {i}: p={pred:.4f} (сомнительный)")
        elif pred >= 0.9:
            text_data.append(f"Сегмент {i}: p={pred:.4f} (явная скрипка)")
    
    # Создание таблицы с результатами
    df = pd.DataFrame(segment_info)
    if not df.empty:
        fig_table = plt.figure(figsize=(10, len(df)*0.5))
        ax = plt.subplot(111)
        ax.axis('off')
        tbl = plt.table(cellText=df.values, 
                        colLabels=df.columns, 
                        loc='center',
                        cellLoc='center')
        tbl.auto_set_font_size(False)
        tbl.set_fontsize(10)
        tbl.scale(1, 1.5)
        plt.title("Результаты по сегментам")
        pdf.savefig(fig_table)
        plt.close(fig_table)
    
    # Визуализация распределения вероятностей
    if predictions:
        fig_hist = plt.figure(figsize=(10, 5))
        plt.hist(predictions, bins=20, range=(0, 1))
        plt.title("Распределение вероятностей по сегментам")
        plt.xlabel("Вероятность скрипки")
        plt.ylabel("Количество сегментов")
        pdf.savefig(fig_hist)
        plt.close(fig_hist)
    
    avg_pred = np.mean(predictions) if predictions else 0.0
    
    # Добавление текстовой информации
    text_data.extend([
        "\nРезультаты:",
        f"Всего сегментов: {len(predictions)}",
        f"Средняя вероятность: {avg_pred:.4f}",
        f"Итоговый класс: {'violin' if avg_pred > 0.5 else 'other'}"
    ])
    
    # Создание страницы с текстом
    fig_text = plt.figure(figsize=(10, len(text_data)*0.3))
    ax = plt.subplot(111)
    ax.axis('off')
    plt.text(0, 1, "\n".join(text_data), fontsize=10, verticalalignment='top')
    pdf.savefig(fig_text)
    plt.close(fig_text)
    
    return predictions, avg_pred

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
    
    # Создаем PDF файл для сохранения результатов
    with PdfPages('analysis_report.pdf') as pdf:
        try:
            model = tf.keras.models.load_model('best_model.keras')
            
            # Проверка файла
            info = sf.info(file_path)
            file_info = [
                "Информация о файле:",
                f"Формат: {'стерео' if info.channels > 1 else 'моно'}",
                f"Частота: {info.samplerate} Hz",
                f"Длительность: {info.duration:.2f} сек",
                "\nОбработка файла..."
            ]
            
            fig_file_info = plt.figure(figsize=(10, len(file_info)*0.3))
            ax = plt.subplot(111)
            ax.axis('off')
            plt.text(0, 1, "\n".join(file_info), fontsize=10, verticalalignment='top')
            pdf.savefig(fig_file_info)
            plt.close(fig_file_info)
            
            predictions, avg_pred = debug_predict(file_path, model, pdf)
            
        except Exception as e:
            # Сохраняем информацию об ошибке в PDF
            fig_error = plt.figure(figsize=(10, 2))
            ax = plt.subplot(111)
            ax.axis('off')
            plt.text(0, 1, f"Ошибка: {str(e)}", fontsize=12, color='red')
            pdf.savefig(fig_error)
            plt.close(fig_error)