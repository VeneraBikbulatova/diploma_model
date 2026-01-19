import tkinter as tk
from tkinter import filedialog, messagebox
import sounddevice as sd
import numpy as np
import librosa
import soundfile as sf
from tensorflow.keras.models import load_model
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import threading
import queue
import time
import tensorflow as tf

class AudioMLApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Violin Classifier")
        self.root.geometry("800x600")
        self.recording = False
        self.plot_canvases = []
        
        # Очередь для межпоточного взаимодействия
        self.queue = queue.Queue()
        
        # Загрузка модели
        try:
            self.model = tf.keras.models.load_model('best_model.keras', compile=False)
            self.model_status = "Модель загружена успешно"
        except Exception as e:
            self.model_status = f"Ошибка загрузки модели: {str(e)}"
            messagebox.showerror("Ошибка", self.model_status)
        
        self.figures = [
            plt.Figure(figsize=(7, 2.5), dpi=100),
            plt.Figure(figsize=(7, 2.5), dpi=100),
            plt.Figure(figsize=(7, 4.5), dpi=100)
        ]        
        self.full_spec_canvas = None
        self.full_spec_fig = self.figures[2]
        
        # GUI элементы
        self.create_widgets()
        
        # Проверка очереди каждые 100 мс
        self.root.after(100, self.process_queue)
    
    def create_widgets(self):
        # Основной контейнер с прокруткой
        main_container = tk.Frame(self.root)
        main_container.pack(fill=tk.BOTH, expand=True)

        # 1. Создаем систему прокрутки
        self.scroll_canvas = tk.Canvas(main_container)
        scrollbar = tk.Scrollbar(main_container, orient=tk.VERTICAL, command=self.scroll_canvas.yview)
        self.scrollable_frame = tk.Frame(self.scroll_canvas)
    
        # Настройка прокрутки
        self.scrollable_frame.bind(
            "<Configure>",
            lambda e: self.scroll_canvas.configure(
                scrollregion=self.scroll_canvas.bbox("all")
            )
        )
    
        self.scroll_canvas.create_window((0, 0), window=self.scrollable_frame, anchor="nw")
        self.scroll_canvas.configure(yscrollcommand=scrollbar.set)
    
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        self.scroll_canvas.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
    
        # 2. Статус модели
        self.status_label = tk.Label(
            self.scrollable_frame, 
            text=self.model_status, 
            wraplength=700,
            justify=tk.LEFT
        )
        self.status_label.pack(pady=10, padx=10, anchor=tk.W)
    
        # 3. Кнопки управления
        btn_frame = tk.Frame(self.scrollable_frame)
        btn_frame.pack(pady=10, fill=tk.X)
    
        self.record_btn = tk.Button(
            btn_frame, 
            text="Записать аудио (5 сек)", 
            command=self.start_recording,
            width=25,
            height=2
        )
        self.record_btn.pack(side=tk.LEFT, padx=10, expand=True)
    
        self.select_btn = tk.Button(
            btn_frame, 
            text="Выбрать файл", 
            command=self.select_audio_file,
            width=25,
            height=2
        )
        self.select_btn.pack(side=tk.LEFT, padx=10, expand=True)
    
        # 4. Графики сигналов
        self.graphs_frame = tk.Frame(self.scrollable_frame)
        self.graphs_frame.pack(pady=20, fill=tk.BOTH, expand=True)
    
        # Создаем фигуры и холсты для графиков
        graph_container1 = tk.Frame(self.graphs_frame)
        graph_container1.pack(pady=5, fill=tk.BOTH, expand=True)
        tk.Label(graph_container1, text="Исходный сигнал", font=('Arial', 10, 'bold')).pack(anchor=tk.W)
        canvas1 = FigureCanvasTkAgg(self.figures[0], master=graph_container1)
        canvas1.get_tk_widget().pack(fill=tk.BOTH, expand=True)
        self.plot_canvases.append(canvas1)
    
        graph_container2 = tk.Frame(self.graphs_frame)
        graph_container2.pack(pady=5, fill=tk.BOTH, expand=True)
        tk.Label(graph_container2, text="Обработанный сигнал", font=('Arial', 10, 'bold')).pack(anchor=tk.W)
        canvas2 = FigureCanvasTkAgg(self.figures[1], master=graph_container2)
        canvas2.get_tk_widget().pack(fill=tk.BOTH, expand=True)
        self.plot_canvases.append(canvas2)
    
        spec_container = tk.Frame(self.graphs_frame)
        spec_container.pack(pady=5, fill=tk.BOTH, expand=True)
        tk.Label(spec_container, text="Распределение вероятностей по сегментам", font=('Arial', 10, 'bold')).pack(anchor=tk.W)
        self.full_spec_canvas = FigureCanvasTkAgg(self.figures[2], master=spec_container)
        self.full_spec_canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
    
        # 5. Блок результатов
        self.result_frame = tk.Frame(self.scrollable_frame)
        self.result_frame.pack(pady=20, fill=tk.X)

        self.result_label = tk.Label(
            self.result_frame, 
            text="", 
            font=('Arial', 12),
            wraplength=700,
            justify=tk.LEFT,
            bg="#f0f0f0",
            padx=10,
            pady=10
        )
        self.result_label.pack(fill=tk.X)

        # 6. Настройка прокрутки колесом мыши
        def on_mousewheel(event):
            self.scroll_canvas.yview_scroll(int(-1*(event.delta/120)), "units")

        self.scroll_canvas.bind_all("<MouseWheel>", on_mousewheel)

        # Добавляем отступ внизу для красоты
        tk.Frame(self.scrollable_frame, height=20).pack()
        
    def plot_probability_distribution(self, predictions):
        try:
            # Очищаем предыдущий график
            self.full_spec_fig.clf()
            ax = self.full_spec_fig.add_subplot(111)
        
            # Создаем гистограмму
            ax.hist(predictions, bins=20, range=(0, 1), color='skyblue', edgecolor='black')
            ax.set_title('Распределение вероятностей по сегментам')
            ax.set_xlabel('Вероятность скрипки')
            ax.set_ylabel('Количество сегментов')
            ax.grid(True, linestyle='--', alpha=0.7)
        
            # Принудительное обновление холста
            self.full_spec_canvas.draw()
        
        except Exception as e:
            print(f"Ошибка при построении гистограммы: {str(e)}")
            self.queue.put((messagebox.showerror, ["Ошибка", f"Не удалось построить гистограмму: {str(e)}"]))
    
    def process_queue(self):
        try:
            while True:
                callback, args = self.queue.get_nowait()
                callback(*args)
        except queue.Empty:
            pass
        finally:
            self.root.after(100, self.process_queue)

    def start_recording(self):
        self.record_btn.config(state=tk.DISABLED)
        self.recording = True
        self.update_recording_timer(5)  # Начинаем отсчёт с 5 секунд
    
        threading.Thread(target=self.record_audio, daemon=True).start()

    def update_recording_timer(self, seconds_left):
        if self.recording and seconds_left > 0:
            self.status_label.config(text=f"Запись... Осталось {int(seconds_left)} сек")  # Используем f-строку
            self.root.after(1000, self.update_recording_timer, seconds_left-1)
        elif not self.recording:
            self.status_label.config(text="Запись прервана")
        else:
            self.status_label.config(text="Обработка записи...")
        
    def record_audio(self):
        try:
            fs = 22050
            duration = 5
        
            recording = sd.rec(int(duration * fs), samplerate=fs, channels=1)
        
            for _ in range(duration * 10):
                if not self.recording:
                    sd.stop()
                    return
                time.sleep(0.1)
            
            sd.wait()
            filename = "recording.wav"
            sf.write(filename, recording, fs)
        
            self.queue.put((self.process_audio, [filename]))
        
        except Exception as e:
            self.recording = False
            self.queue.put((messagebox.showerror, ["Ошибка", str(e)]))
            self.queue.put((self.reset_ui, []))

    def reset_ui(self):
        self.record_btn.config(state=tk.NORMAL)
        self.status_label.config(text="Готов к записи")
        self.recording = False
    
    def enable_record_button(self):
        self.record_btn.config(state=tk.NORMAL)
        self.status_label.config(text="Готово к записи")
        
    def select_audio_file(self):
        filename = filedialog.askopenfilename(filetypes=[("Audio files", "*.wav *.mp3")])
        if filename:
            self.process_audio(filename)
    
    def process_audio(self, file_path):
        try:
            # Очистка предыдущих графиков
            for fig in self.figures:
                fig.clf()
        
            # Загрузка и обработка аудио
            y, sr = librosa.load(file_path, sr=22050, mono=True)
        
            # Обрезка тишины
            y_trimmed = librosa.effects.trim(y, top_db=25)[0]
        
            y_clean = librosa.effects.preemphasis(y_trimmed)
            S = librosa.stft(y_clean)
            S_mag = np.abs(S)
            mask = librosa.util.softmask(S_mag, 0.2 * S_mag, power=2)
            y_processed = librosa.istft(S * mask)
        
            # Визуализация
            self.plot_waveforms(y, y_processed, sr)
        
            predictions, avg_pred = self.predict_audio(y_processed, sr)
        
            self.plot_probability_distribution(predictions)
        
            avg_pred_float = float(avg_pred)
            result_text = (f"Средняя вероятность: {avg_pred_float:.4f}\n"
                     f"Итоговый класс: {'СКРИПКА' if avg_pred_float > 0.5 else 'ДРУГОЙ ИНСТРУМЕНТ'}")
            self.result_label.config(text=result_text)
        
            self.status_label.config(text=f"Обработан файл: {str(file_path)}")
            self.record_btn.config(state=tk.NORMAL)
        
        except Exception as e:
            messagebox.showerror("Ошибка", str(e))
            self.record_btn.config(state=tk.NORMAL)
    
    def plot_waveforms(self, original, processed, sr):
        # Оригинальный сигнал
        ax0 = self.figures[0].add_subplot(111)
        ax0.plot(original)
        ax0.set_title("Исходный аудиосигнал")
        self.plot_canvases[0].draw()
        
        # Обработанный сигнал
        ax1 = self.figures[1].add_subplot(111)
        ax1.plot(processed)
        ax1.set_title("Обработанный аудиосигнал")
        self.plot_canvases[1].draw()
    
    def predict_audio(self, y_processed, sr, segment_duration=3.0, target_time_steps=44):
        try:
            segment_samples = int(segment_duration * sr)
            predictions = []
    
            for start in range(0, len(y_processed), segment_samples):
                segment = y_processed[start:start+segment_samples]
        
                if len(segment) < segment_samples//2:
                    continue
            
                S = librosa.feature.melspectrogram(y=segment, sr=sr, n_mels=128,
                                     hop_length=512, n_fft=2048)
                S_db = librosa.power_to_db(S, ref=np.max)
        
                if S_db.shape[1] != target_time_steps:
                    S_db = librosa.util.fix_length(S_db, size=target_time_steps, axis=1)
        
                S_norm = (S_db - S_db.min()) / (S_db.max() - S_db.min() + 1e-10)
                input_data = S_norm[np.newaxis, ..., np.newaxis]
        
                pred = self.model.predict(input_data, verbose=0)[0][0]
                predictions.append(float(pred))
    
            avg_pred = float(np.mean(predictions)) if predictions else 0.0
            return predictions, avg_pred
        except Exception as e:
            self.queue.put((messagebox.showerror, ["Ошибка предсказания", str(e)]))
            return [], 0.0

if __name__ == "__main__":
    root = tk.Tk()
    app = AudioMLApp(root)
    root.mainloop()