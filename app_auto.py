import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.utils import custom_object_scope
import tkinter as tk
from tkinter import messagebox

# 1. Определяем кастомный слой
class CastToFloat32(tf.keras.layers.Layer):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
    
    def call(self, inputs):
        return tf.cast(inputs, tf.float32)
    
    def get_config(self):
        return super().get_config()

# 2. Функция безопасной загрузки модели
def load_safe_model(model_path):
    custom_objects = {
        'CastToFloat32': CastToFloat32,
        # Добавьте другие кастомные слои здесь, если они есть
    }
    
    try:
        with custom_object_scope(custom_objects):
            model = load_model(model_path, compile=False)
        return model, "Модель успешно загружена"
    except Exception as e:
        return None, f"Ошибка загрузки: {str(e)}"

class AudioMLApp:
    def __init__(self, root):
        self.root = root
        # ... (остальной код инициализации) ...
        
        # 3. Загружаем модель с обработкой кастомного слоя
        self.model, status = load_safe_model('best_model.h5')
        if self.model is None:
            messagebox.showerror("Ошибка", status)
            self.root.destroy()
            return
        
        print("Модель успешно загружена!")
        # ... (остальной код приложения) ...

if __name__ == "__main__":
    root = tk.Tk()
    app = AudioMLApp(root)
    root.mainloop()