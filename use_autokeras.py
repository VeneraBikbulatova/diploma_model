import numpy as np
import os
import autokeras as ak
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
import gc
import random
import time
import shutil

def load_data_in_batches(data_dir, batch_size=100):
    """Генератор для загрузки данных"""
    files = [f for f in os.listdir(data_dir) if f.endswith('_spectrograms.npy')]
    random.shuffle(files)
    
    for i in range(0, len(files), batch_size):
        batch_files = files[i:i+batch_size]
        X_batch = []
        y_batch = []
        
        for file in batch_files:
            try:
                spect = np.load(os.path.join(data_dir, file))
                label = 1 if file.startswith('Violin') else 0
                labels = np.full((len(spect), 1), label)
                
                X_batch.append(spect)
                y_batch.append(labels)
            except Exception as e:
                continue
        
        if X_batch:
            X = np.expand_dims(np.concatenate(X_batch, axis=0), axis=-1)
            y = np.concatenate(y_batch, axis=0)
            
            if len(np.unique(y)) >= 2:
                yield X, y
            
            del X, y
            gc.collect()

def clean_directory(dir_path):
    """Безопасная очистка директории"""
    try:
        if os.path.exists(dir_path):
            shutil.rmtree(dir_path)
    except Exception as e:
        print(f"Не удалось очистить {dir_path}: {e}")

def train_single_model(train_dir, test_dir, epochs=20):
    """Обучение и сохранение одной модели"""
    # Очистка предыдущих результатов
    clean_directory('autokeras_model')
    
    # Создаем временную папку для AutoKeras
    os.makedirs('autokeras_model', exist_ok=True)
    
    # Инициализация классификатора с одним trial
    clf = ak.ImageClassifier(
        max_trials=1,  # Только одна модель
        overwrite=True,
        directory='autokeras_model',
        seed=42
    )
    
    # Callback для ранней остановки
    callbacks = [
        EarlyStopping(patience=3, monitor='val_accuracy'),
        ModelCheckpoint(
            'best_model.h5',
            monitor='val_accuracy',
            save_best_only=True,
            mode='max'
        )
    ]
    
    # Обучение
    try:
        for X_train, y_train in load_data_in_batches(train_dir, batch_size=50):
            clf.fit(
                X_train,
                y_train,
                epochs=epochs,
                callbacks=callbacks,
                validation_split=0.2,
                verbose=2
            )
            break  # Обучаем только на первом батче для одной модели
    except Exception as e:
        print(f"\nОшибка обучения: {e}")
    
    # Сохранение модели
    try:
        # Проверяем, сохранился ли checkpoint
        if os.path.exists('best_model.h5'):
            print("\nЛучшая модель сохранена как 'best_model.h5'")
            return 'best_model.h5'
        
        # Если checkpoint не создан, экспортируем модель
        model = clf.export_model()
        if model:
            model.save('single_model.h5')
            print("\nМодель сохранена как 'single_model.h5'")
            return 'single_model.h5'
    except Exception as e:
        print(f"\nОшибка сохранения модели: {e}")
    
    return None

if __name__ == "__main__":
    # Параметры
    train_dir = "lmd_full/a/spectrograms"
    test_dir = "lmd_full/b/spectrograms"
    
    # Запуск
    print("Начало обучения одной модели...")
    start = time.time()
    
    model_path = train_single_model(
        train_dir,
        test_dir,
        epochs=20
    )
    
    if model_path:
        print(f"\nМодель успешно создана и сохранена в {model_path}")
    else:
        print("\nНе удалось создать модель")
    
    print(f"Общее время выполнения: {time.time()-start:.2f} секунд")