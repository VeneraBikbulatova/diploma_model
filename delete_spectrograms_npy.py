import os
import random
import argparse

def remove_half_file_pairs(output_dir):
    """Удаляет половину случайно выбранных пар файлов (labels + spectrograms)"""
    try:
        # Получаем все файлы меток
        label_files = [f for f in os.listdir(output_dir) if f.endswith('_labels.npy')]
        
        if not label_files:
            print("В указанной директории не найдено файлов меток.")
            return

        # Собираем базовые имена файлов (без суффиксов)
        base_names = []
        for label_file in label_files:
            base_name = label_file.replace('_labels.npy', '')
            # Проверяем существование соответствующего файла спектрограмм
            spectrogram_file = f"{base_name}_spectrograms.npy"
            if os.path.exists(os.path.join(output_dir, spectrogram_file)):
                base_names.append(base_name)
            else:
                print(f"Предупреждение: для {label_file} не найден соответствующий файл спектрограмм")

        if not base_names:
            print("Нет подходящих пар файлов для удаления.")
            return

        # Выбираем случайную половину базовых имен для удаления
        num_to_remove = max(1, len(base_names) // 2)  # Гарантируем хотя бы 1 файл
        bases_to_remove = random.sample(base_names, num_to_remove)

        # Удаляем выбранные пары файлов
        removed_count = 0
        for base_name in bases_to_remove:
            try:
                # Удаляем файл меток
                label_path = os.path.join(output_dir, f"{base_name}_labels.npy")
                if os.path.exists(label_path):
                    os.remove(label_path)
                
                # Удаляем файл спектрограмм
                spectrogram_path = os.path.join(output_dir, f"{base_name}_spectrograms.npy")
                if os.path.exists(spectrogram_path):
                    os.remove(spectrogram_path)
                
                removed_count += 1
            except Exception as e:
                print(f"Ошибка при удалении файлов для {base_name}: {str(e)}")

        # Выводим статистику
        print("\nРезультаты:")
        print(f"Всего пар файлов найдено: {len(base_names)}")
        print(f"Удалено пар файлов: {removed_count}")
        print(f"Осталось пар файлов: {len(base_names) - removed_count}")
        
    except Exception as e:
        print(f"Критическая ошибка: {str(e)}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Удаление половины пар файлов (labels + spectrograms).')
    parser.add_argument('--output_dir', type=str, required=True, 
                       help='Папка с сохраненными спектрограммами и метками')
    
    args = parser.parse_args()
    
    # Проверяем существование директории
    if not os.path.isdir(args.output_dir):
        print(f"Ошибка: директория {args.output_dir} не существует!")
        exit(1)
    
    remove_half_file_pairs(args.output_dir)