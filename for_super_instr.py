import os
import pretty_midi
from multiprocessing import Pool

def get_instrument_name(program_number):
    general_midi_instruments = {
        0: 'Acoustic Grand Piano',
        1: 'Bright Acoustic Piano',
        6: 'Harpsichord',
        8: 'Celesta',
        9: 'Glockenspiel',
        11: 'Vibraphone',
        12: 'Marimba',
        13: 'Xylophone',
        14: 'Tubular Bells',
        15: 'Dulcimer',
        17: 'Percussive Organ',
        19: 'Church Organ',
        21: 'Accordion',
        22: 'Harmonica',
        24: 'Acoustic Guitar _nylon',
        25: 'Acoustic Guitar _steel',
        31: 'Guitar Harmonics',
        32: 'Acoustic Bass',
        40: 'Violin',
        41: 'Viola',
        42: 'Cello',
        43: 'Contrabass',
        44: 'Tremolo Strings',
        45: 'Pizzicato Strings',
        46: 'Orchestral Harp',
        47: 'Timpani',
        56: 'Trumpet',
        57: 'Trombone',
        58: 'Tuba',
        59: 'Muted Trumpet',
        60: 'French Horn',
        65: 'Alto Sax',
        67: 'Baritone Sax',
        68: 'Oboe',
        69: 'English Horn',
        70: 'Bassoon',
        71: 'Clarinet',
        72: 'Piccolo',
        73: 'Flute',
        75: 'Pan Flute',
        79: 'Ocarina',
        105: 'Banjo',
        108: 'Kalimba',
        109: 'Bagpipe',
        110: 'Fiddle',
        112: 'Tinkle Bell',
    }
    return general_midi_instruments.get(program_number, f'Unknown_Instrument_{program_number}')

def process_midi_file(midi_path, output_folder):
    try:
        midi_data = pretty_midi.PrettyMIDI(midi_path)
        
        os.makedirs(output_folder, exist_ok=True)

        for instrument in midi_data.instruments:
            program_number = instrument.program
            instrument_name = get_instrument_name(program_number)
            
            safe_name = "".join(c if c.isalnum() or c in (' ', '_') else '_' for c in instrument_name)
            output_path = os.path.join(output_folder, f'{safe_name}.mid')
            
            if os.path.exists(output_path):
                
                try:
                    existing_midi = pretty_midi.PrettyMIDI(output_path)
                    # Находим соответствующий инструмент или создаем новый
                    found = False
                    for existing_instr in existing_midi.instruments:
                        if existing_instr.program == program_number:
                            existing_instr.notes.extend(instrument.notes)
                            found = True
                            break
                
                    if not found:
                        new_instrument = pretty_midi.Instrument(program=program_number)
                        new_instrument.notes = instrument.notes.copy()
                        existing_midi.instruments.append(new_instrument)
                
                    existing_midi.write(output_path)
                except Exception as e:
                    print(f"Ошибка при обновлении {output_path}: {str(e)}")
                    continue
            else:
                try:
                    # Если нет файла для инструмента - здесь создаем его
                    new_midi = pretty_midi.PrettyMIDI()
                    new_instrument = pretty_midi.Instrument(program=program_number)
                    new_instrument.notes = instrument.notes.copy()
                    new_midi.instruments.append(new_instrument)
                    new_midi.write(output_path)
                except Exception as e:
                    print(f"Ошибка при создании {output_path}: {str(e)}")
                    continue
        return True
    except Exception as e:
        print(f"Ошибка при обработке {midi_path}:")
        return False
            
# def process_all_midi(input_folder, output_folder, num_processes=os.cpu_count()):
#     file_list = []
#     for root, _, files in os.walk(input_folder):
#         for file in files:
#             if file.lower().endswith('.mid'):
#                 file_path = os.path.join(root, file)
#                 file_list.append((file_path, output_folder))    
    
#     print(f"Найдено {len(file_list)} MIDI-файлов в {input_folder}")
    
#     if not file_list:
#         print("Нет MIDI-файлов для обработки!")
#         return
    
#     with Pool(num_processes) as pool:
#         results = pool.map(process_midi_file, file_list)
    
#     success_count = sum(1 for r in results if r)
#     print(f"Успешно обработано {success_count}/{len(file_list)} файлов")
        
# # Сначала используем на lmd_full/a, потом на lmd_full/b, чтобы в обучающей и тестовой выборках были различные данные
# midi_folder_a = 'lmd_full/a'
# midi_folder_b = 'lmd_full/b'
# output_folder_a = os.path.join(midi_folder_a, 'instruments')
# output_folder_b = os.path.join(midi_folder_b, 'instruments')
# def midi_to_instruments():
#     try:
#         with Pool(2) as pool:
#             pool.starmap(process_all_midi, [(midi_folder_a, output_folder_a), 
#                                       (midi_folder_b, output_folder_b)])
#     except Exception as e:
#         return False

# midi_to_instruments()

def process_midi_batch(file_list, output_folder, num_processes):
    if not file_list:
        print("Нет файлов для обработки!")
        return 0

    # Явно создаем выходную папку
    os.makedirs(output_folder, exist_ok=True)
   
    # Подготавливаем аргументы для каждого процесса
    tasks = [(fp, output_folder) for fp in file_list]
    
    with Pool(num_processes) as pool:
        results = pool.starmap(process_midi_file, tasks)
    
    success_count = sum(1 for r in results if r)
    
    print(f"Обработано: {success_count}/{len(file_list)} файлов")
    return success_count

def midi_to_instruments():
    base_folders = ['lmd_full/b']
    num_processes = 4  # Можно увеличить для более мощных систем
    
    for folder in base_folders:
        if not os.path.exists(folder):
            print(f"ОШИБКА: Папка {folder} не существует!")
            continue
        
        output_folder = os.path.join(folder, 'instruments')
        file_list = []
        
        for root, _, files in os.walk(folder):
            for file in files:
                if file.lower().endswith('.mid'):
                    file_path = os.path.join(root, file)
                    # Исключаем файлы из папки instruments
                    if 'instruments' not in file_path.split(os.sep):
                        file_list.append(file_path)
        
        print(f"\nНайдено {len(file_list)} MIDI-файлов в {folder}")
        
        if file_list:
            success = process_midi_batch(file_list, output_folder, num_processes)
            
            # Проверяем результат
            if success > 0:
                print(f"Проверяем выходную папку {output_folder}...")
                if os.path.exists(output_folder):
                    created_files = [f for f in os.listdir(output_folder) if f.endswith('.mid')]
                    print(f"Создано {len(created_files)} файлов инструментов")
                else:
                    print("Выходная папка не создана!")
            else:
                print("Ни один файл не был обработан успешно")

if __name__ == "__main__":
    print("Запуск обработки MIDI файлов...")
    midi_to_instruments()