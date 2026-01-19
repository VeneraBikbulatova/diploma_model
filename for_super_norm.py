import pretty_midi
import os
from multiprocessing import Pool

def crop_midi(midi_path):
    temp_path = midi_path + ".tmp"
    try:
        midi_data = pretty_midi.PrettyMIDI(midi_path)
        if len(midi_data.instruments) != 1:
            print("Файл должен содержать ровно один инструмент")
            return
        instrument = midi_data.instruments[0]
        instrument_name = pretty_midi.program_to_instrument_name(instrument.program)
        if instrument_name != 'Violin':
            max_notes = 10000
            if len(instrument.notes) > max_notes:
                instrument.notes = instrument.notes[:max_notes]
        temp_path = midi_path + ".tmp"
        midi_data.write(temp_path)
        os.replace(temp_path, midi_path)
    except Exception as e:
        if os.path.exists(temp_path):
            try:
                os.remove(temp_path)
            except Exception as e:
                print(f"Не удалось удалить временный файл {temp_path}: {str(e)}")
        return
            
            
midi_folder_a = "lmd_full/a/instruments/valid"
midi_folder_b = "lmd_full/b/instruments/valid"

def normalize(midi_folder, num_processes=os.cpu_count()):
    file_list = []
    for root, _, files in os.walk(midi_folder):
        for file in files:
            if file.lower().endswith(('.mid', '.midi')):
                file_list.append(os.path.join(root, file))
    with Pool(num_processes) as pool:
        pool.map(crop_midi, file_list)
            
def normalize_midi_instrumemts():
    normalize(midi_folder_a)
    normalize(midi_folder_b)
    
normalize_midi_instrumemts()