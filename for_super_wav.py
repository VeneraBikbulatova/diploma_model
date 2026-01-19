import subprocess
import os
from multiprocessing import Pool

# Используется GeneralUser-GS.sf2 в качестве звукового шрифта
SOUNDFONT_PATH = os.path.abspath("GeneralUser-GS.sf2")
SAMPLE_RATE = 44100

midi_folder_a = "lmd_full/a/instruments"
midi_folder_b = "lmd_full/b/instruments"

def run_command(cmd):
    try:
        subprocess.run(cmd, shell=True, check=True, timeout=60)
    except subprocess.TimeoutExpired:
        return False
    except Exception as e:
        return False

def midi_to_wav_folder(midi_folder, num_processes=os.cpu_count()):
    file_list = [f for f in os.listdir(midi_folder) if f.lower().endswith(".mid")]
    output_dir = os.path.join(midi_folder, "wav")
    os.makedirs(output_dir, exist_ok=True)

    commands = []
    for file in file_list:
        midi_path = os.path.abspath(os.path.join(midi_folder, file))
        output_wav = os.path.abspath(os.path.join(output_dir, os.path.splitext(file)[0] + ".wav"))
        cmd = f'fluidsynth -ni -F "{output_wav}" -r {SAMPLE_RATE} "{SOUNDFONT_PATH}" "{midi_path}"'
        commands.append(cmd)

    with Pool(num_processes) as pool:
        pool.map(run_command, commands)

            
def midi_to_wav():
    with Pool(2) as pool:
        pool.map(midi_to_wav_folder, [midi_folder_a, midi_folder_b])

midi_to_wav()