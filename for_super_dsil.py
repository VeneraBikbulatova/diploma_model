import os
import pretty_midi
from multiprocessing import Pool

INSTRUMENT_NOTES = {
    'Accordion': (40, 108),
    'Acoustic Bass': (28, 67),
    'Acoustic Guitar _nylon': (52, 91),
    'Acoustic Guitar _steel': (52, 91),
    'Acoustic Grand Piano': (21, 108),
    'Alto Sax': (58, 90),
    'Bagpipe': (62, 81),
    'Banjo': (55, 79),
    'Baritone Sax': (58, 90),
    'Bassoon': (35, 77),
    'Bright Acoustic Piano': (21, 108),
    'Celesta': (60, 108),
    'Cello': (36, 81),
    'Church Organ': (36, 96),
    'Clarinet': (52, 91),
    'Contrabass': (28, 71),
    'Dulcimer': (48, 96),
    'English Horn': (48, 83),
    'Fiddle': (55, 105),
    'Flute': (60, 96),
    'French Horn': (35, 77),
    'Glockenspiel': (77, 108),
    'Guitar Harmonics': (59, 101),
    'Harmonica': (54, 96),
    'Harpsichord': (29, 89),
    'Kalimba': (50, 89),
    'Marimba': (36, 96),
    'Muted Trumpet': (54, 84),
    'Oboe': (58, 89),
    'Ocarina': (55, 96),
    'Orchestral Harp': (29, 103),
    'Pan Flute': (36, 93),
    'Percussive Organ': (36, 96),
    'Piccolo': (74, 108),
    'Pizzicato Strings': (28, 100),
    'Timpani': (74, 93),
    'Tinkle Bell': (84, 115),
    'Tremolo Strings': (28, 100),
    'Trombone': (31, 77),
    'Trumpet': (60, 83),
    'Tuba': (24, 64),
    'Tubular Bells': (55, 84),
    'Vibraphone': (48, 96),
    'Viola': (48, 88),
    'Violin': (55, 105),
    'Xylophone': (53, 96)
}

def is_valid_instrument(instrument_name):
    return instrument_name in INSTRUMENT_NOTES

def remove_midi_pauses(midi_path, midi_folder, output_path=None):
    try:        
        midi = pretty_midi.PrettyMIDI(midi_folder+"/"+midi_path)
    
        instrument_name = midi_path.split('.')[0]
        
        all_notes = [note for inst in midi.instruments for note in inst.notes]
    
        if not all_notes:
            return False
    
        if not is_valid_instrument(instrument_name):
            return False
    
        instrument_range = INSTRUMENT_NOTES[instrument_name]

        first_note_start = min(n.start for n in all_notes)
        last_note_end = max(n.end for n in all_notes)
    
        for instrument in midi.instruments:
            if not instrument.notes:
                continue
            
            if not instrument_range:
                continue
            
            valid_notes = []
            current_time = 0.0
            
            instrument.notes.sort(key=lambda x: x.start)
        
            current_time = 0.0
            
            for note in instrument.notes:
                if instrument_range[0] <= note.pitch <= instrument_range[1]:
                    note_duration = note.end - note.start
                    note.start = current_time
                    note.end = current_time + note_duration
                    current_time += note_duration
                    valid_notes.append(note)
        
            instrument.notes = valid_notes
        
        if output_path is None:
            base, ext = os.path.splitext(midi_path)
            output_path = f"{midi_folder}/valid/{base}_no_pauses{ext}"
            
        midi.instruments = [inst for inst in midi.instruments if inst.notes]

        if not midi.instruments:
            return False

        midi.write(output_path)
        return True
    
    except Exception as e:
        return False

def midi_detele_silence(num_processes=os.cpu_count()):
    file_list_a = [(f, output_folder_a) for f in os.listdir(output_folder_a) if f.endswith('.mid')]
    file_list_b = [(f, output_folder_b) for f in os.listdir(output_folder_b) if f.endswith('.mid')]
    
    with Pool(num_processes) as pool:
        pool.starmap(remove_midi_pauses, file_list_a)
        pool.starmap(remove_midi_pauses, file_list_b)
        
midi_detele_silence()