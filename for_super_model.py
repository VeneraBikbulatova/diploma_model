import librosa
import soundfile as sf
import os
import numpy as np
from multiprocessing import Pool
from sklearn.model_selection import train_test_split
import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
import matplotlib.pyplot as plt
from sklearn.utils import resample
from sklearn.metrics import classification_report, confusion_matrix
import joblib
from matplotlib.backends.backend_pdf import PdfPages
from sklearn.metrics import roc_curve, auc
from functools import partial

def augment_audio_segment(segment, sr=22050):
    augmented = []
    
    noise = 0.005 * np.random.randn(len(segment))
    augmented.append(segment + noise)
    
    augmented.append(librosa.effects.pitch_shift(segment, sr=sr, n_steps=2))
    augmented.append(librosa.effects.pitch_shift(segment, sr=sr, n_steps=-2))
    
    augmented.append(librosa.effects.time_stretch(segment, rate=0.8))
    augmented.append(librosa.effects.time_stretch(segment, rate=1.2))
    
    return augmented

midi_folder_a = "lmd_full/a"
midi_folder_b = "lmd_full/b"
   
wav_folder_a = os.path.join(midi_folder_a, "segments")
wav_folder_b = os.path.join(midi_folder_b, "segments")

output_dir = "preprocessed_data"
segment_duration = 1.0
max_duration_per_instrument = 5 * 60
target_sr = 22050
n_mels = 128
img_height, img_width = n_mels, 128

os.makedirs(output_dir, exist_ok=True)

def process_audio_file(file_path, label, max_segments=None, augment=True):
    try:
        y, sr = librosa.load(file_path, sr=target_sr)
    except Exception as e:
        return []
    
    segment_samples = int(segment_duration * sr)
    fixed_time_steps = 128
    
    segments = []
    for start in range(0, len(y), segment_samples):
        end = start + segment_samples
        if end > len(y):
            break
        segment = y[start:end]
        
        S = librosa.feature.melspectrogram(y=segment, sr=sr, n_mels=n_mels)
        S_db = librosa.power_to_db(S, ref=np.max)
        
        if S_db.shape[1] < fixed_time_steps:
            pad_width = ((0, 0), (0, fixed_time_steps - S_db.shape[1]))
            S_db = np.pad(S_db, pad_width, mode='constant')
        else:
            S_db = S_db[:, :fixed_time_steps]
        
        segments.append((S_db, label))
    
    if augment and label == 1:
            augmented = augment_audio_segment(segment, sr)
            for aug_segment in augmented:
                S_aug = librosa.feature.melspectrogram(y=aug_segment, sr=sr, n_mels=n_mels)
                S_db_aug = librosa.power_to_db(S_aug, ref=np.max)
                segments.append((S_db_aug, label))
    
    return segments


def create_cnn_model(input_shape):
    inputs = tf.keras.Input(shape=input_shape)
    
    # Data augmentation layers
    x = tf.keras.layers.GaussianNoise(0.005)(inputs)
    x = tf.keras.layers.RandomZoom(0.2)(x)
    
    # CNN architecture
    x = layers.Conv2D(128, (3, 3), activation='relu', padding='same')(x)
    x = layers.BatchNormalization()(x)
    x = layers.MaxPooling2D((2, 2))(x)
    
    x = layers.Conv2D(256, (3, 3), activation='relu', padding='same')(x)
    x = layers.BatchNormalization()(x)
    x = layers.MaxPooling2D((2, 2))(x)
    
    x = layers.Conv2D(512, (5, 5), activation='relu', padding='same')(x)
    x = layers.BatchNormalization()(x)
    x = layers.MaxPooling2D((2, 2))(x)
    
    x = layers.GlobalAveragePooling2D()(x)
    
    x = layers.Dense(128, activation='relu')(x)
    x = layers.Dropout(0.6)(x)
    outputs = layers.Dense(1, activation='sigmoid')(x)
    
    model = tf.keras.Model(inputs=inputs, outputs=outputs)
    
    optimizer = tf.keras.optimizers.Adam(learning_rate=0.0001)
    
    model.compile(
        optimizer=optimizer,
        loss='binary_crossentropy',
        metrics=['accuracy', tf.keras.metrics.AUC(), tf.keras.metrics.Recall()]
    )
    
    return model

def train_model(X_train, y_train, X_val, y_val, input_shape):
    model = create_cnn_model(input_shape)
    model.summary()
    
    os.makedirs('saved_models', exist_ok=True)
    os.makedirs('training_metrics', exist_ok=True)
    
    callbacks = [
        EarlyStopping(patience=10, restore_best_weights=True),
        ModelCheckpoint('best_model.keras', save_best_only=True),
        ModelCheckpoint(
            filepath='saved_models/epoch{epoch:03d}_valacc{val_accuracy:.3f}_valloss{val_loss:.3f}.keras',
            monitor='val_accuracy',
            save_best_only=False,
            save_weights_only=False,
            mode='max',
            verbose=0
        ),
        tf.keras.callbacks.CSVLogger('training_metrics/training_log.csv')
    ]

    history = model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=50,
        batch_size=32,
        callbacks=callbacks,
    )
    
    return model, history


def save_training_report(history, y_test, y_pred, y_prob, output_dir):
    report_path = os.path.join(output_dir, "training_report.pdf")
    with PdfPages(report_path) as pdf:
        # Accuracy and Loss plots
        plt.figure(figsize=(10, 4))
        plt.subplot(1, 2, 1)
        plt.plot(history.history['accuracy'], label='Train Accuracy')
        plt.plot(history.history['val_accuracy'], label='Val Accuracy')
        plt.title('Accuracy')
        plt.legend()

        plt.subplot(1, 2, 2)
        plt.plot(history.history['loss'], label='Train Loss')
        plt.plot(history.history['val_loss'], label='Val Loss')
        plt.title('Loss')
        plt.legend()
        plt.tight_layout()
        pdf.savefig()
        plt.close()

        # Confusion Matrix
        cm = confusion_matrix(y_test, y_pred)
        plt.figure()
        plt.imshow(cm, cmap='Blues')
        plt.title('Confusion Matrix')
        plt.colorbar()
        plt.xlabel('Predicted')
        plt.ylabel('True')
        for i in range(cm.shape[0]):
            for j in range(cm.shape[1]):
                plt.text(j, i, cm[i, j], ha='center', va='center', color='red')
        pdf.savefig()
        plt.close()

        # ROC Curve
        fpr, tpr, _ = roc_curve(y_test, y_prob)
        roc_auc = auc(fpr, tpr)
        plt.figure()
        plt.plot(fpr, tpr, color='darkorange', label=f'ROC curve (AUC = {roc_auc:.2f})')
        plt.plot([0, 1], [0, 1], color='navy', linestyle='--')
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('Receiver Operating Characteristic')
        plt.legend(loc="lower right")
        pdf.savefig()
        plt.close()

        # Classification Report
        report_text = classification_report(y_test, y_pred)
        plt.figure(figsize=(8, 5))
        plt.axis('off')
        plt.text(0.01, 0.6, report_text, fontsize=10, fontfamily='monospace')
        plt.title('Classification Report')
        pdf.savefig()
        plt.close()


def balance_samples(X, y):
    if len(X) == 0 or len(y) == 0:
        raise ValueError("Empty data for balancing")
    
    X = np.array(X)
    y = np.array(y)
    
    class0_idx = np.where(y == 0)[0]
    class1_idx = np.where(y == 1)[0]
    
    if len(class0_idx) == 0 or len(class1_idx) == 0:
        raise ValueError("One of the classes is missing in the data")
    
    max_samples = max(len(class0_idx), len(class1_idx))
    
    if len(class0_idx) < max_samples:
        resampled_idx = resample(class0_idx, replace=True, n_samples=max_samples, random_state=42)
        balanced_idx = np.concatenate([resampled_idx, class1_idx])
    else:
        resampled_idx = resample(class1_idx, replace=True, n_samples=max_samples, random_state=42)
        balanced_idx = np.concatenate([class0_idx, resampled_idx])
    
    np.random.shuffle(balanced_idx)
    return X[balanced_idx], y[balanced_idx]

def process_single_file(file_path, violin_class="violin"):
    file = os.path.basename(file_path)
    label = 1 if violin_class.lower() in file.lower() else 0
    print(f"Processing file: {file_path}, label: {label}")
    return process_audio_file(file_path, label)

def prepare_dataset(file_list, violin_class="violin", num_processes=None):
    if num_processes is None:
        num_processes = os.cpu_count()
    
    with Pool(num_processes) as pool:
        results = pool.map(partial(process_single_file, violin_class=violin_class), file_list)
    
    class0_segments = []
    class1_segments = []
    
    for segments in results:
        if segments and segments[0][1] == 1:
            class1_segments.extend(segments)
        else:
            class0_segments.extend(segments)
    
    print(f"Total files to process: {len(file_list)}")
    print(f"Class 0 segments found: {len(class0_segments)}")
    print(f"Class 1 segments found: {len(class1_segments)}")
    
    min_len = min(len(class0_segments), len(class1_segments))
    class0_segments = class0_segments[:min_len]
    class1_segments = class1_segments[:min_len]
    
    all_segments = class0_segments + class1_segments
    np.random.shuffle(all_segments)
    
    shapes = [s[0].shape for s in all_segments]
    if len(set(shapes)) > 1:
        print(f"Different spectrogram shapes detected: {set(shapes)}")
        min_time = min(s[0].shape[1] for s in all_segments)
        print(f"Cropping all to minimum time length: {min_time}")
        all_segments = [(s[0][:, :min_time], s[1]) for s in all_segments]
    
    try:
        X = np.array([s[0] for s in all_segments])
        y = np.array([s[1] for s in all_segments])
    except Exception as e:
        print("Error creating array:")
        print(f"Element shapes: {[s[0].shape for s in all_segments]}")
        raise
    
    return X[..., np.newaxis], y
    
def collect_files(folder):
    for root, _, files in os.walk(folder):
        for f in files:
            if f.endswith('.wav'):
                yield os.path.join(root, f)

def main():    
    output_dir = "output"
    
    print(f"Folder wav_folder_a: {os.path.abspath(wav_folder_a)}")
    print(f"Folder wav_folder_b: {os.path.abspath(wav_folder_b)}")
    
    train_files = list(collect_files(wav_folder_a))
    test_files = list(collect_files(wav_folder_b))

    if not train_files or not test_files:
        raise ValueError("No .wav files found in one of the folders")

    try:
        X_train, y_train = prepare_dataset(train_files)
    except Exception as e:
        raise
    
    try:
        X_test, y_test = prepare_dataset(test_files)
    except Exception as e:
        raise
    
    print(f"X_train size before balancing: {len(X_train)}")
    print(f"y_train size before balancing: {len(y_train)}")
    print(f"Classes in y_train: {np.unique(y_train, return_counts=True)}")
    
    try:
        X_train, y_train = balance_samples(X_train, y_train)
    except ValueError as e:
        raise
    
    X_train, X_val, y_train, y_val = train_test_split(
        X_train, y_train, test_size=0.25, random_state=42, stratify=y_train)
        
    os.makedirs(output_dir, exist_ok=True)
    save_path = os.path.join(output_dir, 'processed_data.joblib')
    joblib.dump({
        'X_train': X_train, 'y_train': y_train,
        'X_val': X_val, 'y_val': y_val,
        'X_test': X_test, 'y_test': y_test
    }, save_path)
    
    input_shape = X_train.shape[1:]
    model, history = train_model(X_train, y_train, X_val, y_val, input_shape)
    
    y_prob = model.predict(X_test).flatten()
    y_pred = (y_prob >= 0.5).astype(int)

    save_training_report(history, y_test, y_pred, y_prob, output_dir)

    # model_save_path = os.path.join(output_dir, "violin_model.h5")
    model.save("violin_model")

if __name__ == "__main__":
    main()