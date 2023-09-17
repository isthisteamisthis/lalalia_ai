import os, sys

flask_dir = os.getcwd() + "\\pyaicover"
model_dir = os.getcwd() + "\\pyaicover\\models"
sys.path.append(flask_dir)
sys.path.append(model_dir)

import librosa
import numpy as np
import sqlite3

UPLOAD_FOLDER = f"{os.getcwd()}\\pyaicover\\mesurements"
ALLOWED_EXTENSIONS = {'wav', 'mp3'}

DATABASE_PATH = f"{os.getcwd()}\\pyaicover\\mesurements\\uploads.db"

NOTE_FREQUENCIES = {
    "도1": 80.00, "도#1": 84.85, "레1": 89.90, "레#1": 95.17, "미1": 100.67, "파1": 106.42, "파#1": 112.44, "솔1": 118.75, "솔#1": 125.37, "라1": 132.29, "라#1": 139.54, "시1": 147.16,
    "도2": 160.00, "도#2": 169.71, "레2": 179.81, "레#2": 190.34, "미2": 201.34, "파2": 212.83, "파#2": 224.88, "솔2": 237.50, "솔#2": 250.74, "라2": 264.58, "라#2": 279.08, "시2": 294.33,
    "도3": 320.00, "도#3": 339.42, "레3": 359.61, "레#3": 380.68, "미3": 402.68, "파3": 425.65, "파#3": 449.76, "솔3": 475.00, "솔#3": 501.48, "라3": 529.16, "라#3": 558.16, "시3": 588.66,
    "도4": 640.00, "도#4": 678.83, "레4": 719.23, "레#4": 761.36, "미4": 805.36, "파4": 851.31, "파#4": 899.52, "솔4": 950.00, "솔#4": 1002.96, "라4": 1058.32, "라#4": 1116.32, "시4": 1177.32,
    "도5": 1280.00
}

# 사람이 낼 수 있는 최저Hz와 최고Hz
MIN_HUMAN_FREQUENCY = 80
MAX_HUMAN_FREQUENCY = 1280

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def frequency_to_note_and_octave(frequency):
    # 문자열로 받은 frequency를 float로 변환
    frequency = float(frequency)

    # 주파수가 사람의 음성 범위를 벗어나면 범위 내에서 가장 가까운 주파수로 조정
    if frequency < MIN_HUMAN_FREQUENCY:
        frequency = MIN_HUMAN_FREQUENCY
    elif frequency > MAX_HUMAN_FREQUENCY:
        frequency = MAX_HUMAN_FREQUENCY

    closest_note = min(NOTE_FREQUENCIES.keys(), key=lambda note: abs(frequency - NOTE_FREQUENCIES[note]))
    note_name = closest_note[:-1]
    octave = closest_note[-1]
    return note_name, octave

def overlap_length(a_start, a_end, b_start, b_end):
    """
    두 범위의 겹치는 부분의 길이를 반환합니다.
    """
    overlap_start = max(a_start, b_start)
    overlap_end = min(a_end, b_end)
    return max(0, overlap_end - overlap_start)

def get_matching_filenames(highest_freq, lowest_freq):
    conn = sqlite3.connect(DATABASE_PATH)
    cursor = conn.cursor()

    # highest_freq와 lowest_freq 사이에 있는 파일 이름 조회
    cursor.execute("""
    SELECT filename, highest_freq, lowest_freq FROM uploads
    """)

    overlaps = []
    for row in cursor.fetchall():
        filename, file_high, file_low = row
        overlap = overlap_length(lowest_freq, highest_freq, file_low, file_high)
        overlaps.append((filename, overlap))

    # 겹치는 길이가 큰 순서대로 정렬
    sorted_overlaps = sorted(overlaps, key=lambda x: x[1], reverse=True)

    # 상위 3개의 파일 이름만 추출
    filenames = [item[0] for item in sorted_overlaps[:3]]

    conn.close()
    return filenames

def get_freq_from_audio(audio_data, rate):
    w = np.fft.fft(audio_data)
    freqs = np.fft.fftfreq(len(w))
    peak = np.argmax(np.abs(w))
    
    # Ensure that peak is within the range of freqs
    if peak >= len(freqs):
        peak = len(freqs) - 1

    freq = freqs[peak]
    freq_in_hertz = abs(freq * rate)
    
    # 주파수가 설정된 범위를 벗어나면 최소 또는 최대 값으로 설정
    if freq_in_hertz < MIN_HUMAN_FREQUENCY:
        freq_in_hertz = MIN_HUMAN_FREQUENCY
    elif freq_in_hertz > MAX_HUMAN_FREQUENCY:
        freq_in_hertz = MAX_HUMAN_FREQUENCY

    freq_in_hertz = round(freq_in_hertz, 2)

    return freq_in_hertz

def analyze_high_frequency(file_path):
    y, sr = librosa.load(file_path)

    # Pitch tracking using librosa
    pitches, magnitudes = librosa.piptrack(y=y, sr=sr)

    # Filter pitches within the human voice range
    valid_indices = (pitches >= MIN_HUMAN_FREQUENCY) & (pitches <= MAX_HUMAN_FREQUENCY)
    valid_pitches = pitches[valid_indices]
    valid_magnitudes = magnitudes[valid_indices]

    # Extract the pitch with the highest magnitude
    highest_freq = valid_pitches[np.argmax(valid_magnitudes)]
    highest_freq = round(highest_freq, 2)
    return str(highest_freq)

def analyze_low_frequency(file_path):
    y, sr = librosa.load(file_path, sr=None)
    
    # Pitch tracking using librosa
    pitches, magnitudes = librosa.piptrack(y=y, sr=sr)
    
    # Filter pitches within the human voice range
    valid_indices = (pitches >= MIN_HUMAN_FREQUENCY) & (pitches <= MAX_HUMAN_FREQUENCY)
    valid_pitches = pitches[valid_indices]
    valid_magnitudes = magnitudes[valid_indices]
    
    # Extract the pitch with the highest magnitude
    significant_freq_index = np.argmax(valid_magnitudes)
    
    # Extract the pitch with the lowest frequency among pitches with significant magnitudes
    lowest_freq = np.min(valid_pitches[valid_magnitudes > valid_magnitudes[significant_freq_index] * 0.75])
    
    lowest_freq = round(lowest_freq, 2)
    return str(lowest_freq)

def get_matching_filenames(highest_freq, lowest_freq):
    conn = sqlite3.connect(DATABASE_PATH)
    cursor = conn.cursor()

    # highest_freq와 lowest_freq 사이에 있는 파일 이름 조회
    cursor.execute("""
    SELECT filename FROM uploads 
    WHERE highest_freq <= ? AND lowest_freq >= ?
    LIMIT 3
    """, (highest_freq, lowest_freq))

    filenames = [row[0] for row in cursor.fetchall()]

    # 만약 결과가 3개 미만이면 부족한 수만큼 파일 이름을 임의로 추가
    if len(filenames) < 3:
        remaining = 3 - len(filenames)
        placeholders = ', '.join('?' * len(filenames))
        query = f"SELECT filename FROM uploads WHERE filename NOT IN ({placeholders}) LIMIT ?"
        cursor.execute(query, (*filenames, remaining))
        additional_filenames = [row[0] for row in cursor.fetchall()]
        filenames.extend(additional_filenames)

    conn.close()

    return filenames