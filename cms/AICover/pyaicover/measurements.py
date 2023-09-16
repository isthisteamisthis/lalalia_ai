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
    "라1": 55.00, "라#1": 58.27, "시1": 61.74,
    "도2": 65.41, "도#2": 69.30, "레2": 73.42, "레#2": 77.78, "미2": 82.41, "파2": 87.31,
    "파#2": 92.50, "솔2": 98.00, "솔#2": 103.83, "라2": 110.00, "라#2": 116.54, "시2": 123.47,
    "도3": 130.81, "도#3": 138.59, "레3": 146.83, "레#3": 155.56, "미3": 164.81, "파3": 174.61,
    "파#3": 185.00, "솔3": 196.00, "솔#3": 207.65, "라3": 220.00, "라#3": 233.08, "시3": 246.94,
    "도4": 261.63, "도#4": 277.18, 
}

# 사람이 낼 수 있는 최저Hz와 최고Hz
MIN_HUMAN_FREQUENCY = 60
MAX_HUMAN_FREQUENCY = 270

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

def get_matching_filenames(highest_freq, lowest_freq):
    conn = sqlite3.connect(DATABASE_PATH)
    cursor = conn.cursor()

    # 사용자의 highest_freq가 DB의 highest_freq보다 작거나 같고,
    # 사용자의 lowest_freq가 DB의 lowest_freq보다 크거나 같은 파일 이름 조회
    cursor.execute("""
    SELECT filename FROM uploads 
    WHERE ? <= highst_freq AND ? >= lowest_freq
    LIMIT 3
    """, (highest_freq, lowest_freq))

    filenames = [row[0].replace('.wav', '') for row in cursor.fetchall()]  # .wav 확장자 제거

    # 만약 결과가 없다면 임의로 3개의 파일 이름을 선택
    if not filenames:
        cursor.execute("SELECT filename FROM uploads LIMIT 3")
        filenames = [row[0].replace('.wav', '') for row in cursor.fetchall()]  # .wav 확장자 제거

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