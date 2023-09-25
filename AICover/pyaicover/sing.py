import os, sys

flask_dir = os.getcwd() + "\\pyaicover"
model_dir = os.getcwd() + "\\pyaicover\\models"
sys.path.append(flask_dir)
sys.path.append(model_dir)

import sounddevice as sd
import soundfile as sf
import numpy as np
import librosa
import pygame
from pydub import AudioSegment
import shutil


SEMITONES = ['C0', 'C#0', 'D0', 'D#0', 'E0', 'F0', 'F#0', 'G0', 'G#0', 'A0', 'A#0', 'B0',
            'C1', 'C#1', 'D1', 'D#1', 'E1', 'F1', 'F#1', 'G1', 'G#1', 'A1', 'A#1', 'B1',
            'C2', 'C#2', 'D2', 'D#2', 'E2', 'F2', 'F#2', 'G2', 'G#2', 'A2', 'A#2', 'B2',
            'C3', 'C#3', 'D3', 'D#3', 'E3', 'F3', 'F#3', 'G3', 'G#3', 'A3', 'A#3', 'B3',
            'C4', 'C#4', 'D4', 'D#4', 'E4', 'F4', 'F#4', 'G4', 'G#4', 'A4', 'A#4', 'B4',
            'C5', 'C#5', 'D5', 'D#5', 'E5', 'F5', 'F#5', 'G5', 'G#5', 'A5', 'A#5', 'B5',
            'C6', 'C#6', 'D6', 'D#6', 'E6', 'F6', 'F#6', 'G6', 'G#6', 'A6', 'A#6', 'B6',
            'C7', 'C#7', 'D7', 'D#7', 'E7', 'F7', 'F#7', 'G7', 'G#7', 'A7', 'A#7', 'B7',
            'C8', 'C#8', 'D8', 'D#8', 'E8', 'F8', 'F#8', 'G8']

def split_music(music):
    file_name, file_extension = os.path.splitext(music.filename)
    nsfile_name = file_name.replace(' ', '_')
    os.makedirs(model_dir + "\\spleeter", exist_ok=True)
    
    music_path = f'{model_dir}\\spleeter\\{nsfile_name}{file_extension}'
    music.save(music_path)

    # 음원 분리
    spl = f'spleeter separate -p spleeter:5stems -o pyaicover/models/spleeter {model_dir}\\spleeter\\{nsfile_name}{file_extension}'
    os.system(spl)

    # 서버에서 받은 음원 삭제
    if os.path.isfile(music_path):
        os.remove(music_path)

    background_path = f"{model_dir}/spleeter/{nsfile_name}/"

    vocals = AudioSegment.from_wav(background_path + "vocals.wav")
    bass = AudioSegment.from_wav(background_path + "bass.wav")
    drums = AudioSegment.from_wav(background_path + "drums.wav")
    other = AudioSegment.from_wav(background_path + "other.wav")
    piano = AudioSegment.from_wav(background_path + "piano.wav")

    r1 = bass.overlay(drums, position=0)
    r2 = r1.overlay(other, position=0)
    result_MR = r2.overlay(piano, position=0)

    os.makedirs(f"{os.getcwd()}\\pyaicover\\output\\{nsfile_name}", exist_ok=True)
    vocals.export(f"{os.getcwd()}\\pyaicover\\output\\{nsfile_name}\\Vocals_{nsfile_name}{file_extension}", format="mp3")
    result_MR.export(f"{os.getcwd()}\\pyaicover\\output\\{nsfile_name}\\MR_{nsfile_name}{file_extension}", format="mp3")
    print("MR 준비 완료")

    # 사용된 음원 삭제
    shutil.rmtree(f"{model_dir}\\spleeter\\{nsfile_name}", ignore_errors=True)

    return nsfile_name, file_extension

def pitch_to_note(pitch, base_frequency=440.0):
    """주어진 피치를 서양 음악의 12음도에 해당하는 음계와 옥타브로 변환"""
    if pitch == 0.0:
        return "None"
    index = round(12 * np.log2(pitch / base_frequency))
    
    # index 값이 음수일 때의 처리
    while index < 0:
        index += 12  # 12를 더해 음계를 한 옥타브 위로 올림
    
    octave = index // 12
    note = SEMITONES[index % 12]
    return f"{note[:-1]}{octave}"


def load_audio_file(file_path):
    audio, sr = librosa.load(file_path, sr=None)
    return audio, sr


def find_audio_segments(samples, sr, threshold=0.01, min_duration=0.1):
    audio_segments = []
    is_audio = False
    start_sample = 0
    step_size = int(sr * min_duration)
    for i in range(0, len(samples) - step_size, step_size):
        segment = samples[i:i + step_size]
        segment_energy = np.mean(np.abs(segment))
        if segment_energy > threshold:
            if not is_audio:
                is_audio = True
                start_sample = i
            end_sample = i + step_size
            audio_segments.append((start_sample, end_sample))
            is_audio = False

    return audio_segments


def play_audio(file_path):
    pygame.mixer.pre_init(44100, 16, 2, 4096)
    pygame.mixer.init()
    pygame.mixer.music.load(file_path)
    pygame.mixer.music.play()


def record_audio(duration, sr):
    print("Recording...")
    audio_data = sd.rec(int(duration * sr), samplerate=sr, channels=1, dtype=np.float32)
    sd.wait()
    print("Recording complete.")
    return audio_data.flatten()


def analyze_pitch_fft(samples, sr):
    n = len(samples)
    freqs = np.fft.fftfreq(n, 1/sr)
    spectrum = np.fft.fft(samples)
    magnitude = np.abs(spectrum)
    
    # 주파수가 양수인 부분만 선택
    positive_freqs = freqs[freqs >= 0]
    positive_magnitude = magnitude[freqs >= 0]
    
    # 가장 높은 진폭을 가진 주파수를 찾음 (피치로 간주)
    pitch = positive_freqs[np.argmax(positive_magnitude)]
    
    return pitch


def save_audio(samples, sr, file_path):
    sf.write(file_path, samples, sr)


def evaluate_pitch_notes(reference, user):
    try:
        pitch_difference = SEMITONES.index(user) - SEMITONES.index(reference)
        
        if pitch_difference == 0:
            return "Perfect", 100
        elif abs(pitch_difference) <= 1:
            return "Good", 80
        elif abs(pitch_difference) <= 3:
            return "Normal", 60
        else:
            return "Bad", 40
    except:
        return "None", 0

def evaluate_rhythm(ref_start, ref_end, user_start, user_end):
    overlap_start = max(ref_start, user_start)
    overlap_end = min(ref_end, user_end)
    overlap_duration = max(0, overlap_end - overlap_start)
    
    ref_duration = ref_end - ref_start
    overlap_ratio = overlap_duration / ref_duration
    
    if overlap_ratio >= 0.9:
        return "In-Sync", 100
    elif overlap_ratio >= 0.4:
        return "Almost In-Sync", 70
    else:
        return "Out-of-Sync", 0

def samples_to_time_str(samples_index, sr):
    """샘플 인덱스를 MM:SS.mmm 형식의 문자열로 변환"""
    seconds_total = samples_index / sr
    minutes = int(seconds_total // 60)
    seconds = int(seconds_total % 60)
    milliseconds = int((seconds_total * 1000) % 1000)
    return f"{minutes:02}:{seconds:02}.{milliseconds:03}"
