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


def split_music(music):
    file_name, file_extension = os.path.splitext(music.filename)
    nsfile_name = file_name.replace(' ', '_')
    os.makedirs(model_dir + "\\spleeter", exist_ok=True)

    try:
        os.rename(os.path.join(model_dir + "\\spleeter\\", file_name + file_extension), os.path.join(model_dir + "\\spleeter\\", nsfile_name + file_extension))
    except FileNotFoundError:
        pass
    
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


### 음정 채점 기준 변경 필요 ###
def evaluate_pitch_fft(reference, user):
    # 평균율 pitch에서 한 음과 다음 음 사이의 주파수 비율
    semitone_ratio = 2 ** (1 / 12)
    
    # 반음 위 아래로의 주파수 범위
    upper_semitone = reference * semitone_ratio
    lower_semitone = reference / semitone_ratio
    
    # 한 음 위 아래로의 주파수 범위
    upper_whole_tone = reference * (semitone_ratio ** 2)
    lower_whole_tone = reference / (semitone_ratio ** 2)
    
    if abs(reference - user) < 10:  # 10Hz 허용 오차
        return "Perfect", 100
    elif lower_semitone < user < upper_semitone:
        return "Good", 80
    elif lower_whole_tone < user < upper_whole_tone:
        return "Normal", 60
    else:
        return "Bad", 40


### 박자 채점 변경 필요 ###
def evaluate_rhythm(ref_start, ref_end, user_start, user_end):
    overlap_start = max(ref_start, user_start)
    overlap_end = min(ref_end, user_end)
    overlap_duration = max(0, overlap_end - overlap_start)
    
    ref_duration = ref_end - ref_start
    overlap_ratio = overlap_duration / ref_duration
    
    if overlap_ratio >= 0.9:  # 70% 이상 겹친다면
        return "In-Sync", 100
    elif overlap_ratio >= 0.4:  # 40% 이상 겹친다면
        return "Almost In-Sync", 70
    else:
        return "Out-of-Sync", 0