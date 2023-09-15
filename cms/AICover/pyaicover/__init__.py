import os, sys

model_dir = os.getcwd() + "\\pyaicover\\models"
sys.path.append(model_dir)

from flask import Flask, render_template, jsonify, request
from .models import AICover
import requests
import shutil
import time
import librosa
from pydub import AudioSegment
import numpy as np
import soundfile as sf

app = Flask(__name__)

UPLOAD_FOLDER = 'uploads'
ALLOWED_EXTENSIONS = {'wav', 'mp3'}

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

NOTE_FREQUENCIES = {
    "라1": 55.00, "라#1": 58.27, "시1": 61.74,
    "도2": 65.41, "도#2": 69.30, "레2": 73.42, "레#2": 77.78, "미2": 82.41, "파2": 87.31,
    "파#2": 92.50, "솔2": 98.00, "솔#2": 103.83, "라2": 110.00, "라#2": 116.54, "시2": 123.47,
    "도3": 130.81, "도#3": 138.59, "레3": 146.83, "레#3": 155.56, "미3": 164.81, "파3": 174.61,
    "파#3": 185.00, "솔3": 196.00, "솔#3": 207.65, "라3": 220.00, "라#3": 233.08, "시3": 246.94,
    "도4": 261.63, "도#4": 277.18, 
}

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
    D = librosa.stft(y)
    frequencies = librosa.fft_frequencies(sr=sr)
    magnitude = np.abs(D)

    human_voice_indices = np.where((frequencies >= MIN_HUMAN_FREQUENCY) & (frequencies <= MAX_HUMAN_FREQUENCY))
    filtered_magnitudes = magnitude[human_voice_indices]

    highest_index = np.argmax(np.max(filtered_magnitudes, axis=1))
    highest_freq = round(frequencies[human_voice_indices][highest_index], 2)
    return str(highest_freq)

def analyze_low_frequency(file_path):
    y, sr = librosa.load(file_path)
    D = librosa.stft(y)
    frequencies = librosa.fft_frequencies(sr=sr)
    magnitude = np.abs(D)

    human_voice_indices = np.where((frequencies >= MIN_HUMAN_FREQUENCY) & (frequencies <= MAX_HUMAN_FREQUENCY))
    filtered_magnitudes = magnitude[human_voice_indices]

    lowest_index = np.argmin(np.min(filtered_magnitudes, axis=1) + np.max(filtered_magnitudes))
    lowest_freq = round(frequencies[human_voice_indices][lowest_index], 2)
    return str(lowest_freq)


@app.route('/')
def index():
    return render_template('index.html')

def amplify_and_save(file_path):
    y, sr = librosa.load(file_path, sr=None)
    y = y * 2.0
    # Avoid clipping
    y = np.clip(y, -1.0, 1.0)
    sf.write(file_path, y, sr)

@app.route('/upload-high', methods=['POST'])
def upload_high():
    file = request.files['file']

    print(file.filename)
    file_name, file_extension = os.path.splitext(file.filename)
    print("파일 이름:", file_name)
    print("파일 확장자:", file_extension)
    
    nsfile_name = file_name.replace(' ', '_')
    os.makedirs(model_dir + "\\measure", exist_ok=True)
    print("폴더 생성")

    measure_path = f'{model_dir}\\measure\\{nsfile_name}{file_extension}'
    file.save(measure_path)
    print("저장 성공")

    volume = AudioSegment.from_file(measure_path, format="mp4")
    volume = volume + 10
    volume.export(measure_path, format='wav')
        
    print("불러오기")
    amplify_and_save(measure_path)
        
    pitch = analyze_high_frequency(measure_path)
    note, octave = frequency_to_note_and_octave(pitch)
    return jsonify({"highestfrequency": str(pitch), "note": note, "octave": octave})

@app.route('/upload-low', methods=['POST'])
def upload_low():
    file = request.files['file']

    print(file.filename)
    file_name, file_extension = os.path.splitext(file.filename)
    print("파일 이름:", file_name)
    print("파일 확장자:", file_extension)
    
    nsfile_name = file_name.replace(' ', '_')
    os.makedirs(model_dir + "\\measure", exist_ok=True)
    print("폴더 생성")

    measure_path = f'{model_dir}\\measure\\{nsfile_name}{file_extension}'
    file.save(measure_path)
    print("저장 성공")

    volume = AudioSegment.from_file(measure_path, format="mp4")
    volume = volume + 10
    volume.export(measure_path, format='wav')
        
    print("불러오기")
    amplify_and_save(measure_path)
        
    pitch = analyze_low_frequency(measure_path)
    note, octave = frequency_to_note_and_octave(pitch)
    return jsonify({"lowestfrequency": str(pitch), "note": note, "octave": octave})


@app.route('/aicover', methods=['GET', 'POST'])
def create_aicover():
    starttime = time.time()
    music = request.files['music']
    model_name = request.form['model']
    sid = model_name + ".pth"  # 절대 경로 입력하지말고 weights 폴더 안의 파일 이름만 입력할 것
    vc_transform = int(request.form['octave'])  # 옥타브 설정(-12 ~ 12)
    index_rate = float(request.form['index'])  # index 파일 참조 비율 : 높을수록 모델쪽의 발음/음정, 낮을수록 기존 보컬쪽의 발음/음정 중시

    print("모델:", sid)
    print("옥타브:", vc_transform)
    print("참조율:", index_rate)

    # 받은 음악 파일 저장
    print(music.filename)
    file_name, file_extension = os.path.splitext(music.filename)
    print("파일 이름:", file_name)
    print("파일 확장자:", file_extension)
    
    nsfile_name = file_name.replace(' ', '_')
    os.makedirs(model_dir + "\\spleeter", exist_ok=True)

    try:
        os.rename(os.path.join(model_dir + "\\spleeter\\", file_name + file_extension), os.path.join(model_dir + "\\spleeter\\", nsfile_name + file_extension))
    except FileNotFoundError:
        pass
    
    music_path = f'{model_dir}\\spleeter\\{nsfile_name}{file_extension}'
    music.save(music_path)

    print('기다려주세요.')

    # 음원 분리
    spl = f'spleeter separate -p spleeter:5stems -o pyaicover/models/spleeter {model_dir}\\spleeter\\{nsfile_name}{file_extension}'
    os.system(spl)

    # 서버에서 받은 음원 삭제
    if os.path.isfile(music_path):
        os.remove(music_path)

    print("분리가 완료되었습니다.")
    print("폴더명:", nsfile_name)
    print("분리된 음원의 경로:", f"{model_dir}\\spleeter\\{nsfile_name}")

    # AI Cover 생성
    input_audio = f"{model_dir}/spleeter/{nsfile_name}/vocals.wav"
    file_index2 = f"./AICover/pyaicover/models/logs/{model_name}/added_IVF1203_Flat_nprobe_1_{model_name}_v2.index"
    background_path = f"{model_dir}/spleeter/{nsfile_name}/"

    file_name = AICover.make_AICover(sid, vc_transform, input_audio, file_index2, index_rate, background_path)
    
    # 사용된 음원 삭제
    # shutil.rmtree(f"{model_dir}\\spleeter\\{nsfile_name}", ignore_errors=True)

    # 서버로 완성된 AI Cover 전송
    print("전송을 시도합니다.")
    aicover = open(f"./pyaicover/models/merged_results/{input_audio.split('/')[-2]}_{model_name}.mp3", 'rb')
    upload = {'perfect-score': aicover}
    requests.post(' http://192.168.0.88:8080/api/perfect-scores', files = upload)
    print("전송 완료")
    endtime = time.time()

    return f"AI Cover가 성공적으로 제작되었습니다. 소요시간 : {endtime-starttime}" 


@app.route('/aitrain', methods=['GET', 'POST'])
def train_models():
    len = int(request.form['file_length'])

    musics = ['music' + str(i) for i in range(len)]

    for i in range(len):
        globals()[f"music{i}"] = request.files[musics[i]]
        print(globals()[f'music{i}'].filename)

        try:
            file_name, file_extension = os.path.splitext(globals()[f'music{i}'].filename)
            print("파일 이름:", file_name)
            print("파일 확장자:", file_extension)
        
            nsfile_name = file_name.replace(' ', '_')
            os.makedirs(model_dir + "\\spleeter", exist_ok=True)

            try:
                os.rename(os.path.join(model_dir + "\\spleeter\\", file_name + file_extension), os.path.join(model_dir + "\\spleeter\\", nsfile_name + file_extension))
            except FileNotFoundError:
                pass
            
            music_path = f'{model_dir}\\spleeter\\{nsfile_name}{file_extension}'
            globals()[f'music{i}'].save(music_path)

            print('기다려주세요.')

            # 음원 분리
            spl = f'spleeter separate -p spleeter:5stems -o pyaicover/models/spleeter {model_dir}\\spleeter\\{nsfile_name}{file_extension}'
            os.system(spl)

            # 서버에서 받은 음원 삭제
            if os.path.isfile(music_path):
                os.remove(music_path)

            print("분리가 완료되었습니다.")
            print("폴더명:", nsfile_name)
            print("분리된 음원의 경로:", f"{model_dir}\\spleeter\\{nsfile_name}")

            os.makedirs(model_dir + "\\datasets", exist_ok=True)
            shutil.move(f"{model_dir}\\spleeter\\{nsfile_name}\\vocals.wav", f"{model_dir}\\datasets")
            
            os.rename(f"{model_dir}\\datasets\\vocals.wav", f"{model_dir}\\datasets\\Vocals_{nsfile_name}.wav")
            shutil.rmtree(f"{model_dir}\\spleeter\\{nsfile_name}", ignore_errors=True)

        except:
            return "오류가 발생했습니다."
        

    return "ok"
        

        # AI Cover 생성
        # train_datasets = f"{model_dir}\\datasets"
        # train_name = request.form['model']  # 모델의 이름
        # total_epoch11 = request.form['epochs']  # epoch 수

        # pretrained_G14 = "./AICover/pyaicover/models/pretrained_v2/f0G48k.pth"
        # pretrained_D15 = "./AICover/pyaicover/models/pretrained_v2/f0D48k.pth"

        # return f"{train_name} 모델을 서버로 전송해 훈련을 시작합니다."
        # train_model.train1key(train_name, trainset_dir4, total_epoch11, pretrained_G14, pretrained_D15)