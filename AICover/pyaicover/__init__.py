import os, sys

flask_dir = os.getcwd() + "\\pyaicover"
model_dir = os.getcwd() + "\\pyaicover\\models"
sys.path.append(flask_dir)
sys.path.append(model_dir)

uvr_dir = os.getcwd() + "\\pyaicover\\uvr"
sys.path.append(uvr_dir)

from flask import Flask, render_template, jsonify, request
from .models import AICover
from .uvr.uvr import separate_process
from .uvr.gui_data.constants import *
import measurements as me
import sing

import requests
import shutil
import time
from pydub import AudioSegment
from datetime import datetime


app = Flask(__name__)

UPLOAD_FOLDER = f'{flask_dir}\\mesurements'
ALLOWED_EXTENSIONS = {'wav', 'mp3'}

DATABASE_PATH = f"{model_dir}\\uploads.db"

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

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

os.makedirs(flask_dir + "\\output_fft", exist_ok=True)

pit_diff_list = []

### Flask 서버
@app.route('/')
def index():
    return render_template('index.html')

### 사용자 음역대 추출
@app.route('/upload-high', methods=['POST'])
def upload_high():
    file = request.files['file']

    print(file.filename)
    os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

    filename = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
    file.save(filename)

    volume = AudioSegment.from_file(filename, format="mp4")
    volume = volume + 10
    volume.export(filename, format='wav')

    pitch = me.analyze_high_frequency(filename)
    note, octave = me.frequency_to_note_and_octave(pitch)

    return jsonify({"highestfrequency": str(pitch), "note": note, "octave": octave})


@app.route('/upload-low', methods=['POST'])
def upload_low():
    file = request.files['file']

    print(file.filename)
    os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

    filename = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
    file.save(filename)

    volume = AudioSegment.from_file(filename, format="mp4")
    volume = volume + 10
    volume.export(filename, format='wav')

    pitch = me.analyze_low_frequency(filename)
    note, octave = me.frequency_to_note_and_octave(pitch)

    return jsonify({"lowestfrequency": str(pitch), "note": note, "octave": octave})


@app.route('/get-recommendations', methods=['GET', 'POST'])
def get_recommendations():
    print("들어온 requests:", request.get_json())
    highest_freq = request.get_json()['high']
    print("highest_freq:", highest_freq)
    lowest_freq = request.get_json()['low']
    print("lowest_freq:", lowest_freq)
    filenames = me.get_matching_filenames(highest_freq, lowest_freq)
    print("추천 음악:", filenames)

    return filenames


### 노래방 기능
@app.route('/singasong', methods=['GET', 'POST'])
def sing_song():
    # 음원 분리
    print("음원 로딩 중입니다...")
    id = request.form['id']
    music = request.files['music']
    user = request.files['user']
    vofile_name, vofile_extension = sing.split_music(music)
    usfile_name, usfile_extension = os.path.splitext(user.filename)

    # 사용자 id를 기반으로 한 평가 폴더 생성
    os.makedirs(f"{flask_dir}\\output_fft\\{id}", exist_ok=True)
    user_path = f"{flask_dir}\\output_fft\\{id}\\{usfile_name}{usfile_extension}"
    user.save(user_path)
    vocal_path = f"{os.getcwd()}\\pyaicover\\output\\{vofile_name}\\Vocals_{vofile_name}{vofile_extension}"
    
    print("저장 완료!")

    ref_samples, sr = sing.load_audio_file(vocal_path)
    ref_segments = sing.find_audio_segments(ref_samples, sr)
    
    # user가 부른 노래
    user_samples, sr = sing.load_audio_file(user_path)
    user_segments = sing.find_audio_segments(user_samples, sr)

    print("집계 중...")

    current_time = datetime.now().strftime("%Y%m%d%H%M%S")
    answer_path = os.path.join(f"{flask_dir}\\output_fft\\{id}", f"{current_time}_reference.txt")

    # 답지 노래 txt로 저장
    with open(answer_path, "w") as ref_f:
        for i, (ref_start, ref_end) in enumerate(ref_segments):
            ref_pitch = sing.analyze_pitch_fft(ref_samples[ref_start:ref_end], sr)
            ref_note = sing.pitch_to_note(ref_pitch)
            
            ref_start_str = sing.samples_to_time_str(ref_start, sr)
            ref_end_str = sing.samples_to_time_str(ref_end, sr)
            
            ref_f.write(f"Segment {i+1}: Ref Start {ref_start_str}, Ref End {ref_end_str}, Ref Note {ref_note}\n")


    total_score = 0
    note_count = 0

    score_path = os.path.join(f"{flask_dir}\\output_fft\\{id}", f"{current_time}_user_evaluation.txt")
    
    with open(score_path, "w") as f:
        for i, (ref_start, ref_end) in enumerate(ref_segments):
            ref_pitch = sing.analyze_pitch_fft(ref_samples[ref_start:ref_end], sr)
            ref_note = sing.pitch_to_note(ref_pitch)
            
            for user_start, user_end in user_segments:
                rhythm_evaluation, rhythm_score = sing.evaluate_rhythm(ref_start, ref_end, user_start, user_end)
                
                user_pitch = sing.analyze_pitch_fft(user_samples[ref_start:ref_end], sr)
                user_note = sing.pitch_to_note(user_pitch)
                pitch_evaluation, pitch_score = sing.evaluate_pitch_notes(ref_note, user_note)
                ref_start_str = sing.samples_to_time_str(ref_start, sr)
                ref_end_str = sing.samples_to_time_str(ref_end, sr)
                user_start_str = sing.samples_to_time_str(user_start, sr)
                user_end_str = sing.samples_to_time_str(user_end, sr)
                
                # Remove User Start, User End
                f.write(f"Segment {i+1}: Ref Start {ref_start_str}, Ref End {ref_end_str}, Ref Note {ref_note}, User Note {user_note}, Pitch: {pitch_evaluation}, Pitch Score: {pitch_score}\n")
                
                total_score += pitch_score
                note_count += 1
                break

        final_score = total_score / note_count if note_count > 0 else 0
        f.write(f"\nFinal Score: {final_score:.2f}/100")

    print(f"Final Score: {final_score:.2f}")

    return f'{final_score:.2f}'


### AI Cover 제작
@app.route('/aicover', methods=['GET', 'POST'])
def create_aicover():
    starttime = time.time()
    id = request.form['id']
    music = request.files['music']
    model_name = request.form['model']
    sid = model_name + ".pth"  # 절대 경로 입력하지말고 weights 폴더 안의 파일 이름만 입력할 것
    vc_transform = int(request.form['octave'])  # 옥타브 설정(-12 ~ 12)
    index_rate = float(request.form['index'])  # index 파일 참조 비율 : 높을수록 모델쪽의 발음/음정, 낮을수록 기존 보컬쪽의 발음/음정 중시
    mode = 'pred'

    print("ID:", id)
    print("모델:", sid)
    print("옥타브:", vc_transform)
    print("참조율:", index_rate)

    # 받은 음악 파일 저장 후 음원 분리
    print(music.filename)
    file_name, file_extension = os.path.splitext(music.filename)
    nsfile_name = file_name.replace(' ', '_')
    os.makedirs(uvr_dir + "\\pred", exist_ok=True)
    
    music_path = f'{uvr_dir}\\pred\\{nsfile_name}{file_extension}'
    music.save(music_path)

    # 음원 분리
    # VocFT와 KIM Vocal 1을 앙상블을 통해 나온 결과물 폴더 (Vocal, Inst)
    export_path1 = f'{uvr_dir}\\outputs1'
    os.makedirs(export_path1, exist_ok=True)

    # Karaoke 2를 통해 나온 결과물 폴더 (Vocal)
    export_path2 = f'{uvr_dir}\\outputs2'
    os.makedirs(export_path2, exist_ok=True)

    # Reverb HQ를 통해 나온 최종 결과물 폴더 (Vocal)
    export_result = f'{uvr_dir}\\outputs3'
    os.makedirs(export_result, exist_ok=True)

    input_path = (music_path,)
    print("1번째 추론을 시작합니다.")
    separate_process(mode, ENSEMBLE_MODE, input_path, export_path1)

    get_file_name = input_path[0].split('\\')[-1]
    get_file_name = get_file_name.split('.')[0]

    input_path2 = (f"{export_path1}\\1_{get_file_name}_(Vocals).wav",)
    model_var = 'UVR-MDX-NET Karaoke 2'
    print("2번째 추론을 시작합니다.")
    separate_process(mode, MDX_ARCH_TYPE, input_path2, export_path2, model_var)

    input_path3 = (f"{export_path2}\\1_1_{get_file_name}_(Vocals)_(Vocals).wav",)
    model_var = 'Reverb HQ'
    print("마지막 추론을 시작합니다.")
    separate_process(mode, MDX_ARCH_TYPE, input_path3, export_result, model_var)

    # 만약 Other가 있다면 삭제
    if os.path.isfile(f"{export_result}\\1_1_1_{get_file_name}_(Vocals)_(Vocals)_(Other).wav"):
        os.remove(f"{export_result}\\1_1_1_{get_file_name}_(Vocals)_(Vocals)_(Other).wav")

    # 최종 보컬과 MR의 이름 바꾸기
    os.rename(f"{export_result}\\1_1_1_{get_file_name}_(Vocals)_(Vocals)_(No Other).wav", f"{export_result}\\Vocals_{get_file_name}.wav")
    os.rename(f"{export_path1}\\1_{get_file_name}_(Instrumental).wav", f"{export_path1}\\Instruments_{get_file_name}.wav",)

    os.makedirs(f"{uvr_dir}\\pred\\{get_file_name}", exist_ok=True)
    shutil.move(f"{export_result}\\Vocals_{get_file_name}.wav", f"{uvr_dir}\\pred\\{get_file_name}\\Vocals_{get_file_name}.wav")
    shutil.move(f"{export_path1}\\Instruments_{get_file_name}.wav", f"{uvr_dir}\\pred\\{get_file_name}\\Instruments_{get_file_name}.wav")

    # AI Cover 생성
    input_audio = f"{uvr_dir}\\pred\\{get_file_name}\\Vocals_{get_file_name}.wav"
    background_audio = f"{uvr_dir}\\pred\\{get_file_name}\\Instruments_{get_file_name}.wav"
    file_index = f"./AICover/pyaicover/models/logs/{model_name}/added_IVF1203_Flat_nprobe_1_{model_name}_v2.index"

    result_path = AICover.make_AICover(sid, vc_transform, input_audio, file_index, index_rate, background_audio)

    # 서버로 완성된 AI Cover 전송
    print("전송을 시도합니다.")
    aicover = open(result_path, 'rb')
    upload = {'file': aicover, 'id':id}
    requests.post('http://172.20.10.12:8080/api/created-song', files=upload)
    print("전송 완료")
    endtime = time.time()

    return f"AI Cover 생성이 완료되었습니다. 소요시간:{endtime - starttime}초"