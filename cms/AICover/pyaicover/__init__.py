import os, sys

flask_dir = os.getcwd() + "\\pyaicover"
model_dir = os.getcwd() + "\\pyaicover\\models"
sys.path.append(flask_dir)
sys.path.append(model_dir)

from flask import Flask, render_template, jsonify, request
from .models import AICover
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

os.makedirs(f"{os.getcwd()}\\pyaicover\\output_fft", exist_ok=True)

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
    music = request.files['music']
    nsfile_name, file_extension = sing.split_music(music)

    vocal_path = f"{os.getcwd()}\\pyaicover\\output\\{nsfile_name}\\Vocals_{nsfile_name}{file_extension}"
    MR_path = f"{os.getcwd()}\\pyaicover\\output\\{nsfile_name}\\MR_{nsfile_name}{file_extension}"

    ref_samples, sr = sing.load_audio_file(vocal_path)
    ref_segments = sing.find_audio_segments(ref_samples, sr)

    print("음원 재생을 시작합니다.")
    sing.play_audio(MR_path)

    user_samples = sing.record_audio(len(ref_samples) / sr, sr)
    user_segments = sing.find_audio_segments(user_samples, sr)
    print("4단계 완료")

    current_time = datetime.now().strftime("%Y%m%d%H%M%S")

    # 답지 노래 txt로 저장
    with open(f"output_fft/{current_time}_reference.txt", "w") as ref_f:
        for i, (ref_start, ref_end) in enumerate(ref_segments):
            ref_pitch = sing.analyze_pitch_fft(ref_samples[ref_start:ref_end], sr)
            ref_f.write(f"Segment {i+1}: Ref Start {ref_start}, Ref End {ref_end}, Ref Pitch {ref_pitch:.2f} Hz\n")

    # 사용자가 부른 노래 저장
    sing.save_audio(user_samples, sr, f'output_fft/{current_time}_user_song.wav')
    print("5단계 완료")

    total_score = 0
    note_count = 0

    with open(f"output_fft/{current_time}_user_evaluation.txt", "w") as f:
        for i, (ref_start, ref_end) in enumerate(ref_segments):
            ref_pitch = sing.analyze_pitch_fft(ref_samples[ref_start:ref_end], sr)
            
            for user_start, user_end in user_segments:
                rhythm_evaluation, rhythm_score = sing.evaluate_rhythm(ref_start, ref_end, user_start, user_end)
                
                if rhythm_evaluation == "In-Sync":
                    user_pitch = sing.analyze_pitch_fft(user_samples[user_start:user_end], sr)
                    pitch_evaluation, pitch_score = sing.evaluate_pitch_fft(ref_pitch, user_pitch)
                    
                    ### 저장형식 변경 필요 ###
                    # f.write(f"Segment {i+1}: Ref Start {ref_start}, Ref End {ref_end}, User Start {user_start}, User End {user_end}, Rhythm: {rhythm_evaluation}, Pitch: {pitch_evaluation}, Rhythm Score: {rhythm_score}, Pitch Score: {pitch_score}\n")
                    f.write(f"Segment {i+1}: Ref Start {ref_start}, Ref End {ref_end}, User Start {user_start}, User End {user_end}, Ref Pitch {ref_pitch:.2f} Hz, User Pitch {user_pitch:.2f} Hz, Pitch: {pitch_evaluation}, Pitch Score: {pitch_score}\n")
                    
                    total_score += (pitch_score)  # 여기에 rhythm_score 추가 가능
                    note_count += 1
                    break

        final_score = total_score / note_count if note_count > 0 else 0
        f.write(f"\nFinal Score: {final_score:.2f}/100")

    print(f"Final Score: {final_score:.2f}/100")


### AI Cover 제작
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

    print("폴더명:", nsfile_name)
    print("분리된 음원의 경로:", f"{model_dir}\\spleeter\\{nsfile_name}")

    # AI Cover 생성
    input_audio = f"{model_dir}/spleeter/{nsfile_name}/vocals.wav"
    file_index2 = f"./AICover/pyaicover/models/logs/{model_name}/added_IVF1203_Flat_nprobe_1_{model_name}_v2.index"
    background_path = f"{model_dir}/spleeter/{nsfile_name}/"

    file_name = AICover.make_AICover(sid, vc_transform, input_audio, file_index2, index_rate, background_path)
    
    # 사용된 음원 삭제
    shutil.rmtree(f"{model_dir}\\spleeter\\{nsfile_name}", ignore_errors=True)

    # 서버로 완성된 AI Cover 전송
    print("전송을 시도합니다.")
    aicover = open(f"./pyaicover/models/merged_results/{input_audio.split('/')[-2]}_{model_name}.mp3", 'rb')
    upload = {'file': aicover}
    requests.post(' http://192.168.0.109:8080/api/created-song', files = upload)
    print("전송 완료")
    endtime = time.time()

    return f"AI Cover가 성공적으로 제작되었습니다. 소요시간 : {endtime-starttime}" 


# 음성 모델 훈련
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
            print("오류 발생")
            return "오류가 발생했습니다."
        

    return "ok"
        

        # train_datasets = f"{model_dir}\\datasets"
        # train_name = request.form['model']  # 모델의 이름
        # total_epoch11 = request.form['epochs']  # epoch 수

        # pretrained_G14 = "./AICover/pyaicover/models/pretrained_v2/f0G48k.pth"
        # pretrained_D15 = "./AICover/pyaicover/models/pretrained_v2/f0D48k.pth"

        # return f"{train_name} 모델을 서버로 전송해 훈련을 시작합니다."
        # train_model.train1key(train_name, trainset_dir4, total_epoch11, pretrained_G14, pretrained_D15)