import os, sys

model_dir = os.getcwd() + "\\pyaicover\\models"
sys.path.append(model_dir)

from flask import Flask, request
from .models import AICover
import requests
import shutil
import time

app = Flask(__name__)

@app.route('/')
def index():
    return "메인화면입니다."

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
    spl = f'spleeter separate -p spleeter:2stems -o pyaicover/models/spleeter {model_dir}\\spleeter\\{nsfile_name}{file_extension}'
    os.system(spl)

    # 서버에서 받은 음원 삭제
    if os.path.isfile(music_path):
        os.remove(music_path)

    print("분리가 완료되었습니다.")
    print("폴더명:", nsfile_name)
    print("분리된 음원의 경로:", f"{model_dir}\\spleeter\\{nsfile_name}")

    # AI Cover 생성
    input_audio = f"{model_dir}\\spleeter\\{nsfile_name}\\vocals.wav"
    file_index2 = f"./AICover/pyaicover/models/logs/{model_name}/added_IVF1203_Flat_nprobe_1_{model_name}_v2.index"
    background_path = f"{model_dir}\\spleeter\\{nsfile_name}\\accompaniment.wav"

    file_name = AICover.make_AICover(sid, vc_transform, input_audio, file_index2, index_rate, background_path)
    
    # 사용된 음원 삭제
    shutil.rmtree(f"{model_dir}\\spleeter\\{nsfile_name}", ignore_errors=True)

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
    try:
        trainset_dir4 = request.files['file']  # train data(vocal 파일)
        train_name = request.form['model']  # 모델의 이름
        total_epoch11 = request.form['epochs']  # epoch 수

        pretrained_G14 = "./AICover/pyaicover/models/pretrained_v2/f0G48k.pth"
        pretrained_D15 = "./AICover/pyaicover/models/pretrained_v2/f0D48k.pth"

        return f"{train_name} 모델을 서버로 전송해 훈련을 시작합니다."
        # train_model.train1key(train_name, trainset_dir4, total_epoch11, pretrained_G14, pretrained_D15)
    
    except:
        return "오류가 발생했습니다."