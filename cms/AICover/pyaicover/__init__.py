import os, sys

model_dir = os.getcwd() + "\\pyaicover\\models"
sys.path.append(model_dir)

from flask import Flask, request
from .models import AICover
import requests
import shutil

app = Flask(__name__)

@app.route('/')
def index():
    return "메인화면입니다."

@app.route('/aicover', methods=['GET', 'POST'])
def create_aicover():
    # print("파일:", request.files['file'])
    # print("모델 이름:", request.form['model'], "/ 타입:", type(request.form['model']))
    # print("옥타브:", request.form['octave'], "/ 타입:", type(request.form['octave']))
    

    model_name = request.form['model']

    sid = model_name + ".pth"  # 절대 경로 입력하지말고 weights 폴더 안의 파일 이름만 입력할 것
    vc_transform = int(request.form['octave'])  # 옥타브 설정(-12 ~ 12)
    index_rate = 0.75  # index 파일 참조 비율 : 높을수록 모델쪽의 발음/음정, 낮을수록 기존 보컬쪽의 발음/음정 중시

    print("모델:", sid)
    print("옥타브:", vc_transform)

    music = request.files['music']
    print(music.filename)

    # Extract the directory and file name from the full path
    file_name, file_extension = os.path.splitext(music.filename)
    print("파일 이름:", file_name)
    print("파일 확장자:", file_extension)
    
    # Replace spaces with underscores in the file name
    nsfile_name = file_name.replace(' ', '_')
    os.makedirs(model_dir + "\\spleeter", exist_ok=True)

    # Rename the file if it exists
    try:
        os.rename(os.path.join(model_dir + "\\spleeter\\", file_name + file_extension), os.path.join(model_dir + "\\spleeter\\", nsfile_name + file_extension))
    except FileNotFoundError:
        pass

    music.save(f'{model_dir}\\spleeter\\{nsfile_name}{file_extension}')

    print('기다려주세요.')

    # Run Spleeter
    spl = f'spleeter separate -p spleeter:2stems -o pyaicover/models/spleeter {model_dir}\\spleeter\\{nsfile_name}{file_extension}'
    os.system(spl)

    print("완료되었습니다.")
    print("폴더명:", nsfile_name)
    print("최종경로:", f"{model_dir}\\spleeter\\{nsfile_name}")

    input_audio = f"{model_dir}\\spleeter\\{nsfile_name}\\vocals.wav"
    file_index2 = f"./AICover/pyaicover/models/logs/{model_name}/added_IVF1203_Flat_nprobe_1_{model_name}_v2.index"
    background_path = f"{model_dir}\\spleeter\\{nsfile_name}\\accompaniment.wav"

    file_name = AICover.make_AICover(sid, vc_transform, input_audio, file_index2, index_rate, background_path)
    print(file_name)
    
    aicover = open(f"./pyaicover/models/vocal_results/{file_name}", 'rb')
    print("전송을 시도합니다.")
    upload = {'perfect-score': aicover}
    requests.post(' http://192.168.0.88:8080/api/perfect-scores', files = upload)
    print("전송 완료")

    return "AI Cover가 성공적으로 제작되었습니다."


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