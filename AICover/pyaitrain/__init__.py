import os, sys

flask_dir = os.getcwd() + "\\pyaicover"
model_dir = os.getcwd() + "\\pyaicover\\models"
sys.path.append(flask_dir)
sys.path.append(model_dir)

uvr_dir = os.getcwd() + "\\pyaicover\\uvr"
sys.path.append(uvr_dir)

from flask import Flask, render_template, request
from .models import train_model

app = Flask(__name__)

### Flask 서버
@app.route('/')
def index():
    return render_template('index.html')


# 음성 모델 훈련
@app.route('/aitrain', methods=['GET', 'POST'])
def train_models():
    # 서버 db에서 녹음된 음성 파일들 갖고 오기
    len = int(request.form['file_length'])
    musics = ['music' + str(i) for i in range(len)]

    for i in range(len):
        globals()[f"music{i}"] = request.files[musics[i]]
        print(globals()[f'music{i}'].filename)

        try:
            file_name, file_extension = os.path.splitext(globals()[f'music{i}'].filename)
            nsfile_name = file_name.replace(' ', '_')
            os.makedirs(model_dir + "\\datasets", exist_ok=True)
            
            music_path = f'{model_dir}\\datasets\\{nsfile_name}{file_extension}'
            globals()[f'music{i}'].save(music_path)

        except:
            print("오류 발생")
            return "오류가 발생했습니다."

    train_datasets = f"{model_dir}\\datasets"
    train_name = request.form['model']  # 모델의 이름
    total_epoch11 = request.form['epochs']  # epoch 수

    pretrained_G14 = "./AICover/pyaicover/models/pretrained_v2/f0G40k.pth"
    pretrained_D15 = "./AICover/pyaicover/models/pretrained_v2/f0D40k.pth"

    print(f"{train_name} 모델을 서버로 전송해 훈련을 시작합니다.")
    train_model.train1key(train_name, train_datasets, total_epoch11, pretrained_G14, pretrained_D15)

    return "train 완료"