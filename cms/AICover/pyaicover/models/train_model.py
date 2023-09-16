import os
import shutil
import sys

now_dir = os.getcwd() + "\\pyaicover\\models"
sys.path.append(now_dir)

import traceback, pdb
import warnings

import numpy as np
import torch

import logging
from random import shuffle
from subprocess import Popen
from time import sleep

import soundfile as sf
from config import Config
from fairseq import checkpoint_utils
import faiss
import ffmpeg
from sklearn.cluster import MiniBatchKMeans

logging.getLogger("numba").setLevel(logging.WARNING)


tmp = os.path.join(now_dir, "TEMP")
shutil.rmtree(tmp, ignore_errors=True)
shutil.rmtree(
    "%s/runtime/Lib/site-packages/lib.infer_pack" % (now_dir), ignore_errors=True
)
shutil.rmtree("%s/runtime/Lib/site-packages/uvr5_pack" % (now_dir), ignore_errors=True)
os.makedirs(tmp, exist_ok=True)
os.makedirs(os.path.join(now_dir, "logs"), exist_ok=True)
os.makedirs(os.path.join(now_dir, "weights"), exist_ok=True)
os.environ["TEMP"] = tmp
warnings.filterwarnings("ignore")
torch.manual_seed(114514)


config = Config()
ngpu = torch.cuda.device_count()
gpu_infos = []
mem = []
if_gpu_ok = False

if torch.cuda.is_available() or ngpu != 0:
    for i in range(ngpu):
        gpu_name = torch.cuda.get_device_name(i)
        if any(
            value in gpu_name.upper()
            for value in [
                "10", "16", "20", "30", "40", "A2", "A3", "A4", "P4",
                "A50", "500", "A60", "70", "80", "90", "M4", "T4", "TITAN",
            ]
        ):
            if_gpu_ok = True
            gpu_infos.append("%s\t%s" % (i, gpu_name))
            mem.append(
                int(torch.cuda.get_device_properties(i).total_memory
                    / 1024
                    / 1024
                    / 1024
                    + 0.4
                )
            )

gpu_info = ""
if if_gpu_ok and len(gpu_infos) > 0:
    gpu_info = "\n".join(gpu_infos)
    default_batch_size = min(mem) // 2
    print(default_batch_size)
else:
    gpu_info = "사용 가능한 GPU가 없습니다."
    default_batch_size = 1

gpus = "-".join([i[0] for i in gpu_infos])
print(gpu_info)

hubert_model = None

def load_hubert():
    global hubert_model
    models, _, _ = checkpoint_utils.load_model_ensemble_and_task(["hubert_base.pt"], suffix="")
    hubert_model = models[0]
    hubert_model = hubert_model.to(config.device)
    if config.is_half:
        hubert_model = hubert_model.half()
    else:
        hubert_model = hubert_model.float()

    hubert_model.eval()

weight_root = "weights"
weight_uvr5_root = "uvr5_weights"
index_root = "logs"
names = []

sr_dict = {"32k": 32000, "40k": 40000, "48k": 48000}

for name in os.listdir(weight_root):
    if name.endswith(".pth"):
        names.append(name)

index_paths = []
for root, dirs, files in os.walk(index_root, topdown=False):
    for name in files:
        if name.endswith(".index") and "trained" not in name:
            index_paths.append("%s/%s" % (root, name))

uvr5_names = []
for name in os.listdir(weight_uvr5_root):
    if name.endswith(".pth") or "onnx" in name:
        uvr5_names.append(name.replace(".pth", ""))

### 이거 다시 분석할 것!!!
# but5.click(train1key, [exp_dir1, sr2, if_f0_3, trainset_dir4, spk_id5, gpus6, np7, f0method8, save_epoch10, total_epoch11, batch_size12, if_save_latest13, pretrained_G14, pretrained_D15, gpus16, if_cache_gpu17], info3)
def train1key(exp_dir1, trainset_dir4, total_epoch11, gpus16, pretrained_G14, pretrained_D15):
    infos = []

    sr2 = "48k"  # 샘플링 주파수 설정 : wav 파일에 따라 결정 (44100 : 40K / 48000 이상 : 48K)
    if_f0_3 = True  # 모델이 음의 높낮이(pitch)를 가지고 있는가? True / False
    save_epoch10 = 1  # 학습 중 중간 저장빈도
    spk_id5 = 0  # ID가 하나씩 늘어날때마다 화자, 가수의 숫자가 늘어남 (듀엣 훈련 가능?)
    np7 = int(np.ceil(config.n_cpu / 1.5))  # pitch 추출 및 데이터 처리에 사용되는 cpu 프로세스 수
    batch_size12 = default_batch_size  # batch 사이즈 조정 : 너무 많이 올리면 VRAM 초과로 중단된다. 3이나 4로 사용
    f0method8 = "harvest"  # 음정 추출 알고리즘 선택 (기본 : harvest / dio는 연설 등 speech에서 사용)
    version19 = "v2"  # 모델 버전 : v1 / v2

    # 가장 최신의 .ckpt 파일만 저장할 것인가?

    # GPU 메모리 캐시 사용 여부 : 사용 시 VRAM 초과로 중단될 우려 있음음

    # 학습마다 모든 가중치를 저장할 것인지? : NO, 기본값으로 둔다.


    # 생성 파일 이름 : exp_dir1은 모델의 이름, logs/모델이름 하위폴더 생성
    model_log_dir = "%s/logs/%s" % (now_dir, exp_dir1)
    preprocess_log_path = "%s/preprocess.log" % model_log_dir
    extract_f0_feature_log_path = "%s/extract_f0_feature.log" % model_log_dir

    # 곡에서 목소리를 추출해 음절? 단위로 저장
    gt_wavs_dir = "%s/0_gt_wavs" % model_log_dir

    feature_dir = (
        "%s/3_feature256" % model_log_dir
        if version19 == "v1"
        else "%s/3_feature768" % model_log_dir
    )

    os.makedirs(model_log_dir, exist_ok=True)

    #########step1 : 데이터 처리
    open(preprocess_log_path, "w").close()
    cmd = (
            config.python_cmd
            + " trainset_preprocess_pipeline_print.py %s %s %s %s "
            % (trainset_dir4, sr_dict[sr2], np7, model_log_dir)
            + str(config.noparallel)
    )

    print("step1: 데이터 처리 중...")
    print(cmd)
    p = Popen(cmd, shell=True)
    p.wait()
    with open(preprocess_log_path, "r") as f:
        print(f.read())

    #########step2a : 음높이(pitch) 추출
    open(extract_f0_feature_log_path, "w")
    if if_f0_3:
        print("step2a: pitch 추출 중...")
        cmd = config.python_cmd + " extract_f0_print.py %s %s %s" % (
            model_log_dir,
            np7,
            f0method8,
        )
        print(cmd)
        p = Popen(cmd, shell=True, cwd=now_dir)
        p.wait()
        with open(extract_f0_feature_log_path, "r") as f:
            print(f.read())

    # Speech용
    else:
        print("step2a:음높이를 추출할 필요가 없습니다.")

    #######step2b : 특징 추출
    print("step2b: 특징 추출 중...")

    gpus = gpus16.split("-")

    leng = len(gpus)

    ps = []
    for idx, n_g in enumerate(gpus):
        cmd = config.python_cmd + " extract_feature_print.py %s %s %s %s %s %s" % (
            config.device,
            leng,
            idx,
            n_g,
            model_log_dir,
            version19,
        )
        print(cmd)

        p = Popen(cmd, shell=True, cwd=now_dir)  # , shell=True, stdin=PIPE, stdout=PIPE, stderr=PIPE, cwd=now_dir
        ps.append(p)

    for p in ps:
        p.wait()

    with open(extract_f0_feature_log_path, "r") as f:
        print(f.read())

    #######step3a : 훈련모델
    print("step3a: train 중...")

    # 파일 목록 생성
    # 모델이 음의 높낮이를 가지고 있는지(True: 가지고 있음 / False: 안 가지고 있음)
    if if_f0_3:
        f0_dir = "%s/2a_f0" % model_log_dir
        f0nsf_dir = "%s/2b-f0nsf" % model_log_dir
        names = (
                set([name.split(".")[0] for name in os.listdir(gt_wavs_dir)])
                & set([name.split(".")[0] for name in os.listdir(feature_dir)])
                & set([name.split(".")[0] for name in os.listdir(f0_dir)])
                & set([name.split(".")[0] for name in os.listdir(f0nsf_dir)])
        )

    # Speech용
    else:
        names = set([name.split(".")[0] for name in os.listdir(gt_wavs_dir)]) & set(
            [name.split(".")[0] for name in os.listdir(feature_dir)]
        )

    opt = []
    for name in names:
        if if_f0_3:
            opt.append(
                "%s/%s.wav|%s/%s.npy|%s/%s.wav.npy|%s/%s.wav.npy|%s"
                % (
                    gt_wavs_dir.replace("\\", "\\\\"), name,
                    feature_dir.replace("\\", "\\\\"), name,
                    f0_dir.replace("\\", "\\\\"), name,
                    f0nsf_dir.replace("\\", "\\\\"), name,
                    spk_id5,
                )
            )

        # Speech용
        else:
            opt.append(
                "%s/%s.wav|%s/%s.npy|%s"
                % (
                    gt_wavs_dir.replace("\\", "\\\\"),
                    name,
                    feature_dir.replace("\\", "\\\\"),
                    name,
                    spk_id5,
                )
            )

    # v2 모델을 기본이므로 768이 기본
    fea_dim = 768 if version19 == "v2" else 256

    if if_f0_3:
        for _ in range(2):
            opt.append(
                "%s/logs/mute/0_gt_wavs/mute%s.wav|%s/logs/mute/3_feature%s/mute.npy|%s/logs/mute/2a_f0/mute.wav.npy|%s/logs/mute/2b-f0nsf/mute.wav.npy|%s"
                % (now_dir, sr2, now_dir, fea_dim, now_dir, now_dir, spk_id5)
            )

    # Speech용
    else:
        for _ in range(2):
            opt.append(
                "%s/logs/mute/0_gt_wavs/mute%s.wav|%s/logs/mute/3_feature%s/mute.npy|%s"
                % (now_dir, sr2, now_dir, fea_dim, spk_id5)
            )

    shuffle(opt)  # 리스트 내의 값을 섞기

    with open("%s/filelist.txt" % model_log_dir, "w") as f:
        f.write("\n".join(opt))

    print("write filelist done")

    # GPU 0, 1, 2번을 사용하려면 GPU 인덱스를 '-'로 분리해서 입력
    # 도둑(Generator, pretrained_G14) / 경찰(Disciminator, pretrained_D15)
    # 도둑 : Disciminator의 판별 비율을 리턴받아 더욱 정교하게 생성
    # 경찰 : Generator로 만든 결과물과 기존에 있는 결과물의 판별 비율이 0.5:0.5가 될때까지 비교
    if gpus16:
        cmd = (
                config.python_cmd
                + " train_nsf_sim_cache_sid_load_pretrain.py -e %s -sr %s -f0 %s -bs %s -g %s -te %s -se %s %s %s -l %s -c %s -sw %s -v %s"
                % (
                    exp_dir1,
                    sr2,
                    1 if if_f0_3 else 0,
                    batch_size12,
                    gpus16,
                    total_epoch11,
                    save_epoch10,
                    "-pg %s" % pretrained_G14 if pretrained_G14 != "" else "",
                    "-pd %s" % pretrained_D15 if pretrained_D15 != "" else "",
                    0,
                    0,
                    0,
                    version19,
                )
        )
    else:
        cmd = (
                config.python_cmd
                + " train_nsf_sim_cache_sid_load_pretrain.py -e %s -sr %s -f0 %s -bs %s -te %s -se %s %s %s -l %s -c %s -sw %s -v %s"
                % (
                    exp_dir1,
                    sr2,
                    1 if if_f0_3 else 0,
                    batch_size12,
                    total_epoch11,
                    save_epoch10,
                    "-pg %s" % pretrained_G14 if pretrained_G14 != "" else "",
                    "-pd %s" % pretrained_D15 if pretrained_D15 != "" else "",
                    0,
                    0,
                    0,
                    version19,
                )
        )

    print(cmd)

    p = Popen(cmd, shell=True, cwd=now_dir)
    p.wait()
    print("훈련이 끝났습니다. 콘솔이나 log 폴더에서 train.log를 확인할 수 있습니다.")

    #######step3b:train index
    print("train index")
    npys = []
    listdir_res = list(os.listdir(feature_dir))
    for name in sorted(listdir_res):
        phone = np.load("%s/%s" % (feature_dir, name))
        npys.append(phone)

    big_npy = np.concatenate(npys, 0)
    big_npy_idx = np.arange(big_npy.shape[0])
    np.random.shuffle(big_npy_idx)
    big_npy = big_npy[big_npy_idx]

    if big_npy.shape[0] > 2e5:
        # if(1):
        info = "Trying doing kmeans %s shape to 10k centers." % big_npy.shape[0]
        print(info)
        try:
            big_npy = (
                MiniBatchKMeans(
                    n_clusters=10000,
                    verbose=True,
                    batch_size=256 * config.n_cpu,
                    compute_labels=False,
                    init="random",
                )
                .fit(big_npy)
                .cluster_centers_
            )
        except:
            info = traceback.format_exc()
            print(info)

    np.save("%s/total_fea.npy" % model_log_dir, big_npy)

    # n_ivf =  big_npy.shape[0] // 39
    n_ivf = min(int(16 * np.sqrt(big_npy.shape[0])), big_npy.shape[0] // 39)
    print("%s,%s" % (big_npy.shape, n_ivf))

    index = faiss.index_factory(256 if version19 == "v1" else 768, "IVF%s,Flat" % n_ivf)
    print("training index")

    index_ivf = faiss.extract_index_ivf(index)  #
    index_ivf.nprobe = 1
    index.train(big_npy)

    faiss.write_index(
        index,
        "%s/trained_IVF%s_Flat_nprobe_%s_%s_%s.index"
        % (model_log_dir, n_ivf, index_ivf.nprobe, exp_dir1, version19),
    )
    print("adding index")

    batch_size_add = 8192

    for i in range(0, big_npy.shape[0], batch_size_add):
        index.add(big_npy[i: i + batch_size_add])

    faiss.write_index(
        index,
        "%s/added_IVF%s_Flat_nprobe_%s_%s_%s.index"
        % (model_log_dir, n_ivf, index_ivf.nprobe, exp_dir1, version19),
    )

    print(
        "added_IVF%s_Flat_nprobe_%s_%s_%s.index 를 성공적으로 작성했습니다."
        % (n_ivf, index_ivf.nprobe, exp_dir1, version19)
    )

    print("전체 프로세스 종료！")


trainset_dir4 = "./datasets" # train data(vocal 파일)
train_name = "hyungku"  # 모델의 이름
total_epoch11 = 5  # epoch 수
gpus16 = gpus
pretrained_G14 = "./pretrained_v2/f0G48k.pth"
pretrained_D15 = "./pretrained_v2/f0D48k.pth"

train1key(train_name, trainset_dir4, total_epoch11, gpus16, pretrained_G14, pretrained_D15)