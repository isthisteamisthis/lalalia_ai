import os
import shutil
import sys

model_dir = os.getcwd() + "\\pyaicover\\models"
now_dir = os.getcwd()
sys.path.append(model_dir)
sys.path.append(now_dir)

from pydub import AudioSegment
import traceback
import warnings
import numpy as np
import torch
import logging
import soundfile as sf
from config import Config
from fairseq import checkpoint_utils
from lib.infer_pack.models import (
    SynthesizerTrnMs256NSFsid,
    SynthesizerTrnMs256NSFsid_nono,
    SynthesizerTrnMs768NSFsid,
    SynthesizerTrnMs768NSFsid_nono,
)
from my_utils import load_audio
from vc_infer_pipeline import VC

logging.getLogger("numba").setLevel(logging.WARNING)

tmp = os.path.join(model_dir, "TEMP")
shutil.rmtree(tmp, ignore_errors=True)

os.makedirs(tmp, exist_ok=True)
os.makedirs(os.path.join(model_dir, "logs"), exist_ok=True)
os.makedirs(os.path.join(model_dir, "weights"), exist_ok=True)
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
else:
    gpu_info = "사용 가능한 GPU가 없습니다."
    default_batch_size = 1

gpus = "-".join([i[0] for i in gpu_infos])
# print(gpu_info)

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

weight_root = f"{model_dir}/weights"
weight_uvr5_root = f"{model_dir}/uvr5_weights"
index_root = f"{model_dir}/logs"
names = []

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


def vc_single(sid, input_audio_path, f0_up_key, f0_file, f0_method, file_index, file_index2, 
              index_rate, filter_radius, resample_sr, rms_mix_rate, protect): 

    global tgt_sr, net_g, vc, hubert_model, version
    
    if input_audio_path is None:
        return "오디오 파일을 업로드해주세요", None
    
    f0_up_key = int(f0_up_key)
    
    try:
        audio = load_audio(input_audio_path, 16000)
        audio_max = np.abs(audio).max() / 0.95
        if audio_max > 1:
            audio /= audio_max
        times = [0, 0, 0]
        if not hubert_model:
            load_hubert()
        if_f0 = cpt.get("f0", 1)
        file_index = (
            (
                file_index.strip(" ")
                .strip('"')
                .strip("\n")
                .strip('"')
                .strip(" ")
                .replace("trained", "added")
            )
            if file_index != ""
            else file_index2
        ) 
        audio_opt = vc.pipeline(hubert_model, net_g, sid, audio, input_audio_path, times,
                                f0_up_key, f0_method, file_index, index_rate, if_f0,
                                filter_radius, tgt_sr, resample_sr, rms_mix_rate, version,
                                protect, f0_file=f0_file)
        
        if tgt_sr != resample_sr >= 16000:
            tgt_sr = resample_sr

        print("변환 완료. 저장을 시작합니다.")
        
        return tgt_sr, audio_opt
    
    except:
        info = traceback.format_exc()
        print(info)
        return info, (None, None)


def get_vc(sid, to_return_protect0, to_return_protect1):
    global n_spk, tgt_sr, net_g, vc, cpt, version
    if sid == "" or sid == []:
        global hubert_model
        if hubert_model is not None:
            print("clean_empty_cache")

            del net_g, n_spk, vc, hubert_model, tgt_sr

            hubert_model = net_g = n_spk = vc = hubert_model = tgt_sr = None

            if torch.cuda.is_available():
                torch.cuda.empty_cache()

            if_f0 = cpt.get("f0", 1)

            version = cpt.get("version", "v1")

            if version == "v1":
                if if_f0 == 1:
                    net_g = SynthesizerTrnMs256NSFsid(
                        *cpt["config"], is_half=config.is_half
                    )
                else:
                    net_g = SynthesizerTrnMs256NSFsid_nono(*cpt["config"])

            elif version == "v2":
                if if_f0 == 1:
                    net_g = SynthesizerTrnMs768NSFsid(
                        *cpt["config"], is_half=config.is_half
                    )
                else:
                    net_g = SynthesizerTrnMs768NSFsid_nono(*cpt["config"])

            del net_g, cpt

            if torch.cuda.is_available():
                torch.cuda.empty_cache()

            cpt = None

        return {"visible": False, "__type__": "update"}

    # weights 폴더 안에 있는 모델.pth 불러오기
    person = "%s/%s" % (weight_root, sid)
    print("loading %s" % person)
    cpt = torch.load(person, map_location="cpu")
    tgt_sr = cpt["config"][-1]  # 48000
    cpt["config"][-3] = cpt["weight"]["emb_g.weight"].shape[0]  # n_spk : 109

    if_f0 = cpt.get("f0", 1)

    to_return_protect0 = {
        "visible": True,
        "value": to_return_protect0,
        "__type__": "update",
    }
    to_return_protect1 = {
        "visible": True,
        "value": to_return_protect1,
        "__type__": "update",
    }

    # get() 메소드의 두번째 매개변수는, 해당 딕셔너리에서 첫번째 매개변수인 key에 대응하는 값을 가져오지 못할 경우 None 대신 반환하는 기본값
    version = cpt.get("version", "v1")  # v2
    
    net_g = SynthesizerTrnMs768NSFsid(*cpt["config"], is_half=config.is_half)

    del net_g.enc_q
    print(net_g.load_state_dict(cpt["weight"], strict=False))
    net_g.eval().to(config.device)

    if config.is_half:
        net_g = net_g.half()
    else:
        net_g = net_g.float()

    vc = VC(tgt_sr, config)
    n_spk = cpt["config"][-3]

    return (
        {"visible": True, "maximum": n_spk, "__type__": "update"},
        to_return_protect0,
        to_return_protect1,
    )

sr_dict = {"32k": 32000, "40k": 40000, "48k": 48000}




def make_AICover(sid, vc_transform, input_audio, file_index2, index_rate, background_audio):
    # 보컬 합성하기
    protect0 = 0.33
    model_name = sid.split('.')[-2]
    spk_item, protect0, _ = get_vc(sid, protect0, protect0)
    f0method0 = "rmvpe"
    filter_radius0 = 3
    file_index1 = ""

    resample_sr0 = 0
    rms_mix_rate0 = 0.25

    f0_file = ""
    input_audio = input_audio.replace('\\', '/')
    tgt_sr, audio_opt = vc_single(0, input_audio, vc_transform, f0_file, f0method0, file_index1, file_index2, index_rate, filter_radius0, resample_sr0, rms_mix_rate0, protect0['value'])

    file_name = input_audio.split('/')[-2] + f"_{model_name}.wav"
    os.makedirs("./pyaicover/models/vocal_results", exist_ok=True)

    try:
        sf.write(f"./pyaicover/models/vocal_results/{file_name}", audio_opt, tgt_sr)
    except:
        print("저장에 실패했습니다.")

    rvc_vocal = AudioSegment.from_wav(f"./pyaicover/models/vocal_results/{file_name}")
    rvc_vocal = rvc_vocal + 3  # 데시벨 조절
    echo1 = rvc_vocal - 12
    echo2 = rvc_vocal - 15
    rvc_vocal = rvc_vocal.overlay(echo1, position=125)
    rvc_vocal = rvc_vocal.overlay(echo2, position=250)  # 에코 추가

    background = AudioSegment.from_wav(background_audio)

    result = background.overlay(rvc_vocal)
    os.makedirs("./pyaicover/models/merged_results", exist_ok=True)
    result_path = f"./pyaicover/models/merged_results/{file_name.split('_')[0]}_{model_name}.mp3"
    result.export(result_path, format="mp3")

    return result_path