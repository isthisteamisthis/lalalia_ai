import os
import sys

now_dir = os.getcwd()
uvr_dir = os.getcwd() + "\\pyaicover\\uvr"

import json
import urllib
import psutil
import hashlib
import torch
import torch.onnx

from gui_data.constants import *
from gui_data.saved_ensembles import *
from lib_v5 import spec_utils

import time
import shutil
from separate import SeperateDemucs, SeperateMDX, SeperateVR, save_format
from typing import List
import librosa
import traceback

if getattr(sys, 'frozen', False):
    BASE_PATH = sys._MEIPASS
else:
    BASE_PATH = os.path.dirname(os.path.abspath(__file__))

os.chdir(BASE_PATH) 


#Check GPUs
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

#Models
MODELS_DIR = os.path.join(BASE_PATH, 'models')
VR_MODELS_DIR = os.path.join(MODELS_DIR, 'VR_Models')
MDX_MODELS_DIR = os.path.join(MODELS_DIR, 'MDX_Net_Models')
DEMUCS_MODELS_DIR = os.path.join(MODELS_DIR, 'Demucs_Models')
DEMUCS_NEWER_REPO_DIR = os.path.join(DEMUCS_MODELS_DIR, 'v3_v4_repo')
MDX_MIXER_PATH = os.path.join(BASE_PATH, 'lib_v5', 'mixer.ckpt')

#Cache & Parameters
VR_HASH_DIR = os.path.join(VR_MODELS_DIR, 'model_data')
VR_HASH_JSON = os.path.join(VR_MODELS_DIR, 'model_data', 'model_data.json')
MDX_HASH_DIR = os.path.join(MDX_MODELS_DIR, 'model_data')
MDX_HASH_JSON = os.path.join(MDX_MODELS_DIR, 'model_data', 'model_data.json')
DEMUCS_MODEL_NAME_SELECT = os.path.join(DEMUCS_MODELS_DIR, 'model_data', 'model_name_mapper.json')
MDX_MODEL_NAME_SELECT = os.path.join(MDX_MODELS_DIR, 'model_data', 'model_name_mapper.json')
ENSEMBLE_CACHE_DIR = os.path.join(BASE_PATH, 'gui_data', 'saved_ensembles')
SETTINGS_CACHE_DIR = os.path.join(BASE_PATH, 'gui_data', 'saved_settings')
VR_PARAM_DIR = os.path.join(BASE_PATH, 'lib_v5', 'vr_network', 'modelparams')
SAMPLE_CLIP_PATH = os.path.join(BASE_PATH, 'temp_sample_clips')
ENSEMBLE_TEMP_PATH = os.path.join(BASE_PATH, 'ensemble_temps')

#Other
COMPLETE_CHIME = os.path.join(BASE_PATH, 'gui_data', 'complete_chime.wav')
FAIL_CHIME = os.path.join(BASE_PATH, 'gui_data', 'fail_chime.wav')
CHANGE_LOG = os.path.join(BASE_PATH, 'gui_data', 'change_log.txt')
SPLASH_DOC = os.path.join(BASE_PATH, 'tmp', 'splash.txt')

model_hash_table = {}

with open('./gui_data/saved_ensembles/train.json') as ensembles:
    ensembles_train = json.load(ensembles)

train_stem = ensembles_train['ensemble_main_stem']
train_type = ensembles_train['ensemble_type']
train_models1 = ensembles_train['selected_models'][0]
train_models2 = ensembles_train['selected_models'][1]
train_models3 = ensembles_train['selected_models'][2]
train_models4 = ensembles_train['selected_models'][3]
train_models5 = ensembles_train['selected_models'][4]
train_models6 = ensembles_train['selected_models'][5]
train_models7 = ensembles_train['selected_models'][6]
train_models = [train_models1, train_models2, train_models3, train_models4, train_models5, train_models6, train_models7]

train_models8 = "MDX-Net: UVR-MDX-NET Karaoke 2"
train_models9 = "MDX-Net: Reverb HQ"

# 파일 불러오기
input_path = '통신으로 받아 저장한 음악 파일 위치'

# 1st step : MAIN STEM PAIR=Vocals/Instrumental, GPU Conversion=True, ENSEMBLE_ALGORITHM=Average/Average
with open('./gui_data/saved_ensembles/pred.json') as ensembles:
    ensembles_pred = json.load(ensembles)

pred_stem = ensembles_pred['ensemble_main_stem']
pred_type = ensembles_pred['ensemble_type']
pred_models1 = ensembles_pred['selected_models'][0]
pred_models2 = ensembles_pred['selected_models'][1]
pred_models = [pred_models1, pred_models2]

# 2nd step : Batch size=4, GPU Conversion=True, Vocals Only=True, Volume Compensation=Auto
pred_models3 = "MDX-Net: UVR-MDX-NET Karaoke 2"

# 3rd step : Batch size=4, GPU Conversion=True, No Other Only=True, Volume Compensation=Auto
pred_models4 = "MDX-Net: Reverb HQ"

vr_cache_source_mapper = {}
mdx_cache_source_mapper = {}
demucs_cache_source_mapper = {}

os.chdir(now_dir)

def separate_process(mode, method, input_path, output_path, model_var=None):
    stime = time.perf_counter()
    time_elapsed = lambda:f'Time Elapsed: {time.strftime("%H:%M:%S", time.gmtime(int(time.perf_counter() - stime)))}'
    export_path = output_path
    true_model_count = 0
    iteration = 0
    is_verified_audio = True
    inputPaths = input_path
    inputPath_total_len = len(inputPaths) if mode == 'train' else 1
    is_ensemble = False

    is_primary_stem_only_var = 'PY_VAR73'
    is_secondary_stem_only_var = 'PY_VAR74'
    
    if method == ENSEMBLE_MODE:
        model, ensemble = assemble_model_data(mode), Ensembler(mode)
        export_path, is_ensemble = ensemble.ensemble_folder_name, True
    elif method == VR_ARCH_PM:
        model = assemble_model_data(mode, model_var, VR_ARCH_TYPE)
    elif method == MDX_ARCH_TYPE:
        model = assemble_model_data(mode, model_var, MDX_ARCH_TYPE)
    elif method == DEMUCS_ARCH_TYPE:
        model = assemble_model_data(mode, model_var, DEMUCS_ARCH_TYPE)

    all_models = cached_source_model_list_check(model)

    true_model_4_stem_count = sum(m.demucs_4_stem_added_count if m.process_method == DEMUCS_ARCH_TYPE else 0 for m in model)
    true_model_pre_proc_model_count = sum(2 if m.pre_proc_model_activated else 0 for m in model)
    true_model_count = sum(2 if m.is_secondary_model_activated else 1 for m in model) + true_model_4_stem_count + true_model_pre_proc_model_count

    for file_num, audio_file in enumerate(inputPaths, start=1):
        cached_sources_clear()
        base_text = process_get_baseText(total_files=1, file_num=file_num)

        for current_model_num, current_model in enumerate(model, start=1):
            iteration += 1

            if is_ensemble:
                print(f'Ensemble Mode - {current_model.model_basename} - Model {current_model_num}/{len(model)}{NEW_LINES}')

            model_name_text = f'({current_model.model_basename})' if not is_ensemble else ''
            print(base_text + f'Loading model {model_name_text}...')

            set_progress_bar = lambda step, inference_iterations=0:process_update_progress(true_model_count, inputPath_total_len, iteration, step=(step + (inference_iterations)))
            write_to_console = lambda progress_text, base_text=base_text:print(base_text + progress_text)

            audio_file_base = f"{file_num}_{os.path.splitext(os.path.basename(audio_file))[0]}"
            audio_file_base = audio_file_base if not is_ensemble else f"{audio_file_base}_{current_model.model_basename}"
            

            process_data = {
                            'model_data': current_model, 
                            'export_path': export_path,
                            'audio_file_base': audio_file_base,
                            'audio_file': audio_file,
                            'set_progress_bar': set_progress_bar,
                            'write_to_console': write_to_console,
                            'process_iteration': process_iteration,
                            'cached_source_callback': cached_source_callback,
                            'cached_model_source_holder': cached_model_source_holder,
                            'list_all_models': all_models,
                            'is_ensemble_master': is_ensemble,
                            'is_4_stem_ensemble': False}
            
            if current_model.process_method == VR_ARCH_TYPE:
                seperator = SeperateVR(current_model, process_data)
            if current_model.process_method == MDX_ARCH_TYPE:
                seperator = SeperateMDX(current_model, process_data)
            if current_model.process_method == DEMUCS_ARCH_TYPE:
                seperator = SeperateDemucs(current_model, process_data)
                
            seperator.seperate()
            
            if is_ensemble:
                print('\n')

        if is_ensemble:
            audio_file_base = audio_file_base.replace(f"_{current_model.model_basename}","")
            print(base_text + ENSEMBLING_OUTPUTS)
            
            ensemble_main_stem_var = train_type if mode == 'train' else pred_type
            
            if ensemble_main_stem_var == FOUR_STEM_ENSEMBLE:
                for output_stem in DEMUCS_4_SOURCE_LIST:
                    ensemble.ensemble_outputs(audio_file_base, export_path, output_stem, is_4_stem=True)
            else:
                if is_secondary_stem_only_var:
                    ensemble.ensemble_outputs(audio_file_base, export_path, PRIMARY_STEM)
                if is_primary_stem_only_var:
                    ensemble.ensemble_outputs(audio_file_base, export_path, SECONDARY_STEM)
                    ensemble.ensemble_outputs(audio_file_base, export_path, SECONDARY_STEM, is_inst_mix=True)

        print(DONE)
            
        torch.cuda.empty_cache()
        
    shutil.rmtree(export_path) if is_ensemble and len(os.listdir(export_path)) == 0 else None

    if inputPath_total_len == 1 and is_verified_audio:
        set_progress_bar(1.0)
        print('\nProcess Complete\n')
        print(time_elapsed())


# 앙상블
class Ensembler():
    def __init__(self, mode, is_manual_ensemble=False):
        self.is_save_all_outputs_ensemble = True
        chosen_ensemble_name = 'train' if mode == 'train' else 'pred'
        ensemble_algorithm = train_type.partition("/")
        ensemble_main_stem_pair = pred_stem.partition("/")
        time_stamp = round(time.time())
        self.audio_tool = MANUAL_ENSEMBLE
        self.main_export_path = f'{uvr_dir}\\outputs1'
        self.chosen_ensemble = ''
        ensemble_folder_name = f'{uvr_dir}\\outputs1'
        self.ensemble_folder_name = os.path.join(ensemble_folder_name, '{}_Outputs_{}'.format(chosen_ensemble_name, time_stamp))
        self.is_testing_audio = ''
        self.primary_algorithm = ensemble_algorithm[0]
        self.secondary_algorithm = ensemble_algorithm[2]
        self.ensemble_primary_stem = ensemble_main_stem_pair[0]
        self.ensemble_secondary_stem = ensemble_main_stem_pair[2]
        self.is_normalization = False
        self.wav_type_set = "PCM_16"
        self.mp3_bit_set = "320K"
        self.save_format = "WAV"
        if not is_manual_ensemble:
            os.mkdir(self.ensemble_folder_name)

    def ensemble_outputs(self, audio_file_base, export_path, stem, is_4_stem=False, is_inst_mix=False):
        """Processes the given outputs and ensembles them with the chosen algorithm"""
        
        if is_4_stem:
            algorithm = train_type
            stem_tag = stem
        else:
            if is_inst_mix:
                algorithm = self.secondary_algorithm
                stem_tag = f"{self.ensemble_secondary_stem} {INST_STEM}"
            else:
                algorithm = self.primary_algorithm if stem == PRIMARY_STEM else self.secondary_algorithm
                stem_tag = self.ensemble_primary_stem if stem == PRIMARY_STEM else self.ensemble_secondary_stem

        stem_outputs = self.get_files_to_ensemble(folder=export_path, prefix=audio_file_base, suffix=f"_({stem_tag}).wav")
        audio_file_output = f"{self.is_testing_audio}{audio_file_base}{self.chosen_ensemble}_({stem_tag})"
        stem_save_path = os.path.join('{}'.format(self.main_export_path),'{}.wav'.format(audio_file_output))
        
        if stem_outputs:
            spec_utils.ensemble_inputs(stem_outputs, algorithm, self.is_normalization, self.wav_type_set, stem_save_path)
            save_format(stem_save_path, self.save_format, self.mp3_bit_set)
        
        if self.is_save_all_outputs_ensemble:
            for i in stem_outputs:
                save_format(i, self.save_format, self.mp3_bit_set)
        else:
            for i in stem_outputs:
                try:
                    os.remove(i)
                except Exception as e:
                    print(e)

    def ensemble_manual(self, audio_inputs, audio_file_base, is_bulk=False):
        """Processes the given outputs and ensembles them with the chosen algorithm"""
        
        is_mv_sep = True
        
        if is_bulk:
            number_list = list(set([os.path.basename(i).split("_")[0] for i in audio_inputs]))
            for n in number_list:
                current_list = [i for i in audio_inputs if os.path.basename(i).startswith(n)]
                audio_file_base = os.path.basename(current_list[0]).split('.wav')[0]
                stem_testing = "instrum" if "Instrumental" in audio_file_base else "vocals"
                if is_mv_sep:
                    audio_file_base = audio_file_base.split("_")
                    audio_file_base = f"{audio_file_base[1]}_{audio_file_base[2]}_{stem_testing}"
                self.ensemble_manual_process(current_list, audio_file_base, is_bulk)
        else:
            self.ensemble_manual_process(audio_inputs, audio_file_base, is_bulk)
            
    def ensemble_manual_process(self, audio_inputs, audio_file_base, is_bulk):
        
        algorithm = train_type
        algorithm_text = "" if is_bulk else f"_({train_type})"
        stem_save_path = os.path.join('{}'.format(self.main_export_path),'{}{}{}.wav'.format(self.is_testing_audio, audio_file_base, algorithm_text))
        spec_utils.ensemble_inputs(audio_inputs, algorithm, self.is_normalization, self.wav_type_set, stem_save_path)
        save_format(stem_save_path, self.save_format, self.mp3_bit_set)

    def get_files_to_ensemble(self, folder="", prefix="", suffix=""):
        """Grab all the files to be ensembled"""
        
        return [os.path.join(folder, i) for i in os.listdir(folder) if i.startswith(prefix) and i.endswith(suffix)]


# 모델 1개 당 1번씩 불러옴 : onnx의 정보?
class ModelData():
    def __init__(self, model_name: str, 
                 selected_process_method=ENSEMBLE_MODE, 
                 is_secondary_model=False, 
                 primary_model_primary_stem=None, 
                 is_primary_model_primary_stem_only=False, 
                 is_primary_model_secondary_stem_only=False, 
                 is_pre_proc_model=False,
                 is_dry_check=False):
        
        self.is_gpu_conversion = 0
        self.is_normalization = False
        self.is_primary_stem_only = False
        # Karaoke2 : True / 나머지 : False
        self.is_secondary_stem_only = True if model_name == "UVR-MDX-NET Karaoke 2" else False
        self.is_denoise = False
        self.mdx_batch_size = default_batch_size
        self.is_mdx_ckpt = False
        self.wav_type_set = None
        self.mp3_bit_set = "320k"
        self.save_format = "WAV"
        self.is_invert_spec = False
        self.is_mixer_mode = False
        self.demucs_stems = "All Stems"
        self.demucs_source_list = []
        self.demucs_stem_count = 0
        self.mixer_path = MDX_MIXER_PATH
        self.model_name = model_name  # str
        self.process_method = selected_process_method  # Ensemble_Mode
        self.model_status = True
        self.primary_stem = None
        self.secondary_stem = None
        self.is_ensemble_mode = False
        self.ensemble_primary_stem = None
        self.ensemble_secondary_stem = None
        self.primary_model_primary_stem = None
        self.is_secondary_model = False
        self.secondary_model = None
        self.secondary_model_scale = None
        self.demucs_4_stem_added_count = 0
        self.is_demucs_4_stem_secondaries = False
        self.is_4_stem_ensemble = False
        self.pre_proc_model = None
        self.pre_proc_model_activated = False
        self.is_pre_proc_model = False
        self.is_dry_check = False
        self.model_samplerate = 44100
        self.model_capacity = 32, 128
        self.is_vr_51_model = False
        self.is_demucs_pre_proc_model_inst_mix = False
        self.manual_download_Button = None
        self.secondary_model_4_stem = []
        self.secondary_model_4_stem_scale = []
        self.secondary_model_4_stem_names = []
        self.secondary_model_4_stem_model_names_list = []
        self.all_models = []
        self.secondary_model_other = None
        self.secondary_model_scale_other = None
        self.secondary_model_bass = None
        self.secondary_model_scale_bass = None
        self.secondary_model_drums = None
        self.secondary_model_scale_drums = None

        if selected_process_method == ENSEMBLE_MODE:
            partitioned_name = model_name.partition(ENSEMBLE_PARTITION)
            self.process_method = partitioned_name[0]  # MDX_NET, VR_Arch, Demucs
            self.model_name = partitioned_name[2]  # model name
            self.model_and_process_tag = model_name
            self.ensemble_primary_stem, self.ensemble_secondary_stem = return_ensemble_stems(pred_stem)
            self.is_ensemble_mode = True
            self.is_4_stem_ensemble = False
            self.pre_proc_model_activated = False

        if self.process_method == MDX_ARCH_TYPE:
            self.is_secondary_model_activated = False
            self.margin = 44100
            self.chunks = 0
            self.get_mdx_model_path()
            self.get_model_hash()
            if self.model_hash:
                self.mdx_hash_MAPPER = load_model_hash_data(MDX_HASH_JSON)
                self.model_data = self.get_model_data(MDX_HASH_DIR, self.mdx_hash_MAPPER)
                if self.model_data:
                    self.compensate = self.model_data["compensate"]
                    self.mdx_dim_f_set = self.model_data["mdx_dim_f_set"]
                    self.mdx_dim_t_set = self.model_data["mdx_dim_t_set"]
                    self.mdx_n_fft_scale_set = self.model_data["mdx_n_fft_scale_set"]
                    self.primary_stem = self.model_data["primary_stem"]
                    self.secondary_stem = STEM_PAIR_MAPPER[self.primary_stem]
                else:
                    self.model_status = False

        if self.process_method == DEMUCS_ARCH_TYPE:
            self.is_secondary_model_activated =  False
            if not self.is_ensemble_mode:
                self.pre_proc_model_activated = False
            self.overlap = 0.25
            self.margin_demucs = 44100
            self.chunks_demucs = determine_auto_chunks('Auto', self.is_gpu_conversion)
            self.shifts = 2
            self.is_split_mode = True
            self.segment = "Default"
            self.is_chunk_demucs = False
            self.is_demucs_combine_stems = True
            self.is_primary_stem_only = False
            self.is_secondary_stem_only = False
            self.get_demucs_model_path()
            self.get_demucs_model_data()


        # 모델 경로에서 파일의 기본 이름을 추출
        self.model_basename = os.path.splitext(os.path.basename(self.model_path))[0] if self.model_status else None
        print("self.model_basename:", self.model_basename)

        # pre-processing 모델이 활성화되었는지 여부
        self.pre_proc_model_activated = False
        
        # primary_model이 primary stem만 사용하는지 여부
        self.is_primary_model_primary_stem_only = is_primary_model_primary_stem_only
        
        # primary_model이 secondary stem만 사용하는지 여부
        self.is_primary_model_secondary_stem_only = is_primary_model_secondary_stem_only

          
    # MDX 모델 파일의 경로를 결정하고 설정
    def get_mdx_model_path(self):
        mdx_name_select_MAPPER = load_model_hash_data(MDX_MODEL_NAME_SELECT)
        if self.model_name.endswith(CKPT):
            # self.chunks = 0
            # self.is_mdx_batch_mode = True
            self.is_mdx_ckpt = True
            
        ext = '' if self.is_mdx_ckpt else ONNX
        
        for file_name, chosen_mdx_model in mdx_name_select_MAPPER.items():
            if self.model_name in chosen_mdx_model:
                self.model_path = os.path.join(MDX_MODELS_DIR, f"{file_name}{ext}")
                break
        else:
            self.model_path = os.path.join(MDX_MODELS_DIR, f"{self.model_name}{ext}")
            
        self.mixer_path = os.path.join(MDX_MODELS_DIR, f"mixer_val.ckpt")


    # 모델의 데이터 및 설정 정보를 가져오거나 팝업을 통해 사용자로부터 정보를 입력 받는 역할
    def get_model_data(self, model_hash_dir, hash_mapper):
        model_settings_json = os.path.join(model_hash_dir, "{}.json".format(self.model_hash))

        if os.path.isfile(model_settings_json):
            return json.load(open(model_settings_json))
        else:
            for hash, settings in hash_mapper.items():
                if self.model_hash in hash:
                    return settings
            else:
                return self.get_model_data_from_popup()

    #  Demucs 모델 파일의 경로를 결정하고 설정
    def get_demucs_model_path(self):
        demucs_name_select_MAPPER = json.load(urllib.request.urlopen(DEMUCS_MODEL_NAME_DATA_LINK))
        demucs_newer = [True for x in DEMUCS_NEWER_TAGS if x in self.model_name]
        demucs_model_dir = DEMUCS_NEWER_REPO_DIR if demucs_newer else DEMUCS_MODELS_DIR
        
        for file_name, chosen_model in demucs_name_select_MAPPER.items():
            if self.model_name in chosen_model:
                self.model_path = os.path.join(demucs_model_dir, file_name)
                break
        else:
            self.model_path = os.path.join(DEMUCS_NEWER_REPO_DIR, f'{self.model_name}.yaml')

    # Demucs 모델의 데이터 및 설정 정보를 결정하고 설정
    def get_demucs_model_data(self):
        self.demucs_version = DEMUCS_V4

        for key, value in DEMUCS_VERSION_MAPPER.items():
            if value in self.model_name:
                self.demucs_version = key

        self.demucs_source_list = DEMUCS_2_SOURCE if DEMUCS_UVR_MODEL in self.model_name else DEMUCS_4_SOURCE
        self.demucs_source_map = DEMUCS_2_SOURCE_MAPPER if DEMUCS_UVR_MODEL in self.model_name else DEMUCS_4_SOURCE_MAPPER
        self.demucs_stem_count = 2 if DEMUCS_UVR_MODEL in self.model_name else 4
        
        if not self.is_ensemble_mode:
            self.primary_stem = PRIMARY_STEM if self.demucs_stems == ALL_STEMS else self.demucs_stems
            self.secondary_stem = STEM_PAIR_MAPPER[self.primary_stem]


    # 모델 파일의 해시 값을 계산하거나 검색
    def get_model_hash(self):
        self.model_hash = None
        
        if not os.path.isfile(self.model_path):
            self.model_status = False
            self.model_hash is None
        else:
            if model_hash_table:
                for (key, value) in model_hash_table.items():
                    if self.model_path == key:
                        self.model_hash = value
                        break
                    
            if not self.model_hash:
                try:
                    with open(self.model_path, 'rb') as f:
                        f.seek(- 10000 * 1024, 2)
                        self.model_hash = hashlib.md5(f.read()).hexdigest()
                except:
                    self.model_hash = hashlib.md5(open(self.model_path,'rb').read()).hexdigest()
                    
                table_entry = {self.model_path: self.model_hash}
                model_hash_table.update(table_entry)




def assemble_model_data(mode, model=None, arch_type=ENSEMBLE_MODE):
    if arch_type == ENSEMBLE_MODE and mode == 'pred':
        model_data: List[ModelData] = [ModelData(model_name) for model_name in pred_models]
    elif arch_type == ENSEMBLE_MODE and mode == 'train':
        model_data: List[ModelData] = [ModelData(model_name) for model_name in train_models]

    if arch_type == ENSEMBLE_CHECK:
        model_data: List[ModelData] = [ModelData(model)]

    if arch_type == VR_ARCH_TYPE or arch_type == VR_ARCH_PM:
        model_data: List[ModelData] = [ModelData(model, VR_ARCH_TYPE)]

    if arch_type == MDX_ARCH_TYPE:
        model_data: List[ModelData] = [ModelData(model, MDX_ARCH_TYPE)]

    if arch_type == DEMUCS_ARCH_TYPE:
        model_data: List[ModelData] = [ModelData(model, DEMUCS_ARCH_TYPE)] #

    return model_data


def cached_source_model_list_check(model_list: list[ModelData]):
    model: ModelData
    primary_model_names = lambda process_method:[model.model_basename if model.process_method == process_method else None for model in model_list]
    secondary_model_names = lambda process_method:[model.secondary_model.model_basename if model.is_secondary_model_activated and model.process_method == process_method else None for model in model_list]

    vr_primary_model_names = primary_model_names(VR_ARCH_TYPE)
    mdx_primary_model_names = primary_model_names(MDX_ARCH_TYPE)
    demucs_primary_model_names = primary_model_names(DEMUCS_ARCH_TYPE)
    vr_secondary_model_names = secondary_model_names(VR_ARCH_TYPE)
    mdx_secondary_model_names = secondary_model_names(MDX_ARCH_TYPE)
    demucs_secondary_model_names = [model.secondary_model.model_basename if model.is_secondary_model_activated and model.process_method == DEMUCS_ARCH_TYPE and not model.secondary_model is None else None for model in model_list]
    demucs_pre_proc_model_name = [model.pre_proc_model.model_basename if model.pre_proc_model else None for model in model_list]#list(dict.fromkeys())
    
    for model in model_list:
        if model.process_method == DEMUCS_ARCH_TYPE and model.is_demucs_4_stem_secondaries:
            if not model.is_4_stem_ensemble:
                demucs_secondary_model_names = model.secondary_model_4_stem_model_names_list
                break
            else:
                for i in model.secondary_model_4_stem_model_names_list:
                    demucs_secondary_model_names.append(i)
    
    all_models = vr_primary_model_names + mdx_primary_model_names + demucs_primary_model_names + vr_secondary_model_names + mdx_secondary_model_names + demucs_secondary_model_names + demucs_pre_proc_model_name
    
    return all_models



def cached_sources_clear():
    vr_cache_source_mapper = {}
    mdx_cache_source_mapper = {}
    demucs_cache_source_mapper = {}


def process_get_baseText(total_files, file_num):
    """Create the base text for the command widget"""
    
    text = 'File {file_num}/{total_files} '.format(file_num=file_num,
                                                total_files=total_files)
    
    return text


def verify_audio(audio_file, is_process=True):
    is_good = False
    error_data = ''
    
    if os.path.isfile(audio_file):
        try:
            librosa.load(audio_file, duration=3, mono=False, sr=44100)
            is_good = True
        except Exception as e:
            error_name = f'{type(e).__name__}'
            traceback_text = ''.join(traceback.format_tb(e.__traceback__))
            message = f'{error_name}: "{e}"\n{traceback_text}"'

    if is_process:
        return is_good
    else:
        return is_good, error_data
    

def return_ensemble_stems(pred_stem, is_primary=False): 
    """Grabs and returns the chosen ensemble stems."""
    ensemble_stem = pred_stem.partition("/")

    if is_primary:
        return ensemble_stem[0]
    else:
        return ensemble_stem[0], ensemble_stem[2]
    
def load_model_hash_data(dictionary):
    '''Get the model hash dictionary'''

    with open(dictionary) as d:
        data = d.read()

    return json.loads(data)


def process_determine_secondary_model(self, process_method, main_model_primary_stem, is_primary_stem_only=False, is_secondary_stem_only=False):
    """Obtains the correct secondary model data for conversion."""
    
    secondary_model_scale = None
    secondary_model = NO_MODEL
    
    if process_method == VR_ARCH_TYPE:
        secondary_model_vars = self.vr_secondary_model_vars
    if process_method == MDX_ARCH_TYPE:
        secondary_model_vars = self.mdx_secondary_model_vars
    if process_method == DEMUCS_ARCH_TYPE:
        secondary_model_vars = self.demucs_secondary_model_vars

    if main_model_primary_stem in [VOCAL_STEM, INST_STEM]:
        secondary_model = secondary_model_vars["voc_inst_secondary_model"]
        secondary_model_scale = secondary_model_vars["voc_inst_secondary_model_scale"].get()
    if main_model_primary_stem in [OTHER_STEM, NO_OTHER_STEM]:
        secondary_model = secondary_model_vars["other_secondary_model"]
        secondary_model_scale = secondary_model_vars["other_secondary_model_scale"].get()
    if main_model_primary_stem in [DRUM_STEM, NO_DRUM_STEM]:
        secondary_model = secondary_model_vars["drums_secondary_model"]
        secondary_model_scale = secondary_model_vars["drums_secondary_model_scale"].get()
    if main_model_primary_stem in [BASS_STEM, NO_BASS_STEM]:
        secondary_model = secondary_model_vars["bass_secondary_model"]
        secondary_model_scale = secondary_model_vars["bass_secondary_model_scale"].get()

    if secondary_model_scale:
        secondary_model_scale = float(secondary_model_scale)

    if not secondary_model.get() == NO_MODEL:
        secondary_model = ModelData(secondary_model.get(), 
                                    is_secondary_model=True, 
                                    primary_model_primary_stem=main_model_primary_stem, 
                                    is_primary_model_primary_stem_only=is_primary_stem_only, 
                                    is_primary_model_secondary_stem_only=is_secondary_stem_only)
        if not secondary_model.model_status:
            secondary_model = None
    else:
        secondary_model = None
        
    return secondary_model, secondary_model_scale


def process_update_progress(model_count, total_files, iteration, step: float = 1):
    """Calculate the progress for the progress widget in the GUI"""
    
    total_count = model_count * total_files
    base = (100 / total_count)
    progress = base * iteration - base
    progress += base * step
    print(f'Process Progress: {int(progress)}%')


def process_iteration():
    iteration = iteration + 1

def cached_model_source_holder(process_method, sources, model_name=None):
    if process_method == VR_ARCH_TYPE:
        vr_cache_source_mapper = {**vr_cache_source_mapper, **{model_name: sources}}
    if process_method == MDX_ARCH_TYPE:
        mdx_cache_source_mapper = {**mdx_cache_source_mapper, **{model_name: sources}}
    if process_method == DEMUCS_ARCH_TYPE:
        demucs_cache_source_mapper = {**demucs_cache_source_mapper, **{model_name: sources}}
                             
def cached_source_callback(process_method, model_name=None):
    model, sources = None, None
    
    if process_method == VR_ARCH_TYPE:
        mapper = vr_cache_source_mapper
    if process_method == MDX_ARCH_TYPE:
        mapper = mdx_cache_source_mapper
    if process_method == DEMUCS_ARCH_TYPE:
        mapper = demucs_cache_source_mapper
    
    for key, value in mapper.items():
        if model_name in key:
            model = key
            sources = value
    
    return model, sources


def determine_auto_chunks(chunks, gpu):
    """Determines appropriate chunk size based on user computer specs"""
    
    if OPERATING_SYSTEM == 'Darwin':
        gpu = -1

    if chunks == BATCH_MODE:
        chunks = 0
        #self.chunks_var.set(AUTO_SELECT)

    if chunks == 'Full':
        chunk_set = 0
    elif chunks == 'Auto':
        if gpu == 0:
            gpu_mem = round(torch.cuda.get_device_properties(0).total_memory/1.074e+9)
            if gpu_mem <= int(6):
                chunk_set = int(5)
            if gpu_mem in [7, 8, 9, 10, 11, 12, 13, 14, 15]:
                chunk_set = int(10)
            if gpu_mem >= int(16):
                chunk_set = int(40)
        if gpu == -1:
            sys_mem = psutil.virtual_memory().total >> 30
            if sys_mem <= int(4):
                chunk_set = int(1)
            if sys_mem in [5, 6, 7, 8]:
                chunk_set = int(10)
            if sys_mem in [9, 10, 11, 12, 13, 14, 15, 16]:
                chunk_set = int(25)
            if sys_mem >= int(17):
                chunk_set = int(60) 
    elif chunks == '0':
        chunk_set = 0
    else:
        chunk_set = int(chunks)
                
    return chunk_set







# mode = "pred"
# print("default_batch_size:", default_batch_size)


# start_time = time.time()

# input_path = ("D:/User/user/uvr/pred/가시.mp3", "D:/User/user/uvr/pred/야생화.mp3",)
# print("1번째 추론을 시작합니다.")
# UVR.separate_process(mode, UVR.ENSEMBLE_MODE, input_path, export_path1)


# input_path2 = []
# for i in range(len(input_path)):
#     get_file_name = input_path[i].split('/')[-1]
#     get_file_name = get_file_name.split('.')[0]
#     input_path2.append(f"{export_path1}\\{i+1}_{get_file_name}_(Vocals).wav")

# input_path2 = tuple(input_path2)
# model_var = 'UVR-MDX-NET Karaoke 2'
# print("2번째 추론을 시작합니다.")
# separate_process(mode, MDX_ARCH_TYPE, input_path2, export_path2, model_var)

# input_path3 = []
# for i in range(len(input_path)):
#     get_file_name = input_path[i].split('/')[-1]
#     get_file_name = get_file_name.split('.')[0]
#     input_path3.append(f"{export_path2}\\{i+1}_{i+1}_{get_file_name}_(Vocals)_(Vocals).wav")

# input_path3 = tuple(input_path3)
# model_var = 'Reverb HQ'
# print("마지막 추론을 시작합니다.")
# separate_process(mode, MDX_ARCH_TYPE, input_path3, export_result, model_var)

# end_time = time.time()
# print(f"프로세스 종료! 총 {end_time - start_time}초 소요되었습니다.")