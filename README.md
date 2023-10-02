# 프로젝트명

### 🎙️ 랄라리아 (메타버스 아카데미 2기 융합팀 9월 월말평가 팀 프로젝트)

# 팀원 소개 및 역할

### 👥 AI 개발 팀원
- 메타버스 아카데미 2기 AI반 차민수, 이소영, 여형구 총 3명

### 👥 역할 분담
- 주제 선정 : 모든 팀원 토의 및 강사/멘토님 협조

- 음역대 측정 및 노래 추천 : 여형구
- 노래방 점수 측정 : 이소영
- AI Cover 제작 : 차민수
- flask 서버 통신 : 모든 팀원
- 코드 병합 : 차민수

- AI 파트 PPT 작성 : 모든 팀원

# 프로젝트 진행 기록

### 기간
- 2023.09.04 ~ 2023.09.27

### 세부내용
- 23.09.04 ~ 23.09.06 : 주제 선정 및 발표 PPT 제작
- 23.09.07 : 주제 발표, 최고음/최저음 음역대 측정, 노래 주파수 추출, 음원의 Vocal과 MR 분리 작업
- 23.09.08 : 최고음/최저음 음역대 측정, 노래 주파수 추출, 음원의 Vocal과 MR 분리 작업, 음원 합성 코드 작업
- 23.09.09 ~ 23.09.10 : 음성 모델 훈련에 사용할 데이터 생성 및 수집
- 23.09.11 : 음역대 flask 통신 작업, DB에 음역대 측정한 노래 저장, 실시간 음정 분석 및 사용자 음성 받아오기, 음성 모델 훈련, 파트 별 더미데이터 생성
- 23.09.12 : 오디오 신호 분해, RVC Web UI 코드 분석
- 23.09.13 : 음정 추출, 사용자의 보컬 녹음, RVC Web UI 코드 분석
- 23.09.14 : 실시간 음정 분석, 박자 분석, RVC Web UI 코드 최적화
- 23.09.15 : 푸리에 변환을 이용한 성능 메트릭 및 점수 추출, RVC Web UI 코드 최적화 및 flask 서빙
- 23.09.16 ~ 23.09.17 : flask-spring 서버 통신 테스트, PPT 제작 및 시연영상 편집 
- 23.09.18 : 프로토타입 발표
- 23.09.19 : UVR 코드 분석
- 23.09.20 : UVR 코드 분석
- 23.09.21 : UVR 코드 최적화
- 23.09.22 : UVR 코드 최적화
- 23.09.23 : UVR 코드 최적화 및 flask 서빙
- 23.09.24 ~ 23.09.25 : PPT 제작 및 시연영상 촬영, 최종 코드 병합 후 테스트
- 23.09.26 : PPT 제작 및 시연영상 편집
- 23.09.27 : 최종 PT 발표

# 주요 내용

### 🎤 음역대 측정 기능
- 서버로부터 클라이언트에서 측정한 최고음과 최저음을 전송받아 NOTE_FREQUENCIES에 따른 음역대를 추출
- 추출한 음역대를 바탕으로 sqlite db에서 음역대에 가장 많이 겹쳐있는 노래를 랜덤하게 3개 추천
- db에 저장된 노래들 역시 전부 최고음과 최저음 음역대를 추출

### 🎤 노래방 점수 측정 기능
- 서버에서 사용자의 녹음과 원본 음원을 전송받으면 원본 음원을 보컬과 백그라운드로 분리해 보컬의 pitch 분석
- 사용자의 녹음과 보컬의 pitch를 비교해 기준범위(2**(1/12))에 따라 음의 차이를 분석 후 점수화

### 🎤 AI Cover 제작


# 기술 스택

### 언어
![Python](https://img.shields.io/badge/python-3776AB?style=flat&logo=python&logoColor=white)

### 주요 라이브러리
![FFmpeg](https://img.shields.io/badge/FFmpeg-007808?style=flat&logo=FFmpeg&logoColor=white)
![Flask](https://img.shields.io/badge/Flask-000000?style=flat&logo=Flask&logoColor=white)
![librosa](https://img.shields.io/badge/librosa-7f03a8?style=flat&logo=librosa&logoColor=white)
![Pytorch](https://img.shields.io/badge/Pytorch-EE4C2C?style=flat&logo=Pytorch&logoColor=white)
![spleeter](https://img.shields.io/badge/spleeter-0aa07b?style=flat&logo=spleeter&logoColor=white)
![scikitlearn](https://img.shields.io/badge/scikitlearn-F7931E?style=flat&logo=scikitlearn&logoColor=white)
![sqlite](https://img.shields.io/badge/sqlite-003857?style=flat&logo=sqlite&logoColor=white)

### 개발 툴
![visualstudiocode](https://img.shields.io/badge/visualstudiocode-007ACC?style=flat&logo=visualstudiocode&logoColor=white)

### 협업 툴
![Notion](https://img.shields.io/badge/notion-000000?style=flat&logo=notion&logoColor=white)
![Discord](https://img.shields.io/badge/discord-5865F2?style=flat&logo=discord&logoColor=white)

# 참고 자료
### 오픈소스(github)
- https://github.com/RVC-Project/Retrieval-based-Voice-Conversion-WebUI
- https://github.com/Anjok07/ultimatevocalremovergui
