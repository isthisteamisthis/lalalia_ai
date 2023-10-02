<div align="center">
  
## 💽 랄라리아 : LALALIA (AI)
<img width="330" alt="image" src="https://github.com/isthisteamisthis/.github/assets/119282494/8e02f14a-df51-469b-ae4c-01a76b61154a">
<br>

[![Hits](https://hits.seeyoufarm.com/api/count/incr/badge.svg?url=https%3A%2F%2Fgithub.com%2Fcca-ffodregamdi%2Frunning-hi-back&count_bg=%23FFA49F&title_bg=%24555555&icon=&icon_color=%24E7E7E7&title=views&edge_flat=false)](https://hits.seeyoufarm.com)

</div>
<br>

<br>

### 🎤 프로젝트, LALALIA
>  **직접 제작한 곡이 세상에 불려지길 원하는 당신을 위해,** <br>
>  **목소리를 세상에 알리고 싶은 당신을 위해 준비했습니다.** <br>
>  **랄라리아는 커뮤니티 기반의 커버 노래 공유 & 매칭 플랫폼입니다.**

<br>
<br>


### 🎤 주요 기능 소개
> #### 1️⃣ **본인의 음역대를 찾아주는 기능**
> **가수 지망생**으로 어플에 진입하면, 본인이 낼 수 있는 명확한 음역대에 맞는 노래를 추천받을 수 있습니다. <br>
> **작곡가**로 어플에 진입하게 되면, 본인이 작곡한 노래의 음역대를 명시하여 본인의 노래를 불러줄 가수 지망생을 찾을 수 있습니다.

<br>

> #### 2️⃣ **뮤직 스코어 기능**
> 뮤직 스코어 기능을 통해 가수 지망생은 해당 노래의 가수의 스타일에 더욱 근접할 수 있도록 연습할 수 있습니다.

<br>

> #### 3️⃣ **AI 커버 기능**
> 가수 지망생은, 노래 실력이 성장한 자신을 직접 마주할 수 있으며, <br>
> 작곡가는 유명인의 목소리를 합성해 들어봄으로서 본인의 노래에 어울리는 음색을 찾을 수 있습니다. <br>

<br>

<br>

# 👥 AI 개발 팀원 소개 

<table>
  <tr>
    <td align="center"><a href=""><img src="https://avatars.githubusercontent.com/llleeeso" width="150px;" alt="">
    <td align="center"><a href=""><img src="https://avatars.githubusercontent.com/wohoman" width="150px;" alt="">
    <td align="center"><a href="https://github.com/MinSooC"><img src="https://avatars.githubusercontent.com/MinSooC" width="150px;" alt="">
    </td>
  </tr>
  <tr>
    <td align="center"><strong>AI</strong></td>
    <td align="center"><strong>AI</strong></td>
    <td align="center"><strong>AI</strong></td>
  </tr>
      
  <tr>
    <td align="center"><a href="https://github.com/llleeeso"><b>이소영</b></td>
    <td align="center"><a href="https://github.com/wohoman"><b>여형구</b></td>
    <td align="center"><a href="https://github.com/MinSooC"><b>차민수</b></td>
  </tr>

  <tr>
    <td align="center">뮤직 스코어 점수 기능 구현</td>
    <td align="center">음역대 측정 및 음악 추천 기능 구현</td>
    <td align="center">AI 데모곡 제작 및 flask 통신 기능 구현</td>
  </tr>
</table>
<br>

<br>

# 프로젝트 진행 기록

### 기간
- 2023.09.06 ~ 2023.09.26

### 세부내용
- 23.09.06 : 주제 선정 및 발표 PPT 제작
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
<img width="250" alt="image" src="https://github.com/isthisteamisthis/.github/assets/119282494/86a15008-ad16-407d-80b2-aacb593f1c9d">
<img width="250" alt="image" src="https://github.com/isthisteamisthis/.github/assets/119282494/431e2f38-fd51-4d05-8b12-d9d5c3a01ee8">
<br>

    - 서버로부터 클라이언트에서 측정한 최고음과 최저음을 전송받아 NOTE_FREQUENCIES에 따른 음역대를 추출
    - 추출한 음역대를 바탕으로 sqlite db에서 음역대에 가장 많이 겹쳐있는 노래를 랜덤하게 3개 추천
    - db에 저장된 노래들 역시 전부 최고음과 최저음 음역대를 추출

### 🎤 노래방 점수 측정 기능
    - 서버에서 사용자의 녹음과 원본 음원을 전송받으면 원본 음원을 보컬과 백그라운드로 분리해 보컬의 pitch 분석
    - 사용자의 녹음과 보컬의 pitch를 비교해 기준범위(2**(1/12))에 따라 음의 차이를 분석 후 점수화

### 🎤 AI Cover 제작
    - 서버에서 변환할 노래, 모델명, 옥타브, 참조율을 받아 변환 작업을 실행
    - 변환할 노래의 보컬과 백그라운드를 분리 후, 옥타브와 참조율을 적용한 모델을 이용해 보컬을 변환
    - 변환한 보컬과 분리했던 백그라운드 합친 후 서버로 리턴
    - 목소리 모델의 경우 서버에서 사용자의 녹음과 원하는 epoch에 따라 train 진행

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
