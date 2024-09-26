<!-- Please write a description for your implementation -->

사용하는 모델 크기가 약 30GB로, 그 이상의 여유공간이 있어야 한다.  
메모리의 경우도 약 15GB정도 여유가 있으면 좋다.

### for window user *Require conda

```
conda create -n {your_env_name} python=3.10
```
Python version 3.10으로 세팅한 가상환경 생성 후,
```
pip install git+https://github.com/haotian-liu/LLaVA.git@786aa6a19ea10edc6f574ad2e16276974e9aaa3a
pip install torch==2.0.1+cu117 torchvision==0.15.2+cu117 torchaudio==2.0.2 --index-url https://download.pytorch.org/whl/cu117
pip install gradio==4.16.0 gradio_client==0.8.1
pip uninstall bitsandbytes --yes
pip install https://github.com/jllllll/bitsandbytes-windows-webui/releases/download/wheels/bitsandbytes-0.41.0-py3-none-win_amd64.whl
```
다음과 같이 패키지를 설치한다.
그 이후 ```python app.py```를 통해 app을 실행한다.  

조금 낮은 버전을 사용하는 이유 - windows 환경에서 테스트를 진행하고 있는데[(windows 사용법 출처)](https://github.com/haotian-liu/LLaVA/blob/main/docs/Windows.md), 가장 최신 버전의 LLaVA는 2.0.1을 지원하지 않고 있어서 다음과 같이 다른 버전을 사용한다. 

---
### for docker user
```
docker build -t {docker_container_name} .
docker run --gpus=all {docker_container_name}
```

Local환경에서 확인할 때, docker build 실행 시 메모리 오류가 발생하는 것을 확인했습니다.  
from_pretrained에서 ValueError가 발생하는 경우, 다음과 같이 docker build 및 app.py 실행을 진행하면 좋습니다.  
Dockerfile에서 ```RUN python3 app.py```를 주석처리한 뒤 build를 진행합니다.  
※기본 제공되는 Dockerfile은 주석처리되어있습니다.  
```
docker build -t {docker_container_name} .
docker run --gpus=all -it -p 7860:7860 {docker_container_name}
python3 app.py
```
다음과 같이 진행하면 문제없이 실행할 수 있습니다.  
혹여 접속에 문제가 발생한다면, share=True 조건을 사전에 세팅해두었기 때문에  
제공되는 public URL로 접속하면 됩니다. ※인터넷 연결 필수

## LLaVA와 맞지 않는 Gradio를 사용하는 이유
1. 버전이 맞지 않아도, 현재 상황에서 제공하는 기능이 제대로 작동함을 확인
2. Nota에서 제공해준 template를 오류발생 없이 사용하기 위함

🔍 Find Defect! 버튼은 동작하지 않습니다.  

## 실험 진행 환경
### For windows and docker:
CPU: Intel i7-12700KF  
Memory: 32GB  
GPU: RTX 3070 Ti  
CUDA: 11.8  
torch: 2.0.1  
gradio: 3.35.2  
Python: 3.10.14  

from docker file: nvidia/cuda:11.8.0-base-ubuntu22.04  
Linux Ubuntu: 22.04  
Python: 3.10.12


---
### 제출 가이드
1. 도커를 사용하여 환경 설정하기. 이후 도커파일 제출하기.  
2. app.py를 수행하여 Gradio demo 실행하기(도커로 해결 가능)  
3. README.md를 업데이트 하기  

### 기본 요구사항  
> 요구사항: LLaVA-v1.5-7b모델을 Gradio를 이용하여 implementation 진행하기.  
아래와 같은 사항들을 만족해야 한다.  

1. 사용자가 Input으로 이미지와 요청하는 Text Prompt를 입력하면, text로 prompt에 대한 답을 해야한다.  
2. 사용자가 추가적으로 질문을 한다면, 최소 5번까지 이전 대화를 기억해야하며, 대화가 이어져야한다.  
3. 사용자는 언제든지 대화를 중단할 수 있어야 한다. --> 이 뜻이 조금 애매하다!  
4. 이미지와 대화는 application이 종료된 이후 어디에도 저장되면 안된다.

### 필수 요구사항
1. ultra-high resolution의 사진에서, 이 남자가 뭘 하고 있나요?  
2. 어떻게 VLM모델에서 일반적이지 않은 도메인으로 확장할 수 있을까요?


깔끔하게 요구사항만 달성할 수 있도록 코드 작성

# 요구사항 1에 대한 해결 방안  
우선 제공되는 이미지의 크기가 어마어마하다. 3000x1902.  
LLaVA에 입력되는 이미지는 336x336으로, 굉장히 작다.  
생각할 수 있는 방법중에 하나는 다음과 같다.
1. 해당 크기가 큰 이미지를 336x336보다 작거나 같게 적당히 자른다. 빈 공간은 패딩한다.
2. 이후, waldo의 이미지를 보여주며, 이 이미지가 있는 잘린 블럭을 알려달라고 한다.
3. 잘린 블럭에서 waldo가 어떤 행위를 하고 있는지 질의하여 해결한다.  
  
질의를 하는데 있어 기존 이미지 기록을 함께 제공하기 때문에, 나뉘어진 이미지 사이의 관계성을 최소화하기 위해 개별적으로 질문할 수 있도록 구성했다.  
## 자동 파이프라인:  
1. Where's Waldo? 버튼 클릭 시, 미리 save 되어있던 waldo 사진을 불러온다.
2. 자동으로 질의를 하나 남긴다. "This is Waldo. You must remember he's appearance. Just answer he's appearance."
3. 이후, Large-Scale의 이미지를 업로드 하면, 336x336보다 작은 사이즈로 이미지를 나눈다.
4. 나눈 이미지를 llava.img_tensor에 업로드 하고, 다음과 같은 질의를 남긴다
5. "Is Waldo in this picture? Must answer in format "answer" which is Yes or No."
6. 이후, Chatbot의 대답에 근거하여 개별적인 이미지에 대해 질의를 진행한다. 이때, 입력으로 제공되는 llava.img_tensor에는 Waldo 이미지와 찾은 이미지 2장만을 제공한다.
7. "First picture is appearance of Waldo. And second picture is the one you told me that there is a Waldo. Tell me what waldo is doing"
8. 7번 질의를 5번에서 찾은 image 만큼 반복한다.  
  
### 초기 prompt 3가지:  
1. "This is Waldo. You must remember he's appearance."
2. "Is Waldo in this picture? Must answer in format "answer" which is Yes or No."
3. "First picture is appearance of Waldo. And second picture is the one you told me that there is a Waldo. Tell me what waldo is doing"

### 최종 prompt 3가지:
1. This is Waldo. You must remember he's appearance. Just answer he's appearance in one sentense.
2. ~~Is Waldo in this picture? Must answer in format "answer" which is Yes or No.~~
3. Find Waldo. if not visible, dont Answer what Waldo is doing. if visible, Answer what Waldo is doing.

## 추천 Parameters:
- Temperature: 0.1
- Top P: 0.8


++추가사항: 이미지를 임의로 나누는 경우, waldo가 잘리는 경우가 발생할 수 있다.  
따라서 이미지를 sliding하며 탐색하는 경우 더 나은 결과를 얻을 수 있을것이다.  

++해보고 싶었으나 하지 못한 것: Waldo와 입력받는 이미지에 convolution을 통과한 뒤, convolution결과를 이미지로 변경하여 LLaVA에 제공하는 방법을 시도하고 싶었으나, 시간부족으로 진행하지 못했다.

# 요구사항 2에 대한 해결 방안 
제공된 이미지는 insulators 에 대한 이미지들이고, 대부분 확대되어있으며, 동일한 패턴이 반복되는 모습을 보인다!
1. 먼저 이미지에 dataset에서 전체적인 insulators의 방향을 탐색한다.
2. 수직/수평 적으로 이미지를 잘라서 LLaVA에 제공해준다.
3. 제공한 이미지들 중, 가장 다른 이미지가 어떤것인지 질의한다!
4. 이때, 이미지의 배경이 문제가 될 수 있기 때문에, 배경은 제거하는 과정을 거친다.
5. 이미지를 너무 잘게 자르거나, 너무 크게 자르면 문제가 생길 수 있다. 따라서 적당히 자르는 것이 중요해보인다!

하지만 LLaVA에 질의하는것이 아닌, Gradio에서 Application Pipeline을 만들어야 하는 것이므로, chatbot방법을 사용하지 않아도 된다!


# 작업 일지
> 04.25  
[✅] Docker에 GPU 연동하기  
[✅] Gradio로 웹서버 실행해보기  

> 04.26  
[✅] LLaVA실행해보기  
[😢] LLaVA를 Gradio로 데모 돌리기  

> 04.27  
[✅] gradio로 전체적인 front 만들기  
[😢] docker 수행 시, app.py의 gradio가 실행됨과 동시에 llava가 back으로 수행되기    
docker image를 생성한 뒤, 도커에서 LLaVA의 데모를 돌려보지 못했다.  
windows 환경에서는 제약이 많았고, sglang을 사용하지 못해 할 수 없는줄 알았으나, 다시금 방법을 찾아 demo를 실행중에 있다. 
demo 실행까지는 성공했으나, fast api로 data를 주고받는 과정에서 문제가 발생하는것 같았다.  
LLaVA_chatbot.ipynb에서는 잘 작동했던것을 생각하면, 전반적으로 다듬어야하지 않나..생각한다.  
요구사항에서 DB를 저장하지 말아달라고 했기 때문에, Back-front를 완벽하게 분할할 필요는 없다는 생각이 들었다.  

> 04.28  
[😢] gradio 마무리  
[😢] AI 서버 연동하여 결과 chatbot에 보여주기  
아... 1~2일간 동작하지 않던 데모의 문제점을 이제서야 파악했다. llava 버전의 차이로 인해서 계속 동작하지 않고있었다.  
버전관리를 올바르게 하지 못한 내 잘못이다.  
Gradio를 빠르게 마무리 짓고, 서버를 fast-api로 구성하여 빠르게 데모를 시연해보고자 한다.

> 04.29  
[✅] fast-api대신 app.py 내부에 모델을 바로 업로드  
[✅] 동작하는 최종 버전 확인 및 환경 재점검  

> 04.30
[✅] 기본 요구 사항 달성  
[✅]필수 요구사항에 대한 고찰.  
해당 문제를 해결할 수 있는 pipeline을 요구한 것이기 때문에, 해당 문제를 해결할 수 있는 추가적인 버튼을 만든다.  
버튼1: waldo button 버튼2: unusual domain

> 05.01  
[✅] Advanced Task 1 세부화  
[❓] Advanced Task 1 진행  
Waldo를 정확하게 찾지는 못하나, 어느정도 구분이 가능하도록 구성했다.  
![img](./IMG/task1.png)

> 05.02  
[✅] README에 적힌대로 conda 생성 후 실행 시 올바르게 동작하는지 확인  
[✅] README에 적힌대로 docker 생성 후 실행 시 올바르게 동작하는지 확인