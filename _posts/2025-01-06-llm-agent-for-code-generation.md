---
layout: post
title: LLM agent for code generation
tag: [Deep Learning, LLM, Agent, RAG, PEFT]
image: '/images/posts/2024-07-11-Dive-sequence-classification/Untitled.png'
---

# LLM agent for code generation

Model : [https://huggingface.co/donghuna/llama-plan-1B](https://huggingface.co/donghuna/llama-plan-1B)

Dataset : [https://huggingface.co/datasets/donghuna/StaQC-plan](https://huggingface.co/datasets/donghuna/StaQC-plan)

Github : [https://github.com/donghuna/PromptGenerate](https://github.com/donghuna/PromptGenerate)


## 배경

LLM의 지속적인 발전  
최신 대규모 언어 모델(LLM)은 점차 모델 크기를 늘리며 성능을 향상
OpenAI, Meta 등에서 최신 고성능 LLM들이 지속적이고 짧은 주기로 릴리즈

사내 제한
보안 문제로 인해 On-premise 방식의 네이버 HCX와 SR의 GAUSS모델을 사용
해당 모델들은 외부 최신 LLM 대비 성능 뒤처짐


## 사내 모델의 문제점

내부 모델의 성능 한계로 임직원 불만 증가
외부 최신 모델과의 성능 격차로 인해 지속적인 비교와 높은 성능 요구 발생


## 프로젝트 목표

제한된 성능의 모델로도 더 나은 결과를 도출하기 위해 최적화된 프롬프트 생성 모델 구현
효율적인 프롬프트 설계를 통해 내부 모델의 Code generation 품질을 향상시키는 방안 개발
실제 사용성을 고려하여, 모델 크기를 과도하게 키우지 않고, 양자화를 적용하여 응답 속도에 최적화된 모델로 구현


## 코드 생성 성능 향상을 위한 방법

기존 방식
현재는 사용자 프롬프트를 전처리 없이, 그대로 LLM에 입력하여 코드 생성을 수행
프롬프트에 구체적인 정보가 부족, 사용자 의도 반영이 되지 않아 코드 품질이 낮아질 가능성 존재
LLM이 명ㅎ왁한 컨텍스트 없이 작업을 수ㅐㅎㅇ하여 결과물의 일관성이 떨어짐

제안 방식
User prompt를 기반으로 유사 질문과 코드를 검색, 코드 생성을 위한 플랜 생성
Enhanced prompt를 통해 In-Context learning(ICL) 효과를 극대화 하여 모델의 코드 작성 이해도와 품질 향상
제한된 환경에서도 효율적이고 신뢰성 높은 코드 생성 기대


## Dataset

The labels for this project are based on the dataset provided by the [RESOUND Diving Dataset](http://www.svcl.ucsd.edu/projects/resound/dataset.html). This dataset includes a wide range of diving techniques categorized into different classes. The labels encompass various cases such as:

![Untitled](/images/posts/2024-07-11-Dive-sequence-classification/Untitled.png)

- Initial position (facing forward or backward)
- Direction of the dive (forward or backward)
- Body posture (Straight(A), Tucked(B), Pike(C))
- Number of rotations
- Number of twists

### Database Characteristics

The diving videos used for training and evaluation come from the RESOUND dataset. Key characteristics of this database include:

- **Number of Videos**: A substantial number of videos covering different diving techniques.  (Training dataset : 15027, Test dataset : 1970)
- **Resolution**: Videos are provided in high resolution to capture detailed movements.
- **Duration**: Each video clip is of sufficient length to encompass the entire dive from takeoff to entry into the water.

### Label

The dive ID is determined based on the following factors: Takeoff, Somersault, Twist, and Flight Position

![Untitled](/images/posts/2024-07-11-Dive-sequence-classification/Untitled%201.png)

| ID | Take off | Somersault | Twist | Flight position | Dive numbers | Label |
| --- | --- | --- | --- | --- | --- | --- |
| 0 | Forward | 2.5 | 1 | PIKE | 5152B | 21 |
| 1 | Forward | 2.5 | 2 | PIKE | 5154B | 22 |
| 2 | Forward | 2.5 | 3 | PIKE | 5156B | 23 |
| 3 | Forward | 2.5 | 0 | PIKE | 105B | 24 |
| 4 | Forward | 2.5 | 0 | TUCK | 105C | 25 |
| 5 | Forward | 4.5 | 0 | TUCK | 109C | 28 |
| 6 | Forward | 3.5 | 0 | PIKE | 107B | 26 |
| 7 | Forward | 3.5 | 0 | TUCK | 107C | 27 |
| 8 | Forward | 1.5 | 2 | FREE | 5134D | 18 |
| 9 | Forward | 1.5 | 1 | FREE | 5132D | 17 |
| 10 | Forward | 1.5 | 0 | PIKE | 103B | 19 |
| 11 | Forward | 1 | 0 | PIKE | 102B | 20 |
| 12 | Forward | Dive | 0 | PIKE | 101B | 29 |
| 13 | Forward | Dive | 0 | STR | 101A | 30 |
| 14 | Inward | 2.5 | 0 | PIKE | 405B | 33 |
| 15 | Inward | 2.5 | 0 | TUCK | 405C | 34 |
| 16 | Inward | 3.5 | 0 | TUCK | 407C | 35 |
| 17 | Inward | 1.5 | 0 | PIKE | 403B | 31 |
| 18 | Inward | 1.5 | 0 | TUCK | 403C | 32 |
| 19 | Inward | Dive | 0 | PIKE | 401B | 36 |
| 20 | Back | 2 | 1.5 | FREE | 5243D | 9 |
| 21 | Back | 2 | 2.5 | FREE | 5245D | 10 |
| 22 | Back | 2.5 | 1.5 | PIKE | 5253B | 5 |
| 23 | Back | 2.5 | 2.5 | PIKE | 5255B | 6 |
| 24 | Back | 2.5 | 0 | PIKE | 205B | 7 |
| 25 | Back | 2.5 | 0 | TUCK | 205C | 8 |
| 26 | Back | 3 | 0 | PIKE | 206B | 13 |
| 27 | Back | 3 | 0 | TUCK | 206C | 14 |
| 28 | Back | 3.5 | 0 | PIKE | 207B | 11 |
| 29 | Back | 3.5 | 0 | TUCK | 207C | 12 |
| 30 | Back | 1.5 | 1.5 | FREE | 5233D | 1 |
| 31 | Back | 1.5 | 2.5 | FREE | 5235D | 2 |
| 32 | Back | 1.5 | 0.5 | FREE | 5231D | 0 |
| 33 | Back | 1.5 | 0 | PIKE | 203B | 3 |
| 34 | Back | 1.5 | 0 | TUCK | 203C | 4 |
| 35 | Back | Dive | 0 | PIKE | 201B | 15 |
| 36 | Back | Dive | 0 | TUCK | 201C | 16 |
| 37 | Reverse | 2.5 | 1.5 | PIKE | 5353B | 42 |
| 38 | Reverse | 2.5 | 0 | PIKE | 305B | 43 |
| 39 | Reverse | 2.5 | 0 | TUCK | 305C | 44 |
| 40 | Reverse | 3.5 | 0 | TUCK | 307C | 45 |
| 41 | Reverse | 1.5 | 1.5 | FREE | 5333D | 38 |
| 42 | Reverse | 1.5 | 3.5 | FREE | 5337D | 40 |
| 43 | Reverse | 1.5 | 2.5 | FREE | 5335D | 39 |
| 44 | Reverse | 1.5 | 0.5 | FREE | 5331D | 37 |
| 45 | Reverse | 1.5 | 0 | PIKE | 303B | 41 |
| 46 | Reverse | Dive | 0 | PIKE | 301B | 46 |
| 47 | Reverse | Dive | 0 | TUCK | 301C | 47 |

### How to Identify Dive Numbers

Dives are described by their full name (e.g. reverse 3 1/2 somersault with 1/2 twist) or by their numerical identification (e.g. 5371D), or “dive number.”

**Specific dive numbers are not random. They are created by using these guidelines:**

1. All dives are identified by three or four digits and one letter. Twisting dives utilize four numerical digits, while all other dives use three.

2. The first digit indicates the dive’s group: 1 = forward, 2 = back, 3 = reverse, 4 = inward, 5 = twisting, 6 = armstand.

3. In front, back, reverse, and inward dives, a ‘1’ as the second digit indicates a flying action. A ‘0’ indicates none. In twisting and armstand dives, the second digit indicates the dive’s group (forward, back, reverse).

4. The third digit indicates the number of half somersaults.

5. The fourth digit, if applicable, indicates the number of half twists.

6. The letter indicates body position: A = straight, B = pike, C = tuck, D = free.

**Examples:**

107B = Forward dive with 3 1/2 somersaults in a pike position

305C = Reverse dive with 2 1/2 somersaults in a tuck position

5253B = Back dive with 2 1/2 somersaults and 1 1/2 twists in a pike position

### Dive number counts from training data

![Untitled](/images/posts/2024-07-11-Dive-sequence-classification/Untitled%202.png)

## Dataloader Configuration

To effectively train the model, the dataloader is configured with specific preprocessing steps:

- **Image Preprocessing**: Videos are resized and normalized to ensure consistency across the dataset. (224 x 224)
- **Frame Extraction**: A fixed number of frames are extracted from each video to represent the sequence. As shown in the table below, increasing the number of input frames results in higher accuracy, but due to resource constraints, 24 frames were used.

![Untitled](/images/posts/2024-07-11-Dive-sequence-classification/Untitled%203.png)

- **Batch Size**: The batch size is set to optimize memory usage and training time. Due to memory constraints, each batch included 2 videos.
- **Shuffling**: Data shuffling is implemented to ensure a diverse mix of training samples in each batch. Random shuffling was applied.

## Transformer Model Explanation

effectiveness in video understanding. Here’s a brief overview of its components:

- **Self-Attention Mechanism**: Timesformer uses self-attention to capture dependencies between different frames in a video. This allows the model to understand the temporal dynamics of the diving sequence.

![Untitled](/images/posts/2024-07-11-Dive-sequence-classification/Untitled%204.png)

- **Positional Encoding**: To maintain the sequential information of frames, positional encodings are added to the frame embeddings.

![Untitled](/images/posts/2024-07-11-Dive-sequence-classification/Untitled%205.png)

- **Layers and Heads**: The model consists of multiple layers of attention heads that allow it to focus on different parts of the sequence simultaneously.
- **Loss Function**: Cross-entropy loss is used for classification, optimizing the model to predict the correct dive class.
- **Optimization**: Adam optimizer with a carefully tuned learning rate is used to train the model.

### Training and Results

- **Accuracy**: The model achieves high accuracy in classifying various diving techniques.

```
Test Accuracy: 0.5208
```

- **Loss**: The training and validation loss curves indicate good convergence. Performing a few more epochs can potentially improve accuracy.

```
Epoch [1/3], Loss: 2.5981627, Train Accuracy: 0.2734
Validation Loss: 1.5747785, Validation Accuracy: 0.4957

Epoch [2/3], Loss: 1.2904986, Train Accuracy: 0.6159
Validation Loss: 0.8370838, Validation Accuracy: 0.7545

Epoch [3/3], Loss: 0.7121527, Train Accuracy: 0.7798
Validation Loss: 0.6011413, Validation Accuracy: 0.8230
```

![Untitled](/images/posts/2024-07-11-Dive-sequence-classification/Untitled%206.png)

- **Class-wise Performance (Confusion Matrix)**:

![Untitled](/images/posts/2024-07-11-Dive-sequence-classification/Untitled%207.png)

Labels with limited training data may have lower prediction accuracy.

## Architecture

![Untitled](/images/posts/2024-07-11-Dive-sequence-classification/Untitled%208.png)

## Conclusion

The developed Timesformer-based model successfully classifies diving techniques from video inputs. This project demonstrates the potential of Transformer models in video classification tasks and pr
