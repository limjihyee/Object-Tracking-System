# Object-Tracking-System

**UAV-Assisted Moving Object Detection and Tracking Using Deep Reinforcement Learing  in Urban Environments**


## 🎥 simulation
[![유튜브 영상](https://img.youtube.com/vi/EVtdZAfZa-Q/0.jpg)](https://youtu.be/EVtdZAfZa-Q?si=5fJHvjpOULmeR-Ig)

---

## Introduction
- **무인항공기(UAV)의 활용 증가**
  - 물류, 재난 구조, 농업 등 다양한 산업 분야에서 활용도 급증
- **UAV의 주요 활용 분야**
  - 감시 및 정찰
  - 환경 모니터링
  - 군사적 활용
- **기존 방식의 한계**
  - Rule-based 객체 탐지 시스템 : 환경 변화에 유연한 대처 불가
  - 복잡한 환경에서 추적 성능 저하
- **딥러닝 기반 모델의 필요성**
  - 심층강화학습(Deep Reinforcement Learing) DQN 활용
  - 최적의 행동 학습 : 터널 진입, 장애물 회피 등 예외 상황 대응

---

## System Architecture
### Overall system architecture
<img width="799" height="505" alt="image" src="https://github.com/user-attachments/assets/a95dd259-e143-4800-a24a-e280ed14adf8" />

### Vehicle detection pipeline
<img width="601" height="243" alt="image" src="https://github.com/user-attachments/assets/57b3908e-dae1-4fbb-ab77-d188d7e6b55b" />

- **Self-Attection, Heatmap 기반 CNN Regression Object Detection model 생성**
  - Self-Attection Module : 각 픽셀 간 관계를 파악해 Enhanced Feature 추출
  - Heatmap 기반 접근 방식 : 객체 중심 위치의 가능성을 나타내는 공간적 맵인 Heatmap을 이용해 객체 중심 추적 모델 생성

### Vehicle tracking pipeline
<img width="608" height="421" alt="image" src="https://github.com/user-attachments/assets/13362b6b-59a7-409b-b78d-1f7ed34e9685" />

<img width="594" height="205" alt="image" src="https://github.com/user-attachments/assets/4f55f97f-cfef-4ee3-9faa-533f33b7f46e" />

- **강화학습(DQN) 기반 자율적인 Object Tracking system**
  - 환경과 Agent 간의 상호작용을 통해 이미지 입력을 받아 특징 추출하고 기타 정보를 FC Layer에 합하여 최종적으로 각 행동의 Q 값을 계산해 최적의 행동을 선택
 
  - **State**
    - CNN Layer
      - 4 step image
    - Fully Connected Layer
   
  - **Action**
  - **Reward**

---
##  실헙 결과
<img width="504" height="150" alt="image" src="https://github.com/user-attachments/assets/b48a3a42-73ea-461d-a9c3-b4dc03586be0" />

- Regression과 Heatmap의 적절한 조합 3:7에서 가장 낮은 픽셀 오차
- Heatmap + Regressiino 구조로 기존 회귀 단일 방식 대비 높은 예측 정확도 확보

<img width="435" height="297" alt="image" src="https://github.com/user-attachments/assets/eff6f0d0-e108-4e6a-b014-78dc29db5605" />


- 복잡한 기상 조건일수록 학습 난이도 증가
  - 안개로 인한 시야 흐림, 야간 조명 저하, 노이즈 증가 등으로 보상 수렴값이 상대적으로 낮게 형성
- 조도·기상 조건 변화는 학습 과정에 직·간접적인 영향
  - 초기 학습 속도 감소
- 모든 기상 조건에서 안정적인 수렴 달성모든 기상 조건에서 안정적인 수렴 달성
  - 충분한 에피소드 학습 후, 모든 환경에서 보상이 일정 수준 이상으로 수렴

---
## 결론
- 높은 검출 성능 유지
Self-Attention 모듈을 포함한 Heatmap+Regression 기반 CNN 검출기는 복잡한 도시 환경과 부분 가림 상황에서도 높은 검출 정확도를 안정적으로 유지함

- 강인한 추적 성능
통합 DQN 추적 시스템은 기존 Rule-based 방식 대비 탐지율 및 거리 유지 능력에서 우수한 성능을 보임

- 다양한 환경 적응력
고가도로, 터널, 야간, 안개 등 복잡한 환경 요소와 조도·기상 조건에서도 충분한 학습 후 보상이 일정 수준 이상으로 수렴, 다양한 환경에서 안정적인 추적 제어가 가능함을 실험적으로 입증함

---
## 팀원 소개

<table>
  <tr align="center">
    <td><img src="https://github.com.png" width="220"/></td>
    <td><img src="https://github.com/limjihyee.png" width="220"/></td>
    <td><img src="https://github.com/.png" width="220"/></td>
  </tr>
  <tr align="center">
    <td><a href="https://github.com/">박채원</a></td>
    <td><a href="https://github.com/limjihyee">임지혜</a></td>
    <td><a href="https://github.com/">이승준</a></td>
  </tr>
  <tr align="center">
    <td>Object Tracking 및 Unity 환경 구축</td>
    <td>Object Tracking 및 Unity 환경 구축</td>
    <td>Object Detection</td>
  </tr>
</table>

