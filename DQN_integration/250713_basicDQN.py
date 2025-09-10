import os
import numpy as np #"Numeric Python"의 약자, 대규모 다차원 배열과 행렬 연산에 필요한 다양한 함수와 메소드를 제공
import cupy as cp #NumPy 문법을 사용하며 NVIDIA GPU를 사용하여 행렬 연산으로 속도를 향상
import pandas as pd #파이썬 데이터 분석 라이브러리 
import tensorflow as tf
import torch #PyTorch의 핵심 모듈, 텐서연산과 자동미분 등 제공, GPU가속 지원원
import torch.nn as nn #신경망 레이어와 관련된 클래스 및 함수 포함
import torch.nn.functional as F #활성화 함수 등 포함
import torch.optim as optim #최적화 알고리즘 제공
import cv2
import random
import datetime
import math
import wandb
import sys 

import torch.cuda as cuda #PyTorch에서 CUDA를 활용한 GPU 연산을 수행하는 모듈
import torch.backends.cudnn as cudnn #NVIDIA의 CuDNN (CUDA Deep Neural Network) 백엔드를 사용하여 CNN 연산을 최적화

import matplotlib.pyplot as plt #데이터 및 학습 과정을 시각화하는 라이브러리
from skimage.transform import resize # 이미지를 특정 크기로 리사이징 (강화학습 환경에서 입력 크기를 맞추는 데 사용)
from skimage.color import rgb2gray # RGB 이미지를 그레이스케일 변환 
from collections import deque # 빠른 큐(Queue) 연산을 위한 자료구조

import gc #Python의 가비지 컬렉터 (메모리 관리 용도)

from UAV_env import UAV_env #사용자 정의 환경 \ 자체 가상 환경
from Nstep_Buffer import n_step_buffer # N-step 경험 재생 버퍼

import tensorflow_model_optimization as tfmot

# detection def-----------------------------------------------
'''
def apply_clahe(image):
    lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    l = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8)).apply(l)
    return cv2.cvtColor(cv2.merge((l, a, b)), cv2.COLOR_LAB2BGR)

def apply_median_filter(image):
    return cv2.medianBlur(image, 3)

def sharpen_image(image):
    blurred = cv2.GaussianBlur(image, (0,0), 5)
    return cv2.addWeighted(image, 1.5, blurred, -0.5, 0)

def preprocess_image(image, target_size=(256, 256)):
    #image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    image = cv2.resize(image, target_size)
    image = apply_clahe(image)
    image = apply_median_filter(image)
    image = sharpen_image(image)
    image = image.astype(np.float32) / 255.0
    return np.expand_dims(image, axis=0)

def decode_predictions(heatmap, regression, top_k=1):
    heatmap = tf.convert_to_tensor(heatmap, dtype=tf.float32)
    pooled = tf.squeeze(
        tf.nn.max_pool2d(tf.expand_dims(heatmap,0), ksize=3, strides=1, padding='SAME'),
        axis=0
    )
    mask = tf.equal(heatmap, pooled)
    coords = tf.where(mask)
    scores = tf.gather_nd(heatmap, coords)
    top = tf.math.top_k(scores, k=top_k)
    sel = tf.cast(tf.gather(coords, top.indices)[:,:2], tf.float32)
    reg = tf.convert_to_tensor(regression, dtype=tf.float32)
    offsets = tf.repeat(tf.reshape(reg, (1,2)), top_k, axis=0)
    final = (tf.stack([sel[:,1], sel[:,0]], axis=1) + offsets) \
            / [heatmap.shape[1], heatmap.shape[0]]
    return final.numpy()[0]

def predict_from_frame(frame, model):
    inp = preprocess_image(frame)
    heatmap, regression = model.predict(inp, verbose=0)
    return decode_predictions(heatmap[0], regression[0])

def visualize_prediction(frame, norm_xy, color=(255,0,0), radius=1, thickness=-1):
    h, w = frame.shape[:2]
    x_px = int(norm_xy[0] * w)
    y_px = int(norm_xy[1] * h)
    vis = frame.copy()
    cv2.circle(vis, (x_px, y_px), radius, color, thickness)

    return vis
'''
# 경량화 detection -------
# --- 간단 전처리 (리사이즈 + 정규화) ---
def preprocess_image(image, target_size=(256, 256)):
    rgb     = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    resized = cv2.resize(rgb, target_size)
    return np.expand_dims(resized.astype(np.float32) / 255.0, axis=0)

# --- 디코딩 (좌표 + score) ---
def decode_predictions(heatmap, regression, top_k=1):
    heatmap = tf.convert_to_tensor(heatmap, tf.float32)
    pooled  = tf.squeeze(
        tf.nn.max_pool2d(tf.expand_dims(heatmap,0),
                         ksize=3, strides=1, padding='SAME'),
        axis=0
    )
    mask    = tf.equal(heatmap, pooled)
    coords  = tf.where(mask)
    scores  = tf.gather_nd(heatmap, coords)
    topk    = tf.math.top_k(scores, k=top_k)
    best_i  = topk.indices[0]
    best_score = topk.values[0]

    sel = tf.cast(coords[best_i][:2], tf.float32)
    reg = tf.convert_to_tensor(regression, tf.float32)
    h, w = heatmap.shape[:2]
    norm_xy = ((tf.stack([sel[1], sel[0]]) + reg)
               / tf.constant([w, h], dtype=tf.float32))
    
    norm_xy = np.clip(norm_xy, 0.0, 1.0)

    return norm_xy, float(best_score)

# --- 한 프레임당 예측 ---
def predict_from_frame(frame, model):
    inp                 = preprocess_image(frame)
    heatmap, regression = model.predict(inp, verbose=0)
    return decode_predictions(heatmap[0], regression[0])

##########################################################################################################################    

load = False # 모델을 불러올지 여부 False의 경우 새로운 모델 학습
date_time = datetime.datetime.now().strftime("%Y%m%d%H%M%S") # 현재 날짜와 시간을 문자열로 변환
save_path = f"./best_model/{date_time}.pkl" # 모델 저장 경로
load_path = f"./best_model/obstacle.pkl" # 불러올 모델 경로로
project_exe = "../build/build_final/project_250630_Sunny_integration.exe" #실행 프로젝트 파일  --> 원래 거
#project_exe = "../build/build_final/project_250724_mini.exe" # mini 실행 프로젝트 파일

# wandb 에서 실험 추적 및 시각화 
wandb.init(
    project="Tracking_uav",
    entity="mnl431",
    config={
        "architecture": "DQN",
    }
)

class Q_network(nn.Module):
    def __init__(self, num_actions):
        super(Q_network, self).__init__()
        self.num_actions = num_actions
        self.image_cnn = nn.Sequential(
            # nn.Conv2d(입력채널 수, 출력 채널수, kernel_size=필터 크기, stride=필터 이동간격)
            # 64x64x4 -> 30x30x32
            nn.Conv2d(4, 32, kernel_size=6, stride=2, groups=1, bias=True),
            nn.GELU(),
            # 30x30x32  -> 13x13x64
            nn.Conv2d(32, 64, kernel_size=6, stride=2, groups=1, bias=True),
            nn.GELU(),
            # 13x13x64 -> 10x10x64
            nn.Conv2d(64, 64, kernel_size=4, stride=1, groups=1, bias=True),
            nn.GELU(),
            # 10x10x64 -> 7x7x64
            nn.Conv2d(64, 64, kernel_size=4, stride=1, groups=1, bias=True),
            nn.GELU(),
            # 7x7x64 -> 5x5x64
            nn.Conv2d(64, 64, kernel_size=3, stride=1, groups=1, bias=True)
            # 1600
        )

        self.ray_fc = nn.Sequential(
            nn.Linear(30, 16),
            nn.GELU(),
            nn.Linear(16, 16),
            nn.GELU(),
            nn.Linear(16, 32)
            # 32
        )

        self.fc_connected = nn.Sequential(
            nn.Linear(1632, 512),
            nn.GELU(),
            nn.Linear(512, 256),
            nn.GELU(),
            nn.Linear(256, 128),
            nn.GELU(),
            nn.Linear(128, self.num_actions)
        )

        self.init_weights(self.image_cnn)
        self.init_weights(self.ray_fc)
        self.init_weights(self.fc_connected)

    # 가중치 초기화
    def init_weights(self, m):
        if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
            nn.init.kaiming_uniform_(m.weight, mode='fan_in', nonlinearity='leaky_relu')
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)

    # Q-value 출력 (CNN+FCLayer) 
    #UAV 카메라이미지 + UAV ray 센서(9) + info 추가 환경 정보(21)
    def forward(self, camera, ray, info):
        batch = camera.size(0)  # 0의 크기를 반환하기 때문에 batch는 1이 됨
        image_fcinput = self.image_cnn(camera).view(batch, -1) #이미지 cnn 처리
        combined_input = torch.cat([ray, info], dim=2)  # # ray(9개)와 info(21개)를 합쳐 30개짜리 벡터로 만듦
        ray_fcinput = self.ray_fc(combined_input).view(batch, -1)
        
        x = torch.cat([image_fcinput, ray_fcinput], dim=1)
        Q_values = self.fc_connected(x)

        return Q_values
 

class Agent:
    def __init__(self):
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu") # GPU사용/없으면 CPU사용
        self.device = device 

        #vehicle_name = "RedCar"
        
        model_path = r"D:/cnn_Integration/CNN_detection/models/0813.h5"
        
        # QuantizeScope 내에서 모델 로드
        with tfmot.quantization.keras.quantize_scope():
            self.detection_model = tf.keras.models.load_model(
                model_path,
                compile=False
            )

        self.learning_rate = 0.00002  # 0.00002
        self.batch_size = 32 # 학습할 때 한번에 샘플링할 데이터 개수
        self.gamma = 0.95  # 0.95 보상에서 현재 가치 -> 95% 반영
        self.n_step = 1  # 2
        self.num_actions =  15

        self.epsilon = 1 # 초기에 100% 확률로 랜덤 선택 
        self.initial_epsilon = 1.0
        self.epsilon_decay_rate = 0.8 # 
        self.final_epsilon = 0.1  # 최종 값
        self.epsilon_decay_period = 1000000  # 100000 #(231126) 감쇠가 적용되는 총 학습 단계
        self.epsilon_cnt = 0 
        self.epsilon_max_cnt = 1

        # self.epsilon_decay = 0.000006 #0.000006 
        self.soft_update_rate = 0.005  # 0.01 타겟 네트워크를 조금씩 업데이트하여 학습안정성
        self.rate_update_frequency = 150000 #몇번의 학습 후 업데이트 할지 결정
        self.max_rate = 0.04 # 최대 업데이트 비율 제한

        self.data_buffer = deque(maxlen=15000) # 최근 15000개의 데이터 저장하고 랜덤 샘플링
        #self.nstep_memory = n_step_buffer(n_step=self.n_step)

        self.action_history = deque([0, 0, 0, 0], maxlen=4) # 최근 4개의 행동을 저장

        self.policy_net = Q_network(self.num_actions).to(self.device) # 현재 상태에서 최적 행동 예측
        self.Q_target_net = Q_network(self.num_actions).to(self.device) # 일정 주기마다 업데이트
        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=self.learning_rate) # adam 옵티마이저
        self.Q_target_net.load_state_dict(self.policy_net.state_dict()) # 초기에는 policy_net과 target_net을 동일하게 설정 
        self.Q_target_net.eval() # target은 학습되지 않음

        self.epi_loss = 0

        if load == True:
            print("Load trained model..")
            self.load_model()

    def update_epsilon(self, current_step):
       
        if self.epsilon_cnt == self.epsilon_max_cnt:
            pass
            
        else:
            if current_step % self.epsilon_decay_period == 0:
                self.epsilon_cnt += 1
                self.epsilon = self.final_epsilon
            else:
                cos_decay = 0.5 * (1 + math.cos(
                    math.pi * (current_step % self.epsilon_decay_period) / self.epsilon_decay_period))
                self.epsilon = self.final_epsilon + (self.initial_epsilon - self.final_epsilon) * cos_decay

    def epsilon_greedy(self, Q_values):
        # 난수 생성
        if np.random.random() < self.epsilon:
            # action을 random하게 선택
            action = random.randrange(self.num_actions)
            return action
        else:
            # 학습된 Q value 값중 가장 큰 action 선택
            return Q_values.argmax().item()


    def update_info_with_prediction(self, raw_info, predicted_coords, dis_action, toggle): # toggle 인자 추가
        raw_info_np = np.array(raw_info)
    
        # raw_info의 마지막 3개 값을 predicted_coords와 toggle로 덮어쓰기
        combined_info = np.concatenate((predicted_coords.cpu().numpy(), [toggle]))
        raw_info_np[..., -3:] = combined_info
    
        # action_history 추가
        result = np.concatenate((raw_info_np, self.update_action_history(dis_action)))
    
        return torch.Tensor(result)

    # model 저장
    def save_model(self):
        torch.save({
            'state': self.policy_net.state_dict(),
            'optim': self.optimizer.state_dict()},
            save_path)
        return None

    # model 불러오기
    def load_model(self):
        checkpoint = torch.load(load_path)
        self.policy_net.load_state_dict(checkpoint['state'])
        self.Q_target_net.load_state_dict(checkpoint['state'])
        self.optimizer.load_state_dict(checkpoint['optim'])
        return None

    def store_trajectory(self, traj):
        self.data_buffer.append(traj)

    # 한장만 256*256
    # def re_scale_frame_detect(self, obs):
    #     obs = cp.array(obs)
    #     obs = cp.asnumpy(obs)
    #     obs = np.transpose(obs, (1, 2, 0))  # (C,H,W) → (H,W,C)

    #     obs = resize(obs, (256, 256), anti_aliasing=True)

    #     return obs  # (256,256,3)
    
    # 한장짜리 256*256 (탐지 모델에 이미지를 입력)
    def init_obs_detect(self, obs):
        # obs: torch.Tensor (C,H,W)
        img = obs.cpu().numpy()  # torch → numpy
        img = np.transpose(img, (1, 2, 0))  # (C,H,W) → (H,W,C)
        img = np.clip(img * 255, 0, 255).astype(np.uint8)  # uint8 변환
        img = cv2.resize(img, (256, 256))
        img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        return img  # numpy (256,256,3)


    ### DQN 학습용 이미지 전처리
    # 1. resizing : 64 * 64, gray scale로
    def re_scale_frame(self, obs):
        obs = cp.array(obs)
        obs = cp.asnumpy(obs)
        obs = np.transpose(obs, (1, 2, 0))
        obs = resize(rgb2gray(obs), (64, 64))
        return obs

    # 2. image 4개씩 쌓기 (e_scale_frame에서 만든 64x64 흑백 이미지를 4장 쌓기)
    def init_image_obs(self, obs):
        obs = self.re_scale_frame(obs)
        frame_obs = [obs for _ in range(4)]
        frame_obs = np.stack(frame_obs, axis=0)
        frame_obs = cp.array(frame_obs)  # cupy 배열로 변환
        return frame_obs

    # 3. 4장 쌓인 Image return (init_image_obs(obs) 호출)
    def init_obs(self, obs):
        return self.init_image_obs(obs)

    def camera_obs(self, obs):
        camera_obs = cp.array(obs)  # cupy 배열로 변환
        # print(obs.shape) # 4 64 64 3
        camera_obs = cp.expand_dims(camera_obs, axis=0)
        camera_obs = torch.from_numpy(cp.asnumpy(camera_obs)).to(self.device)  # GPU로 전송
        return camera_obs

    def ray_obs(self, obs):
        ray_obs = cp.array(obs)  # cupy 배열로 변환
        ray_obs = cp.expand_dims(ray_obs, axis=0)
        ray_obs = torch.from_numpy(cp.asnumpy(ray_obs)).unsqueeze(0).to(self.device)  # GPU로 전송
        return ray_obs

    def ray_obs_cpu(self, obs):
        obs_gpu = cp.asarray(obs)
        obs_gpu = cp.reshape(obs_gpu, (1, -1))
        return cp.asnumpy(obs_gpu)

    # FIFO, 4개씩 쌓기

    def accumulated_image_obs(self, obs, new_frame):
        temp_obs = obs[1:, :, :]  # 4x3x64x64에서 제일 오래된 이미지 제거 => 3x3x64x64
        new_frame = self.re_scale_frame(new_frame)  # 3x64x64
        # plt.imshow(new_frame)
        # plt.show()
        temp_obs = cp.array(temp_obs)  # cupy 배열로 변환
        new_frame = cp.array(new_frame)  # cupy 배열로 변환
        new_frame = cp.expand_dims(new_frame, axis=0)  # 1x3x64x64
        frame_obs = cp.concatenate((temp_obs, new_frame), axis=0)  # 4x3x64x64
        frame_obs = cp.asnumpy(frame_obs)  # 다시 numpy 배열로 변환
        return frame_obs

    def accumlated_all_obs(self, obs, next_obs):
        return self.accumulated_image_obs(obs, next_obs)

    def update_action_history(self, action):
        self.action_history.append(action)
        return list(self.action_history) # 최신 4개의 행동이 담긴 리스트를 반환
        
    def train_policy(self, obs_camera, obs_camera_detect, obs_ray, info_data):
        if isinstance(obs_camera_detect, torch.Tensor):
            img = obs_camera_detect.cpu().numpy()
        else:
            img = obs_camera_detect

        img = np.squeeze(img)

        if img.ndim == 3:
            if img.shape[0] in [1, 3]:
                img = np.transpose(img, (1, 2, 0))
        elif img.ndim == 2:
            img = np.expand_dims(img, axis=-1)

        if img.max() <= 1.0:
            img = np.clip(img * 255, 0, 255).astype(np.uint8)
        else:
            img = img.astype(np.uint8)

        if img.shape[-1] == 1:
            img = np.repeat(img, 3, axis=2)
        elif img.shape[-1] != 3:
            raise ValueError(f" 이미지 채널 이상 {img.shape}")
        
        # 예측 수행 (예측된 좌표)
        coords_array, score = predict_from_frame(img, self.detection_model)
        
        print(f"score : {score}")
        
        if score < 0.004:
            coords_array = (0.0, 0.0)

        #  예측 수행 (예측된 좌표)
        predicted_coords = np.array(coords_array, dtype=np.float32)
        
        # 시각화 ---------
        h, w = img.shape[:2]
        x_px = int(round(predicted_coords[0] * w))  # x 좌표
        y_px = int(round(predicted_coords[1] * h))  # y 좌표

        #  toggle 값 계산
        if x_px == 0.0 and y_px == 0.0:
            toggle = 0.0
        else:
            toggle = 1.0
        
        predicted_x, predicted_y = x_px/w, y_px/h

        print(f" Predicted Coords (norm): {predicted_coords}")
        print(f" Predicted Coords (px): ({x_px}, {y_px})")
        print(f" Predicted Coords (final): ({predicted_x}, {predicted_y})")
        print(f" toggle: ({toggle})")
        
        # 시각화 ----------------------
        # img_vis = img.copy()
        # cv2.circle(img_vis, (x_px, y_px), radius=5, color=(255, 0, 0), thickness=-1)

        # cv2.imshow("Detection Debug", img_vis)
        # # cv2.imwrite("Detection_Image.jpg", img_vis)
        # cv2.waitKey(0)
        # cv2.destroyAllWindows()

        #  info에 예측 좌표 + toggle 저장
        pixel_coords_array = np.array([predicted_x, predicted_y]) # 시각화 좌표
        predicted_coords_tensor = torch.from_numpy(pixel_coords_array).float().to(obs_camera.device)
        predicted_coords_tensor = predicted_coords_tensor.unsqueeze(0).unsqueeze(0).squeeze()

        info_data[..., -3:-1] = predicted_coords_tensor  # (x, z)
        info_data[..., -1] = toggle                      # 탐지 여부

        Q_values = self.policy_net(obs_camera, obs_ray, info_data)
        max_q = Q_values.max()
        action = self.epsilon_greedy(Q_values)

        return action, Q_values[0][action], max_q, predicted_coords_tensor, toggle


    def batch_torch_obs(self, obs):
        obs = [cp.asarray(ob) for ob in obs]  # obs의 모든 요소를 cupy 배열로 변환
        obs = cp.stack(obs, axis=0)  # obs를 축 0을 기준으로 스택
        obs = cp.squeeze(obs, axis=0) if obs.shape[0] == 1 else obs  # 첫 번째 축 제거
        obs = cp.asnumpy(obs)  # 다시 numpy 배열로 변환
        obs = torch.from_numpy(obs).to(self.device)  # torch tensor로 변환
        return obs

    def batch_ray_obs(self, obs):
        obs = cp.asarray(obs)  # cupy 배열로 변환
        # obs = cp.expand_dims(obs, axis=0)  # 새로운 축 추가
        obs = torch.from_numpy(cp.asnumpy(obs)).to(self.device)  # torch tensor로 변환
        return obs

    def batch_info_obs(self, obs):
        obs = cp.asarray(obs)  # cupy 배열로 변환
        # obs = cp.expand_dims(obs, axis=0)  # 새로운 축 추가
        obs = torch.from_numpy(cp.asnumpy(obs)).to(self.device)  # torch tensor로 변환
        return obs

    # update target network
    # Q-Network의 파라미터를 target network 복사
    def update_target(self, step):
        if step % self.rate_update_frequency == 0:
            self.soft_update_rate += 0.001

        self.soft_update_rate = min(self.soft_update_rate, self.max_rate)
        # print("soft_rate: ", self.soft_update_rate)

        policy_dict = self.policy_net.state_dict()
        target_dict = self.Q_target_net.state_dict()

        # 소프트 업데이트 수행
        for name in target_dict:
            target_dict[name] = (1.0 - self.soft_update_rate) * target_dict[name] + self.soft_update_rate * policy_dict[
                name]

        # 업데이트된 가중치를 타겟 네트워크에 설정
        self.Q_target_net.load_state_dict(target_dict)
        # self.Q_target_net.load_state_dict(self.policy_net.state_dict())

    def train(self, step, update_target):

        # mini_batch, idxs, IS_weights = self.memory.sample(self.batch_size)
        random_mini_batch = random.sample(self.data_buffer, self.batch_size)
      

        self.obs_camera_list, self.obs_ray_list, self.info_list, self.action_list, self.reward_list, self.next_obs_camera_list, self.next_obs_ray_list, self.next_info_list, self.mask_list = zip(
            *random_mini_batch)

        # tensor
        obses_camera = self.batch_torch_obs(self.obs_camera_list)
        obses_ray = self.batch_ray_obs(self.obs_ray_list)
        # print("camera:",obses_camera.shape)
        # print("ray:",obses_ray.shape)

        actions = torch.LongTensor(self.action_list).unsqueeze(1).to(self.device)

        rewards = torch.Tensor(self.reward_list).to(self.device)
        next_obses_camera = self.batch_torch_obs(self.next_obs_camera_list)
        next_obses_ray = self.batch_ray_obs(self.next_obs_ray_list)

        masks = torch.Tensor(self.mask_list).to(self.device)

        obs_info = self.batch_info_obs(self.info_list)
        next_obs_info = self.batch_info_obs(self.next_info_list)
        # print("info:",obs_info.shape)

        Q_values = self.policy_net(obses_camera, obses_ray, obs_info)
        q_value = Q_values.gather(1, actions).view(-1)
        # print(q_value)

        # get target, y(타겟값) 구하기 위한 다음 state에서의 max Q value
        # target network에서 next state에서의 max Q value -> 상수값
        with torch.no_grad():
            target_q_value = self.Q_target_net(next_obses_camera, next_obses_ray, next_obs_info).max(1)[0]

        Y = (rewards + masks * (self.gamma ** self.n_step) * target_q_value).clone().detach()

        MSE = nn.MSELoss()
        #           input,  target
        loss = MSE(q_value, Y.detach())
        # errors = F.mse_loss(q_value, Y, reduction='none')

        # 우선순위 업데이트
        # for i in range(self.batch_size):
        #     tree_idx = idxs[i]
        #     self.memory.batch_update(tree_idx, errors[i])

        self.optimizer.zero_grad()

        # loss 정의 (importance sampling)
        # loss =  (torch.FloatTensor(IS_weights).to(self.device) * errors).mean()
        # 10,000번의 episode동안 몇 번의 target network update가 있는지
        # target network update 마다 max Q-value / loss function 분포

        # # tensor -> list
        # # max Q-value 분포
        # tensor_to_list_q_value = target_q_value.tolist()
        # # max_Q 값들(batch size : 32개)의 평균 값
        # list_q_value_avg = sum(tensor_to_list_q_value)/len(tensor_to_list_q_value)
        # self.y_max_Q_avg.append(list_q_value_avg)

        # # loss 평균 분포(reduction = mean)
        # loss_in_list = loss.tolist()
        # self.y_loss.append(loss_in_list)

        # backward 시작

        loss.backward()
        self.optimizer.step()

        self.epi_loss += loss.item()
        # --------------------------------------------------------------------

##########################################################################################################################    

def main():
    env = UAV_env(time_scale=2.0, filename=project_exe, port=11300)

    cudnn.enabled = True
    cudnn.benchmark = True

    score = 0
    # episode당 step
    episode_step = 0
    # 전체 누적 step
    step = 0
    update_target = 1000  # 2000
    initial_exploration = 10000  # 10000

    agent = Agent()  # 에이전트 인스턴스

    if load:
        agent.load_model()

    for epi in range(5001):
        obs = env.reset()

        # --- 초기 관측값 설정 ---
        obs_camera = torch.Tensor(obs[0]).squeeze(dim=0) # 원본 이미지
        # (84, 84, 3) -> (64, 64, 1) -> 4장씩 쌓아 (64, 64, 4)
        # 같은 Image 4장 쌓기 -> 이후 action에 따라 환경이 바뀌고, 다른 Image data 쌓임
        obs_camera_detect = agent.init_obs_detect(obs_camera) # 탐지용 이미지
        obs_camera = agent.init_obs(obs_camera) # DQN용 4-frame 이미지 (state 1. uav의 카메라 이미지)

        obs_height = obs[1]
        obs_ray = obs[2]
        # # c#에서 받아온 obs
        # #[0.         1.         1.         0.         1.         1.
        # # 0.         1.         1.         0.         1.         1.
        # # 1.         0.         0.17635795 0.         1.         1.
        # # 0.         1.         1.         0.         1.         1.
        # # 0.         1.         1.]
        #
        
        idx_list = [2, 5, 8, 11, 14, 17, 20, 23]
        obs_ray_tensor = [obs_ray[i] for i in range(27) if i in idx_list]
        obs_ray_tensor = np.append(obs_ray_tensor, obs_height[2]) # ray 센서 (9개_장애물)
        obs_ray_tensor = torch.Tensor(obs_ray_tensor)
        obs[3] = np.concatenate((obs[3], [0, 0, 0, 0]))  # action 4step 추가
        obs_info = torch.Tensor(obs[3])

        print(" [Episode Start] 초기 obs_info:", obs_info)

        while True:

            print(" [Before Policy Net] obs_info:", obs_info)

            # action 선택 (현재 상태 기반)
            dis_action, estimate_Q, max_est_Q, predicted_coords, toggle = agent.train_policy(
                                                                    agent.camera_obs(obs_camera),
                                                                   agent.camera_obs(obs_camera_detect), # 현재 스텝의 탐지용 이미지 사용
                                                                   agent.ray_obs(obs_ray_tensor),
                                                                   agent.ray_obs(obs_info))

            print(" [After Policy Net] 선택 action:", dis_action)

            if episode_step == 0:
                print("Max Q-value: ", max_est_Q.cpu().item())
                print("Epsilon:", agent.epsilon)

            # action에 따른 step()
            # next step, reward, done 여부
            next_obs, reward, done = env.step(dis_action)
            
            # --- 다음 상태(next_obs) 처리 ---
            print("  [Next] next_obs_info (raw):", next_obs[3])
            
            # 1. 다음 스텝의 '원본' 카메라 프레임을 먼저 준비합니다.
            next_obs_camera_raw = torch.Tensor(next_obs[0]).squeeze(dim=0)

            # 2. 원본 프레임으로 다음 스텝의 '탐지용' 이미지를 생성합니다.
            next_obs_camera_detect = agent.init_obs_detect(next_obs_camera_raw)
            
            # 3. 원본 프레임으로 다음 스텝의 'DQN용' 4-frame 스택 이미지를 생성합니다. (가장 오래된 프레임은 버리고, 새로 들어온 프레임을 추가하여 4장 스택을 유지)
            next_obs_camera_stack = agent.accumlated_all_obs(obs_camera, next_obs_camera_raw)

            # state는 camera sensor로 얻은 Image만
            next_obs_height = next_obs[1]
            next_obs_ray = next_obs[2]
            next_obs_info = next_obs[3]

            # todo next obs ray
            next_obs_ray_tensor = [next_obs_ray[i] for i in range(27) if i in idx_list]
            next_obs_ray_tensor = np.append(next_obs_ray_tensor, next_obs_height[2])
            next_obs_ray_tensor = torch.Tensor(next_obs_ray_tensor)

            # === [4]  정석 좌표 + 히스토리 업데이트 ===
            next_obs_info = agent.update_info_with_prediction(
                raw_info=next_obs_info,
                predicted_coords=predicted_coords,   # 이건 train_policy 쪽에서 넘겨와야 함
                dis_action=dis_action,
                toggle=toggle
            )
            next_obs_info = torch.Tensor(next_obs_info)
            print(" [Next] next_obs_info (+history):", next_obs_info)

            #next_obs_info = np.concatenate((next_obs_info, agent.update_action_history(dis_action)))


            mask = 0 if done else 1
            # print("%d번째 step에서의 reward : %f, action speed : %f"%(step, reward, action_speed))
            score += reward

            
            agent.store_trajectory(
                [obs_camera, agent.ray_obs_cpu(obs_ray_tensor), agent.ray_obs_cpu(obs_info), dis_action, reward,
                 next_obs_camera_stack, agent.ray_obs_cpu(next_obs_ray_tensor), agent.ray_obs_cpu(next_obs_info), mask])

             # --- 다음 스텝을 위해 모든 관측값을 업데이트 ---
            obs_camera = next_obs_camera_stack
            obs_camera_detect = next_obs_camera_detect #  탐지용 이미지 업데이트
            obs_ray_tensor = next_obs_ray_tensor
            obs_info = next_obs_info
            
            # SumTree 노드 수가 배치 사이즈 이상 되면 학습
            if step > agent.batch_size:
                # if agent.memory.tree.n_entries > agent.n_step:
                agent.train(step, update_target)

                # 모델 저장
                if step % 2000 == 0:
                    agent.save_model()

                # 타겟 네트워크 업데이트
                if step % update_target == 0:
                    agent.update_target(step)

            episode_step += 1
            step += 1
            agent.update_epsilon(step)

            if done:
                cuda.empty_cache()
                gc.collect()
                break

        print('%d 번째 episode의 총 step: %d' % (epi + 1, episode_step))
        print('True_score: %f' % score)
        print('Total step: %d\n' % step)


        # todo wandb
        wandb.log({
            "episode_step": episode_step,
            "score": score,
            "Average score": score / episode_step,
            "init Max Q": max_est_Q.cpu().item(),
            "Average Loss": agent.epi_loss / episode_step if episode_step != 0 else 0,
            "Epsilon": agent.epsilon,
            "Episode": epi},
            step=epi)

        agent.epi_loss = 0
        score = 0
        episode_step = 0


if __name__ == '__main__':
    main()
