import av
import numpy as np
import matplotlib.pyplot as plt
from decord import VideoReader, cpu


def load_video1(video_path, max_frames_num,fps=1,force_sample=False):
    if max_frames_num == 0:
        return np.zeros((1, 336, 336, 3))
    vr = VideoReader(video_path, ctx=cpu(0),num_threads=1)
    total_frame_num = len(vr)
    print("Total frame number:", total_frame_num)
    video_time = total_frame_num / vr.get_avg_fps()
    
    fps = round(vr.get_avg_fps()/fps)
    frame_idx = [i for i in range(0, len(vr), fps)]
    frame_time = [i/fps for i in frame_idx]
    if len(frame_idx) > max_frames_num or force_sample:
        sample_fps = max_frames_num
        uniform_sampled_frames = np.linspace(0, total_frame_num - 1, sample_fps, dtype=int)
        frame_idx = uniform_sampled_frames.tolist()
        frame_time = [i/vr.get_avg_fps() for i in frame_idx]
    frame_time = ",".join([f"{i:.2f}s" for i in frame_time])
    spare_frames = vr.get_batch(frame_idx).asnumpy()
    return spare_frames,frame_time,video_time

def load_video(video_path, max_frames_num, force_sample=False):
    if max_frames_num == 0:
        return np.zeros((1, 336, 336, 3)), "", 0.0

    container = av.open(video_path)
    stream = container.streams.video[0]
    stream.thread_type = "AUTO"

    # 전체 재생시간 (초)
    video_time = float(stream.duration * stream.time_base)

    keyframes = []
    timestamps = []
    for packet in container.demux(stream):
        if not packet.is_keyframe:
            continue
        for frame in packet.decode():
            img = frame.to_ndarray(format="rgb24")
            keyframes.append(img)
            # Fraction → float으로 변환
            time_sec = float(frame.pts * stream.time_base)
            timestamps.append(time_sec)
        if len(keyframes) >= max_frames_num:
            break

    if force_sample and len(keyframes) > max_frames_num:
        idxs = np.linspace(0, len(keyframes)-1, max_frames_num, dtype=int)
        keyframes = [keyframes[i] for i in idxs]
        timestamps = [timestamps[i] for i in idxs]
    print(len(keyframes),keyframes[0].shape)  # 이제 float이므로 포맷 가능
    print(timestamps)
    # 이제 float이므로 포맷 가능
    frame_time = ",".join([f"{t:.2f}s" for t in timestamps])
    frames = np.stack(keyframes, axis=0)

    return frames, frame_time, video_time

import av, numpy as np, cv2
from typing import Tuple

def select_high_quality_uniform_one_pass(
    video_path: str,
    max_frames_num: int,
    lambda_coef: float = 0.5
) -> Tuple[np.ndarray, str, float]:
    """
    I/P 프레임 메타 → 균등 샘플링 → 단일 demux/decode 로 선택된 pts만 한 번에 디코딩
    """
    container = av.open(video_path)
    stream = container.streams.video[0]
    tb = stream.time_base
    video_time = float(stream.duration * tb)

    # 1) 메타 수집
    frame_meta = [(fr.pts, float(fr.pts * tb))
                  for pkt in container.demux(stream)
                  for fr in pkt.decode()
                  if fr.pict_type in (1, 2)]
    if not frame_meta:
        return np.zeros((1,336,336,3)), "", video_time

    # 2) 균등 샘플링
    n_meta = len(frame_meta)
    if n_meta <= max_frames_num:
        idxs = np.arange(n_meta)
    else:
        idxs = np.linspace(0, n_meta-1, max_frames_num, dtype=int)

    sampled = [frame_meta[i] for i in idxs]
    selected_pts  = [pts for pts,_ in sampled]
    selected_times = [ts for _,ts in sampled]

    # 3) 화질 점수 계산을 위해 일단 디코딩 안 한 상태로 pts만 보관
    #    (샤프니스/블록니스는 디코딩 된 이미지에서만 계산)
    #    #—> 여기선 퀄리티 스코어 없이 뽑으려면 4단계를 건너뜀

    # 4) 한 번만 seek & decode
    pts_set = set(selected_pts)
    decoded = {}
    container.seek(min(selected_pts), any_frame=False, backward=True, stream=stream)
    for packet in container.demux(stream):
        for fr in packet.decode():
            if fr.pts in pts_set:
                decoded[fr.pts] = fr.to_ndarray(format="rgb24")
                if len(decoded) == len(pts_set):
                    break
        if len(decoded) == len(pts_set):
            break

    # 5) 선택된 순서대로 배열 생성
    selected_imgs = [decoded[pt] for pt in selected_pts]
    frames = np.stack(selected_imgs, axis=0)
    frame_time_str = ",".join(f"{t:.2f}s" for t in selected_times)

    return frames, frame_time_str, video_time
import time
def visualize_keyframes(video_path, max_frames_num, force_sample=False):
    # 1) 시간 측정 시작
    start_time = time.time()
    
    
    # 1) I-Frame만 추출하는 load_video 호출
    frames, timestamps, video_time = select_high_quality_uniform_one_pass(
        video_path,
        max_frames_num,
        
    )
    # 3) 시간 측정 종료 및 출력
    elapsed = time.time() - start_time
    print(f"▶ 프레임 추출 소요 시간: {elapsed:.2f}초")
    # timestamps: ["0.00s", "1.23s", ...] 리스트라고 가정

    n = frames.shape[0]
    cols = min(n, 4)
    rows = int(np.ceil(n / cols))

    plt.figure(figsize=(cols * 3, rows * 3))
    for i in range(n):
        ax = plt.subplot(rows, cols, i + 1)
        ax.imshow(frames[i])
        ax.set_title(timestamps[i], fontsize=10)
        ax.axis('off')
    plt.suptitle(f"Total Video Duration: {video_time:.2f}s, Key-frames: {n}", fontsize=12)
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.show()

# 사용 예시
video_path = "assets/example.mp4"
visualize_keyframes(video_path, max_frames_num=12, force_sample=True)
