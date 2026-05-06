import os
import cv2
import json
import torch
import numpy as np
import re
import time
from PIL import Image
from collections import deque, Counter
from datetime import datetime
from typing import Optional

# Third-party & Model Imports
from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor
from qwen_vl_utils import process_vision_info

# Custom Internal Modules (Implementations hidden)
from method_lib.object_detect.google_detect import ObjectDetector
from method_lib.object_detect.hand_detect import detect_hands_cv 
from method_lib.memory.understand2 import understand_actions
from method_lib.memory.modify2 import modify_structure
from ollama_test.base_chat import base_chat
from ollama_test.vision_langchain import analyze_image
from method_lib.vedio.qwenapi import infer_image
from method_lib.vedio.detect_hands_from_frame import detect_hands_from_frame

# ============================ Configuration & Templates ============================

# Abstracted Prompt Templates (Hiding specific prompt engineering details)
PROMPT_TEMPLATES = {
    "hand_state": "Analyze the image and determine if the person is holding any object. Reply only with: 'Holding', 'Empty', or 'Occluded'.",
    "action_place": "Briefly describe where the user placed the {target_object}.",
    "action_pick": "Briefly describe from where the user picked up the {target_object}.",
    "object_attr": "Describe the visual appearance of the {target_object} concisely.",
    "fine_grained_loc": "Detail the precise location of the {target_object} in the current scene.",
    "qa_location": "Based on the scene data {scene_data}, where is the {object} located? Reply 'Not Found' if it is not present.",
    "qa_eval": "Evaluate if the answer '{answer}' is correct for the question '{question}' based on the scene data {scene_data}. Reply 'yes' or 'no'."
}

API_KEY = "YOUR_API_KEY_HERE"
MODEL_VL_PATH = "YOUR_MODEL_PATH_HERE" # e.g., "Qwen/Qwen2.5-VL-7B-Instruct"
DETECTOR_MODEL_PATH = "YOUR_DETECTOR_MODEL_PATH_HERE"

# ============================ Utility Functions ============================

def copy_json_file(src_path: str, dst_path: str) -> None:
    with open(src_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    with open(dst_path, 'w', encoding='utf-8') as f:
        json.dump(data, f, indent=4, ensure_ascii=False)

def is_point_in_box(x: float, y: float, box: list) -> bool:
    x1, y1, x2, y2 = box
    return x1 <= x <= x2 and y1 <= y <= y2

def save_segment_from_frames(frames: list, output_path: str, fps: int = 30) -> bool:
    if not frames:
        print(f"[Warning] No frames to save, skipping: {output_path}")
        return False

    height, width = frames[0].shape[:2]
    output_path = os.path.splitext(output_path)[0] + ".avi"
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    writer = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
    
    if not writer.isOpened():
        print(f"[Error] Failed to save or corrupted file: {output_path}")
        return False

    for frame in frames:
        writer.write(frame)
    writer.release()
    print(f"[Info] Segment saved successfully: {output_path} ({len(frames)} frames)")
    return True

def draw_and_save_visualization(frame, hand_coords, object_detections, interacted_objects, frame_idx, save_dir):
    for hand in hand_coords:
        for x, y in hand:
            cv2.circle(frame, (int(x), int(y)), 3, (0, 255, 255), -1)
            
    for det in object_detections:
        label, score, box = det["label"], det["score"], det["box"]
        x1, y1, x2, y2 = map(int, box)
        color = (0, 255, 0) if label in interacted_objects else (0, 0, 255)
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
        cv2.putText(frame, f"{label} {score:.2f}", (x1, max(0, y1 - 10)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
        
    os.makedirs(save_dir, exist_ok=True)
    cv2.imwrite(os.path.join(save_dir, f"frame_{frame_idx:05d}.jpg"), frame)

def get_room_by_frame(json_path: str, frame_number: int) -> str:
    with open(json_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    for block in data:
        if block['start'] <= frame_number <= block['end']:
            return block.get('room', 'Unknown Region')
    raise ValueError(f"Frame {frame_number} is out of structural bounds.")

# ======================== Core Processing Logic ===========================

def segment_on_state_change(
    cap, processor_vl, model_vl, model_name,
    segment_dir="segments", fps_check=30,
    device="cuda", min_interval_seconds=2,
    delay_time_has_to_no=0.5, delay_time_no_to_has=1.5,
    back_time_has_to_no=4.5, back_time_no_to_has=3.5,
    room_info_json: Optional[str] = None
):
    """
    Segments a continuous video stream based on the state transitions of hand-object interaction.
    Analyzes temporal consistency to filter out noise and extract key action segments.
    """
    os.makedirs(segment_dir, exist_ok=True)

    frame_id = 0
    segment_idx = 0
    buffer_frames = []
    
    current_state = "no"
    last_stable_state = "no"
    last_segment_time = -min_interval_seconds
    last_segment_end_time = 0
    
    state_window = deque(maxlen=6)
    stable_num = 4
    
    current_back_time = None
    current_delay_time = None
    pending_segment_time = None
    pending_segment_state = None
    transition_frame_id = None

    print("[Process] Video temporal segmentation initialized...")

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        buffer_frames.append(frame)

        # 1. Process pending segment extraction
        if pending_segment_time is not None:
            current_time_sec = frame_id / fps_check
            if current_time_sec - pending_segment_time >= current_delay_time:
                start_frame_id = max(0, transition_frame_id - int(fps_check * current_back_time))
                end_frame_id = frame_id
                segment_frames = buffer_frames[start_frame_id:end_frame_id]
                state_trans = f"{last_stable_state}2{pending_segment_state}"
                
                duration = end_frame_id - start_frame_id
                keyframe_offset = int(duration - 3 * fps_check + fps_check // 2)
                keyframe_offset = max(0, min(duration - 1, keyframe_offset))
                
                print(f"[Segment Triggered] Boundaries: {start_frame_id} -> {end_frame_id} | Keyframe Offset: {keyframe_offset}")
                
                room = get_room_by_frame(room_info_json, start_frame_id + keyframe_offset)
                segment_path = os.path.join(segment_dir, f"segment_{segment_idx:03d}_{room}_{state_trans}.avi")
                
                saved = save_segment_from_frames(segment_frames, segment_path, fps=int(fps_check))
                last_segment_end_time = frame_id / fps_check
                
                if saved:
                    keyframe = segment_frames[keyframe_offset]
                    keyframe_path = segment_path.replace(".avi", f"_{start_frame_id+keyframe_offset}.jpg")
                    os.rename(segment_path, segment_path.replace(".avi", f"_{start_frame_id+keyframe_offset}.avi"))
                    cv2.imwrite(keyframe_path, keyframe)
                    
                    print(f"[State Shift] {last_stable_state} -> {pending_segment_state}. Saved Segment ID: {segment_idx}")
                    segment_idx += 1
                    last_segment_time = current_time_sec
                    last_stable_state = pending_segment_state

                pending_segment_time = None
                pending_segment_state = None
                transition_frame_id = None

        # 2. Vision Language Model Inference for State Tracking
        if frame_id % 15 == 0:
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            coords = detect_hands_from_frame(frame_rgb, visualize=False)

            if coords:
                frame_pil = Image.fromarray(frame_rgb)
                question = PROMPT_TEMPLATES["hand_state"]

                messages = [{"role": "user", "content": [{"type": "image", "image": frame_pil}, {"type": "text", "text": question}]}]
                text_input = processor_vl.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
                image_inputs, video_inputs = process_vision_info(messages)
                inputs = processor_vl(text=[text_input], images=image_inputs, videos=video_inputs, padding=True, return_tensors="pt").to(device)

                with torch.no_grad():
                    generated_ids = model_vl.generate(**inputs, max_new_tokens=64)
                    output_text = processor_vl.batch_decode(
                        generated_ids[:, inputs.input_ids.shape[1]:], skip_special_tokens=True
                    )[0].strip()
            else:
                output_text = 'Occluded'
            
            # State evaluation logic
            if "occluded" in output_text.lower():
                pass # Skip updating the window for occlusions
            else:
                is_empty = any(neg in output_text.lower() for neg in ["empty", "no", "nothing"])
                inferred_state = "no" if is_empty else "has"
                state_window.append(inferred_state)

                if len(state_window) == state_window.maxlen:
                    count_has = sum(1 for s in state_window if s == "has")
                    count_no = sum(1 for s in state_window if s == "no")
                    new_state = "has" if count_has >= stable_num else "no" if count_no >= stable_num else current_state
                else:
                    new_state = current_state

                if new_state != current_state:
                    print(f"[State Trend Change] {current_state} -> {new_state}")
                    current_state = new_state

                if new_state != last_stable_state and pending_segment_time is None:
                    current_time_sec = frame_id / fps_check
                    if current_time_sec - last_segment_time >= min_interval_seconds:
                        pending_segment_time = current_time_sec
                        pending_segment_state = current_state
                        transition_frame_id = frame_id

                        if last_stable_state == "has" and current_state == "no":
                            current_delay_time, current_back_time = delay_time_has_to_no, back_time_has_to_no
                        elif last_stable_state == "no" and current_state == "has":
                            current_delay_time, current_back_time = delay_time_no_to_has, back_time_no_to_has
                        else:
                            current_delay_time, current_back_time = 1.0, 2.5

        frame_id += 1

    cap.release()
    print("[Process] Video segmentation completed.")

def get_main_object(video_path, model_path, labels_text, frame_interval=8, visualize=True, vis_dir=None):
    """
    Identifies the primary object of interaction within a specific video segment.
    """
    cap = cv2.VideoCapture(video_path)
    detector = ObjectDetector(model_path=model_path)
    interaction_counter = Counter()
    frame_idx = 0

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret: break
        
        if frame_idx % frame_interval == 0:
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            coords = detect_hands_from_frame(frame_rgb, visualize=False)
            
            if coords:
                image_pil = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
                hand_coords = detect_hands_cv(image_pil)
                object_detections = detector.detect(image_pil, labels_text=labels_text)
                
                interacted_objects = set()
                for hand in hand_coords:
                    for x, y in hand:
                        for det in object_detections:
                            if is_point_in_box(x, y, det["box"]):
                                interacted_objects.add(det["label"])
                                
                for obj in interacted_objects:
                    interaction_counter[obj] += 1
                    
                if visualize:
                    draw_and_save_visualization(frame.copy(), hand_coords, object_detections, interacted_objects, frame_idx, vis_dir or "vis_temp")
        frame_idx += 1
        
    cap.release()
    return interaction_counter.most_common(1)[0][0] if interaction_counter else "Unknown Object"

def append_to_detect(json_path: str, frame_number: int, action_type: str):
    with open(json_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    found = False
    for block in data:
        if block['start'] <= frame_number <= block['end']:
            block.setdefault('detect', []).append(action_type)
            found = True
            break
    if not found:
        raise ValueError(f"Frame {frame_number} out of bounds.")
    with open(json_path, 'w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, indent=2)

def get_object_by_frame(json_path: str, frame_number: int) -> str:
    with open(json_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    for block in data:
        if block['start'] <= frame_number <= block['end']:
            return block.get('object', 'Unknown')
    raise ValueError(f"Frame {frame_number} out of bounds.")

def get_unique_objects(json_path: str) -> list:
    with open(json_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    unique_objects = {block['object'] for block in data if 'object' in block}
    return list(unique_objects)

def process_video_folder(dataset_dir, output_dir, model, processor, label_lang_map):
    """
    Main pipeline handler for processing a directory of semantic video datasets.
    Handles semantic segmentation, interaction analysis, and graph/memory updates.
    """
    for fname in os.listdir(dataset_dir):
        if not fname.endswith(".mp4"):
            continue
            
        name = os.path.splitext(fname)[0]
        finished_flag = os.path.join(dataset_dir, f"{name}_finished_scene.json")
        if os.path.exists(finished_flag):
            print(f"[Skip] Video already processed: {name}")
            continue

        video_path = os.path.join(dataset_dir, fname)
        json_path = os.path.join(dataset_dir, name + ".json")
        current_segment_out_dir = os.path.join(output_dir, name)
        os.makedirs(current_segment_out_dir, exist_ok=True)

        print(f"\n[Pipeline] Starting analysis for video: {name}")

        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            print(f"[Error] Failed to open video file: {video_path}")
            continue

        # Stage 1: State-based segmentation
        segment_on_state_change(
            cap, processor, model, model_name="qwenvl",
            segment_dir=current_segment_out_dir, fps_check=30,
            device="cuda", min_interval_seconds=0.5,
            delay_time_has_to_no=0, delay_time_no_to_has=1.5,
            back_time_has_to_no=6, back_time_no_to_has=3.5,
            room_info_json=json_path
        )
        cap.release()

        # Stage 2: Segment analysis and knowledge graph updates
        initial_json_path = video_path.replace(".mp4", "_initial_scene.json")
        building_json_path = initial_json_path.replace("_initial_scene.json","_building_scene.json")
        finished_json_path = initial_json_path.replace("_initial_scene.json","_finished_scene.json")
        final_json_path = initial_json_path.replace("_initial_scene.json","_final_scene.json")
        action_json_path = initial_json_path.replace("_initial_scene.json","_action.json")
        base_json_path = initial_json_path.replace("_initial_scene.json",".json")
        
        copy_json_file(initial_json_path, building_json_path)
        log_list = []
        
        for seg_file in sorted(os.listdir(current_segment_out_dir)):
            if not seg_file.endswith(".avi"): continue
            seg_path = os.path.join(current_segment_out_dir, seg_file)
            
            if "has2no" in seg_file or "no2has" in seg_file:
                print(f"\n[Analysis] Analyzing semantic segment: {seg_path}")
                most_interacted_object = get_main_object(
                    seg_path,
                    model_path=DETECTOR_MODEL_PATH,
                    labels_text=". ".join(dict(list(label_lang_map.items())[:-1]).keys()),
                    visualize=False
                )
                
                target_object = label_lang_map.get(most_interacted_object, most_interacted_object)
                image_path = seg_path.replace(".avi", ".jpg")
                
                if os.path.exists(image_path):
                    action_data = {}
                    room_match = re.search(r'segment_\d{3}_(.*?)_(has2no|no2has)_(\d+)\.avi', seg_file)
                    number = int(room_match.group(3)) if room_match else 0
                    
                    action_data['object'] = target_object
                    action_data['object_groundtruth'] = get_object_by_frame(base_json_path, number)
                    
                    start_time = time.time()

                    # Semantic LLM API calls via abstract prompts
                    if "has2no" in seg_file:
                        prompt = PROMPT_TEMPLATES["action_place"].format(target_object=target_object)
                        action_data['action'] = analyze_image(image_path, prompt, api_key=API_KEY)
                        append_to_detect(action_json_path, number, 'Placed')
                    else:
                        prompt = PROMPT_TEMPLATES["action_pick"].format(target_object=target_object)
                        action_data['action'] = analyze_image(image_path, prompt, api_key=API_KEY)
                        append_to_detect(action_json_path, number, 'Picked')

                    attr_prompt = PROMPT_TEMPLATES["object_attr"].format(target_object=target_object)
                    action_data['object_attribute'] = analyze_image(image_path, attr_prompt, api_key=API_KEY)
                    
                    if room_match:
                        action_data['background'] = room_match.group(1) or room_match.group(2)
                    else:
                        action_data['background'] = "Unknown Region"

                    # Core memory operations
                    understand = understand_actions(action_data, json_path=building_json_path)
                    action_data['caption_time'] = time.time() - start_time
                    
                    start_time = time.time()
                    modify_log = modify_structure(understand, json_path=building_json_path)
                    action_data['graph_update_time'] = time.time() - start_time

                    if 'fine_grained_flag' in video_path: # Extensible flag check
                        fg_prompt = PROMPT_TEMPLATES["fine_grained_loc"].format(target_object=target_object)
                        action_data['fine_grained_location'] = analyze_image(image_path, fg_prompt, api_key=API_KEY)
                        
                    log_list.append({
                        "action_data": action_data,
                        "understand": understand,
                        "modify_log": modify_log,
                        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                    })
                    
        output_log_path = video_path.replace(".mp4", "_building_log.json")
        with open(output_log_path, "w", encoding="utf-8") as f:
            json.dump(log_list, f, ensure_ascii=False, indent=4)
            
        os.rename(building_json_path, finished_json_path)

        # Stage 3: LLM Evaluation Phase
        objects = get_unique_objects(base_json_path)
        with open(finished_json_path, 'r', encoding='utf-8') as f:
            finished_data = json.load(f)
        with open(final_json_path, 'r', encoding='utf-8') as f:
            final_data = json.load(f)
            
        llm_answer_data = []
        for obj in objects:
            start_time = time.time()
            qa_prompt = PROMPT_TEMPLATES["qa_location"].format(scene_data=finished_data, object=obj)
            llm_answer = base_chat(qa_prompt)
            runtime = time.time() - start_time
            
            if 'Not Found' in llm_answer:
                llm_answer_score = -1
            else:
                eval_prompt = PROMPT_TEMPLATES["qa_eval"].format(scene_data=final_data, question=f"Where is {obj}?", answer=llm_answer)
                eval_response = base_chat(eval_prompt)
                llm_answer_score = 1 if 'yes' in eval_response.lower() else 0
                
            llm_answer_data.append({"Question": f"Where is {obj}?", "Answer": llm_answer, "Score": llm_answer_score, "Time": runtime})
            
        llm_answer_data_path = video_path.replace(".mp4", "_llm_answer_data.json")
        with open(llm_answer_data_path, "w", encoding="utf-8") as f:
            json.dump(llm_answer_data, f, ensure_ascii=False, indent=4)

# ======================== Execution Entry ========================

if __name__ == "__main__":
    t_start = time.time()
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Common object mappings 
    lang_dict = {
        "Knife": "Knife", "Scissors": "Scissors", "Container Bowl": "Container Bowl", 
        "Remote control with buttons": "Remote Control", "Phone": "Phone", 
        "Spoon": "Spoon", "Pen": "Pen", "Kettle": "Kettle", "Cup": "Cup", 
        "Book": "Book", "Mouse": "Mouse", "Watch": "Watch", "Earphone": "Earphone", 
        "Sunglasses": "Sunglasses", "Unknown": "Unknown Object"
    }

    try:
        model_vl = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            MODEL_VL_PATH,
            torch_dtype=torch.bfloat16,
            attn_implementation="flash_attention_2",
            device_map="auto"
        )
        processor_vl = AutoProcessor.from_pretrained(MODEL_VL_PATH)
        print(f"[System] Vision Language Model loaded successfully.")
    except Exception as e:
        print(f"[System Error] Model load failure: {e}")
        exit()

    # Placeholder paths for dataset directories
    dataset_base_dirs = [
        "path/to/your/dataset_folder_1",
        # "path/to/your/dataset_folder_2"
    ]
    output_base_dir = "path/to/your/output_segments"

    for dataset_dir in dataset_base_dirs:
        if not os.path.exists(dataset_dir):
            print(f"[Warning] Path does not exist, skipping: {dataset_dir}")
            continue
            
        print(f"\n[System] Initiating processing for directory: {dataset_dir}")
        process_video_folder(
            dataset_dir=dataset_dir,
            output_dir=output_base_dir,
            model=model_vl,
            processor=processor_vl,
            label_lang_map=lang_dict
        )

    print(f"\n[System] Pipeline completed. Total execution time: {time.time() - t_start:.2f} seconds")