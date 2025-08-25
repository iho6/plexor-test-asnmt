import os
import cv2
import argparse
import numpy as np
from ultralytics import YOLO

class EnhancedPhoneDetection:
    def __init__(self, model_name="yolov8n.pt", crop_expand_ratio=0.3, upscale_factor=2.0):
        self.model = YOLO(model_name)
        self.crop_expand_ratio = crop_expand_ratio
        self.upscale_factor = upscale_factor

    def video_to_frames(self, video_path):
        frames = []
        cap = cv2.VideoCapture(video_path)
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            frames.append(frame)
        cap.release()
        return frames

    def expand_bbox(self, bbox, frame_shape, expand_ratio=0.3):
        h, w = frame_shape[:2]
        x1, y1, x2, y2 = bbox
        width = x2 - x1
        height = y2 - y1
        expand_w = int(width * expand_ratio)
        expand_h = int(height * expand_ratio)
        new_x1 = max(0, x1 - expand_w)
        new_y1 = max(0, y1 - expand_h)
        new_x2 = min(w, x2 + expand_w)
        new_y2 = min(h, y2 + expand_h)
        return new_x1, new_y1, new_x2, new_y2

    def upscale_image(self, image, factor):
        if factor <= 1.0:
            return image
        height, width = image.shape[:2]
        new_height = int(height * factor)
        new_width = int(width * factor)
        return cv2.resize(image, (new_width, new_height), interpolation=cv2.INTER_CUBIC)

    def detect_phones_with_person_focus(self, frame):
        person_results = self.model(frame, verbose=False)[0].boxes
        persons = [box for box in person_results if int(box.cls[0]) == 0]
        detected_phones = []

        crops = []
        crop_infos = []

        for person in persons:
            px1, py1, px2, py2 = map(int, person.xyxy[0].tolist())
            expanded_bbox = self.expand_bbox((px1, py1, px2, py2), frame.shape, self.crop_expand_ratio)
            crop_x1, crop_y1, crop_x2, crop_y2 = expanded_bbox
            person_crop = frame[crop_y1:crop_y2, crop_x1:crop_x2]
            if person_crop.size == 0:
                continue
            if self.upscale_factor > 1.0:
                upscaled_crop = self.upscale_image(person_crop, self.upscale_factor)
            else:
                upscaled_crop = person_crop
            crops.append(upscaled_crop)
            crop_infos.append({
                'crop_x1': crop_x1,
                'crop_y1': crop_y1,
                'person': person
            })

        if crops:
            crop_results = self.model(crops, verbose=False)
            for crop_idx, result in enumerate(crop_results):
                phones_in_crop = [box for box in result.boxes if int(box.cls[0]) == 67]
                info = crop_infos[crop_idx]
                crop_x1 = info['crop_x1']
                crop_y1 = info['crop_y1']
                person = info['person']
                for phone in phones_in_crop:
                    phone_x1, phone_y1, phone_x2, phone_y2 = phone.xyxy[0].tolist()
                    if self.upscale_factor > 1.0:
                        phone_x1 /= self.upscale_factor
                        phone_y1 /= self.upscale_factor
                        phone_x2 /= self.upscale_factor
                        phone_y2 /= self.upscale_factor
                    orig_phone_x1 = crop_x1 + phone_x1
                    orig_phone_y1 = crop_y1 + phone_y1
                    orig_phone_x2 = crop_x1 + phone_x2
                    orig_phone_y2 = crop_y1 + phone_y2
                    detected_phones.append({
                        'bbox': (orig_phone_x1, orig_phone_y1, orig_phone_x2, orig_phone_y2),
                        'conf': phone.conf[0],
                        'associated_person': tuple(map(int, person.xyxy[0].tolist())),
                        'person_conf': person.conf[0]
                    })
        return persons, detected_phones

    def annotate_frame(self, frame, persons, detected_phones, debug=False):
        frame_copy = frame.copy()
        for person in persons:
            x1, y1, x2, y2 = map(int, person.xyxy[0].tolist())
            conf = person.conf[0]
            cv2.rectangle(frame_copy, (x1, y1), (x2, y2), (255, 0, 0), 2)
            if debug:
                cv2.putText(frame_copy, f"Person {conf:.2f}", (x1, y1 - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2)
        for phone_info in detected_phones:
            x1, y1, x2, y2 = map(int, phone_info['bbox'])
            conf = phone_info['conf']
            cv2.rectangle(frame_copy, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)
            if debug:
                cv2.putText(frame_copy, f"Phone {conf:.2f}", (int(x1), int(y1) - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        return frame_copy

    def detect_all_objects(self, frames):
        annotated_frames = []
        for frame in frames:
            results = self.model(frame)
            annotated_frames.append(results[0].plot())
        return annotated_frames

    def frames_to_video(self, frames, output_path, fps=30):
        if not frames:
            return
        h, w, _ = frames[0].shape
        out = cv2.VideoWriter(output_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (w, h))
        for frame in frames:
            out.write(frame)
        out.release()

    def run_detection(self, input_dir, output_dir=None, method='enhanced', debug=False):
        """
        Run detection with specified method - orchestrates all functions
        method options:
        - 'enhanced': Use person-focused high-resolution phone detection
        - 'all_objects': Show all detected objects (YOLO default)
        """
        parent_dir = os.path.dirname(os.path.abspath(input_dir))
        base_output_dir = os.path.join(parent_dir, "phone_detection_output")
        if output_dir:
            base_output_dir = os.path.join(output_dir, "phone_detection_output")
        os.makedirs(base_output_dir, exist_ok=True)

        for filename in os.listdir(input_dir):
            if filename.lower().endswith((".mp4", ".avi", ".mov")):
                video_path = os.path.join(input_dir, filename)
                print(f"Processing {video_path}...")
                frames = self.video_to_frames(video_path)

                if method == 'enhanced':
                    print("Running enhanced detection (person-focused high-res processing)...")
                    annotated_frames = []
                    detection_stats = []
                    for i, frame in enumerate(frames):
                        persons, detected_phones = self.detect_phones_with_person_focus(frame)
                        annotated_frame = self.annotate_frame(frame, persons, detected_phones, debug)
                        annotated_frames.append(annotated_frame)
                        stats = {
                            'frame': i,
                            'persons_detected': len(persons),
                            'phones_detected': len(detected_phones)
                        }
                        detection_stats.append(stats)
                        if debug:
                            print(f"Frame {i}: {len(persons)} persons, {len(detected_phones)} phones detected")
                    enhanced_output_path = os.path.join(base_output_dir, f"enhanced_{filename}")
                    self.frames_to_video(annotated_frames, enhanced_output_path)
                    total_persons = sum(s['persons_detected'] for s in detection_stats)
                    total_phones = sum(s['phones_detected'] for s in detection_stats)
                    print(f"Enhanced method - Total persons: {total_persons}, Total phones: {total_phones}")

                if method == 'all_objects':
                    print("Running all object detection (YOLO default)...")
                    all_objects_annotated = self.detect_all_objects(frames)
                    all_objects_output_path = os.path.join(base_output_dir, f"all_objects_{filename}")
                    self.frames_to_video(all_objects_annotated, all_objects_output_path)

def main():
    parser = argparse.ArgumentParser(description="Enhanced phone detection in videos")
    parser.add_argument("--input-dir", type=str, required=True, help="Path to folder with videos")
    parser.add_argument("--output-dir", type=str, default=None, help="Optional output base dir")
    parser.add_argument("--method", type=str, choices=['enhanced', 'all_objects'],
                       default='enhanced', help="Detection method to use")
    parser.add_argument("--crop-expand", type=float, default=0.3,
                       help="Ratio to expand person bbox for cropping (default: 0.3)")
    parser.add_argument("--upscale-factor", type=float, default=0.5,
                       help="Factor to upscale cropped regions (default: 2.0)")
    parser.add_argument("--debug", action="store_true", help="Enable debug output")
    args = parser.parse_args()

    detector = EnhancedPhoneDetection(
        crop_expand_ratio=args.crop_expand,
        upscale_factor=args.upscale_factor
    )
    detector.run_detection(args.input_dir, args.output_dir, args.method, args.debug)

if __name__ == "__main__":
    main()