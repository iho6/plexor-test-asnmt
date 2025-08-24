import os
import cv2
import argparse
from ultralytics import YOLO

class HeldPhoneDetection:
    def __init__(self, model_name="yolov8n.pt"):
        # Load YOLO model
        self.model = YOLO(model_name)

    def video_to_frames(self, video_path):
        """Convert video into list of frames."""
        cap = cv2.VideoCapture(video_path)
        frames = []
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            frames.append(frame)
        cap.release()
        return frames

    def phone_person_overlap(self, frames):
        """Check overlap between detected phones and persons for each frame."""
        results_list = []

        for frame in frames:
            # Restrict detection to person (0) and phone (67) only
            results = self.model(frame, classes=[0, 67])

            persons = []
            phones = []

            # Extract detections
            for box in results[0].boxes:
                cls = int(box.cls[0].item())
                xyxy = box.xyxy[0].tolist()
                if cls == 0:  # person
                    persons.append(xyxy)
                elif cls == 67:  # phone
                    phones.append(xyxy)

            overlap_found = False
            for px1, py1, px2, py2 in persons:
                for fx1, fy1, fx2, fy2 in phones:
                    # Compute overlap area
                    ix1 = max(px1, fx1)
                    iy1 = max(py1, fy1)
                    ix2 = min(px2, fx2)
                    iy2 = min(py2, fy2)
                    if ix1 < ix2 and iy1 < iy2:  # overlap exists
                        overlap_found = True
                        break
                if overlap_found:
                    break

            results_list.append((frame, overlap_found))

        return results_list

    def detect_phone(self, frame, overlap_found):
        """Draw bounding boxes if overlap found, else return original frame."""
        if not overlap_found:
            return frame

        # Run detection again to get bounding boxes (restricted to person & phone)
        results = self.model(frame, classes=[0, 67])
        annotated_frame = results[0].plot()
        return annotated_frame

    def run_held_phone_detection(self, input_dir, output_dir=None):
        """Main pipeline: process all videos in a directory."""
        if output_dir is None:
            output_dir = os.path.join(input_dir, "phone_detections")
        os.makedirs(output_dir, exist_ok=True)

        for file_name in os.listdir(input_dir):
            if not file_name.lower().endswith((".mp4", ".avi", ".mov")):
                continue

            video_path = os.path.join(input_dir, file_name)
            print(f"Processing: {video_path}")
            frames = self.video_to_frames(video_path)
            frame_results = self.phone_person_overlap(frames)

            # Save processed video
            output_path = os.path.join(output_dir, file_name)
            fourcc = cv2.VideoWriter_fourcc(*"mp4v")
            height, width = frames[0].shape[:2]
            out = cv2.VideoWriter(output_path, fourcc, 20.0, (width, height))

            for frame, overlap_found in frame_results:
                out.write(self.detect_phone(frame, overlap_found))

            out.release()
            print(f"Saved: {output_path}")


def main():
    parser = argparse.ArgumentParser(description="Detect hand-held phones in videos.")
    parser.add_argument("--input-dir", type=str, required=True,
                        help="Directory containing input videos")
    parser.add_argument("--output-dir", type=str, default=None,
                        help="Optional output directory for processed videos")
    args = parser.parse_args()

    detector = HeldPhoneDetection()
    detector.run_held_phone_detection(args.input_dir, args.output_dir)


if __name__ == "__main__":
    main()
