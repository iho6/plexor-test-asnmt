# test_detect_held_phone.py

import os
import cv2
import numpy as np
import tempfile
import shutil
import pytest

from scripts.run_detect_help_phone import HeldPhoneDetection, main


@pytest.fixture
def dummy_video(tmp_path):
    """Create a short dummy video for testing."""
    video_path = tmp_path / "dummy.mp4"
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out = cv2.VideoWriter(str(video_path), fourcc, 5.0, (64, 64))

    # create 10 solid-color frames
    for i in range(10):
        frame = np.full((64, 64, 3), i * 20 % 255, dtype=np.uint8)
        out.write(frame)
    out.release()
    return str(video_path)


def test_video_to_frames(dummy_video):
    detector = HeldPhoneDetection(model_path=None)
    frames = detector.video_to_frames(dummy_video)
    assert isinstance(frames, list)
    assert len(frames) == 10
    assert isinstance(frames[0], np.ndarray)
    assert frames[0].shape == (64, 64, 3)


def test_boxes_overlap_true():
    box1 = [0, 0, 50, 50]
    box2 = [25, 25, 75, 75]
    assert HeldPhoneDetection._boxes_overlap(box1, box2) is True


def test_boxes_overlap_false():
    box1 = [0, 0, 20, 20]
    box2 = [30, 30, 50, 50]
    assert HeldPhoneDetection._boxes_overlap(box1, box2) is False


def test_boxes_overlap_edge_case():
    # touching edges only, not overlapping
    box1 = [0, 0, 10, 10]
    box2 = [10, 10, 20, 20]
    assert HeldPhoneDetection._boxes_overlap(box1, box2) is False


def test_phone_person_overlap_with_overlap(monkeypatch):
    detector = HeldPhoneDetection(model_path=None)

    # Mock YOLO predict
    def fake_predict(frames, verbose=False):
        return [[
            type("Box", (), {
                "boxes": type("Boxes", (), {
                    "xyxy": np.array([[0, 0, 20, 20], [10, 10, 30, 30]]),
                    "cls": np.array([0, 1])  # 0=person, 1=phone
                })()
            })()
        ]]

    monkeypatch.setattr(detector.model, "predict", fake_predict)
    frames = [np.zeros((64, 64, 3), dtype=np.uint8)]
    results = detector.phone_person_overlap(frames)
    assert results[0][1] is True


def test_phone_person_overlap_without_overlap(monkeypatch):
    detector = HeldPhoneDetection(model_path=None)

    def fake_predict(frames, verbose=False):
        return [[
            type("Box", (), {
                "boxes": type("Boxes", (), {
                    "xyxy": np.array([[0, 0, 10, 10], [30, 30, 40, 40]]),
                    "cls": np.array([0, 1])  # person & phone no overlap
                })()
            })()
        ]]

    monkeypatch.setattr(detector.model, "predict", fake_predict)
    frames = [np.zeros((64, 64, 3), dtype=np.uint8)]
    results = detector.phone_person_overlap(frames)
    assert results[0][1] is False


def test_phone_person_overlap_no_phone(monkeypatch):
    detector = HeldPhoneDetection(model_path=None)

    def fake_predict(frames, verbose=False):
        return [[
            type("Box", (), {
                "boxes": type("Boxes", (), {
                    "xyxy": np.array([[0, 0, 10, 10]]),
                    "cls": np.array([0])  # only person
                })()
            })()
        ]]

    monkeypatch.setattr(detector.model, "predict", fake_predict)
    frames = [np.zeros((64, 64, 3), dtype=np.uint8)]
    results = detector.phone_person_overlap(frames)
    assert results[0][1] is False


def test_detect_phone_and_annotate_creates_video(dummy_video, tmp_path):
    detector = HeldPhoneDetection(model_path=None)

    frames = detector.video_to_frames(dummy_video)
    results_list = [(frame, [], True) for frame in frames]

    output_video = tmp_path / "out.mp4"
    detector.detect_phone_and_annotate(results_list, str(output_video))

    assert output_video.exists()
    cap = cv2.VideoCapture(str(output_video))
    assert int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) == 10
    cap.release()


def test_run_creates_output_folder(dummy_video, tmp_path):
    input_dir = tmp_path / "videos"
    input_dir.mkdir()
    shutil.copy(dummy_video, input_dir / "test.mp4")

    output_dir = tmp_path / "outputs"

    detector = HeldPhoneDetection(model_path=None)
    detector.run_held_phone_detection(str(input_dir), str(output_dir))

    out_file = output_dir / "phone_detections" / "test_annotated.mp4"
    assert out_file.exists()


def test_cli_requires_input_dir(capsys):
    with pytest.raises(SystemExit):
        main([])  # no args
    captured = capsys.readouterr()
    assert "usage" in captured.err.lower()


def test_cli_runs_with_input_dir(dummy_video, tmp_path):
    input_dir = tmp_path / "cli_videos"
    input_dir.mkdir()
    shutil.copy(dummy_video, input_dir / "clip.mp4")

    args = ["--input-dir", str(input_dir)]
    main(args)

    out_file = input_dir / "phone_detections" / "clip_annotated.mp4"
    assert out_file.exists()
