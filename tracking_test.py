import numpy as np
import supervision as sv
from ultralytics import YOLO
import utils
from sklearn.cluster import KMeans

model = YOLO(model="yolov8n.pt")
tracker = sv.ByteTrack()
box_annotator = sv.BoundingBoxAnnotator()
label_annotator = sv.LabelAnnotator()
trace_annotator = sv.TraceAnnotator()

kmeans = KMeans(n_clusters=2)
labels_k_means_names = ['teamA', 'teamB']

def callback(frame: np.ndarray, n_frame: int) -> np.ndarray:
    results = model(frame, device='mps')[0]
    detections = sv.Detections.from_ultralytics(results)
    detections = tracker.update_with_detections(detections)
    if len(detections) == 0: return frame

    if n_frame == 0:
        train_images = [
            frame[int(xyxy[1]):int(xyxy[3]), int(xyxy[0]):int(xyxy[2])]
            for xyxy
            in detections.xyxy
        ]
        train_features = utils.get_hist_features(train_images)
        kmeans.fit(train_features)

    test_images = [
        frame[max(int(xyxy[1]), 0):max(int(xyxy[3]), 0), max(int(xyxy[0]), 0):max(int(xyxy[2]), 0)]
        for xyxy
        in detections.xyxy
    ]
    test_features = utils.get_hist_features(test_images)

    labels_k_means = kmeans.predict(test_features)

    # Output
    labels = [
        f"#{tracker_id} {labels_k_means_names[label_k]}"
        for tracker_id, label_k
        in zip(detections.tracker_id, labels_k_means)
    ]
    annotated_frame = box_annotator.annotate(
        frame.copy(), detections=detections)
    annotated_frame = label_annotator.annotate(
        annotated_frame, detections=detections, labels=labels)
    return trace_annotator.annotate(
        annotated_frame, detections=detections)

sv.process_video(
    source_path="/Users/jonino/Documents/personal/cv/ml6/senior-ml-engineer-challenge/sample.mp4",
    target_path="/Users/jonino/tests/ml6/result_ml6_challenge.mp4",
    callback=callback
)