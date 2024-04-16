import numpy as np
import supervision as sv
from ultralytics import YOLO
import utils
from sklearn.cluster import KMeans
from sklearn.mixture import GaussianMixture
import cv2
from joblib import dump, load

model_det = YOLO(model="yolov8n-seg.pt")
tracker = sv.ByteTrack()
box_annotator = sv.BoundingBoxAnnotator()
label_annotator = sv.LabelAnnotator()
trace_annotator = sv.TraceAnnotator()
mask_annotator = sv.MaskAnnotator()
polygon_annotator = sv.PolygonAnnotator()
text_anchor = sv.Point(x=510, y=50)
bg_color = sv.Color.BLACK
fg_color = sv.Color.WHITE

# Open the video file and Get the total number of frames
source_path="/Users/jonino/Documents/personal/cv/ml6/senior-ml-engineer-challenge/sample.mp4"
cap = cv2.VideoCapture(source_path)
frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
cap.release()

#
path_out = '/Users/jonino/tests/ml6'
store_clusters = True

#
train_until_frame = 50   #50 #frame_count-1
training_tracking = False
testing_tracking = False
train_features_full = []

#Using conrtrastive learning model for features
method = 'hist'  #'hist', 'bag', 'ae'
model_cl_path = '/Users/jonino/src/teamId/trained_models/embedding.pth'
model_cl = utils.load_model_embed(model_cl_path)

#Classifier
#labels_k_means_names = ['teamA', 'teamB', 'referee', 'fans']
labels_k_means_names = ['teamA', 'teamB']
n_clusters = len(labels_k_means_names)
kmeans = KMeans(n_clusters=n_clusters)
kmeans_saved_path = '/Users/jonino/tests/ml6/mvp_1/K2f50/colors_kmeans_clusters.joblib'    #'/Users/jonino/tests/ml6/mvp_1/colors_kmeans_clusters.joblib'    #None

def get_bag_features(train_images, test_images, K=35, h=64, pixel_number=400000):
    # learn bag of colors on the current game
    flat = []
    for i, c in enumerate(train_images):
        c = c[:h, :, :]
        r = c[:, :, 0].flatten()
        b = c[:, :, 1].flatten()
        g = c[:, :, 2].flatten()
        temp = np.dstack((r, b, g))
        non_black_pixels_mask = np.any(temp != [0, 0, 0], axis=-1)
        flat.extend(temp[non_black_pixels_mask].tolist())
        if len(flat) > pixel_number:
            break
    if len(flat) > pixel_number:
        flat = flat[:pixel_number]
    gmm = GaussianMixture(n_components=K).fit(flat)

    X_train = utils.get_hist_from_gmm(gmm, train_images, K=K)
    X_test = utils.get_hist_from_gmm(gmm, test_images, K=K) if test_images is not None else None
    return X_train, X_test

def callback_training(frame: np.ndarray, n_frame: int) -> np.ndarray:
    results = model_det(frame, device='mps')[0]
    detections = sv.Detections.from_ultralytics(results)
    detections = detections[detections.class_id == 0]   #Only people
    detections = detections[detections.area > 0]  # Only with valid segmentation masks
    if training_tracking: detections = tracker.update_with_detections(detections)

    if len(detections) > 0:
        train_images = [
            frame[max(int(xyxy[1]), 0):max(int(xyxy[3]), 0), max(int(xyxy[0]), 0):max(int(xyxy[2]), 0)]
            for xyxy
            in detections.xyxy
        ]
        train_masks = [
            mask[max(int(xyxy[1]), 0):max(int(xyxy[3]), 0), max(int(xyxy[0]), 0):max(int(xyxy[2]), 0)]
            for xyxy, mask
            in zip(detections.xyxy, detections.mask)
        ]

        if (method == 'hist'): train_features = utils.get_mask_hist_features(train_images, train_masks)
        elif (method == 'bag'): train_features, _ = get_bag_features(train_images, None)
        else: train_features = utils.get_features(train_images, model_cl)

        train_features_full.extend(train_features)

    if n_frame == train_until_frame:
        print('Fitting kmeans...')
        kmeans.fit(train_features_full)
        if store_clusters: dump(kmeans, '{}/colors_kmeans_clusters.joblib'.format(path_out))

    # Output
    labels = [
        f"#{tracker_id} {results.names[class_id]}"
        for class_id, tracker_id
        in zip(detections.class_id, detections.tracker_id)
    ] if training_tracking else [
        f"#{results.names[class_id]}"
        for class_id
        in detections.class_id
    ]

    annotated_frame = box_annotator.annotate(
        frame.copy(), detections=detections)
    annotated_frame = label_annotator.annotate(
        annotated_frame, detections=detections, labels=labels)
    annotated_frame = mask_annotator.annotate(
        annotated_frame, detections=detections)
    return trace_annotator.annotate(annotated_frame, detections=detections) if training_tracking else annotated_frame

def callback_testing(frame: np.ndarray, n_frame: int) -> np.ndarray:
    results = model_det(frame, device='mps')[0]
    detections = sv.Detections.from_ultralytics(results)
    detections = detections[detections.class_id == 0]  # Only people
    detections = detections[detections.area>0]  # Only with valid segmentation masks
    if testing_tracking: detections = tracker.update_with_detections(detections)
    if len(detections) == 0: return frame

    test_images = [
        frame[max(int(xyxy[1]), 0):max(int(xyxy[3]), 0), max(int(xyxy[0]), 0):max(int(xyxy[2]), 0)]
        for xyxy
        in detections.xyxy
    ]
    test_masks = [
        mask[max(int(xyxy[1]), 0):max(int(xyxy[3]), 0), max(int(xyxy[0]), 0):max(int(xyxy[2]), 0)]
        for xyxy, mask
        in zip(detections.xyxy, detections.mask)
    ]

    if (method == 'hist'): test_features = utils.get_mask_hist_features(test_images, test_masks)
    elif (method == 'bag'): train_features, test_features = get_bag_features(train_images, test_images)
    else: test_features = utils.get_features(test_images, model_cl)

    labels_k_means = kmeans.predict(test_features)

    # Output
    detections.class_id = labels_k_means
    labels = [
        f"#{tracker_id} {labels_k_means_names[label_k]}"
        for tracker_id, label_k
        in zip(detections.tracker_id, labels_k_means)
    ] if testing_tracking else [
        f"#{labels_k_means_names[label_k]}"
        for label_k
        in labels_k_means
    ]
    annotated_frame = box_annotator.annotate(
        frame.copy(), detections=detections)
    annotated_frame = label_annotator.annotate(
        annotated_frame, detections=detections, labels=labels)
    annotated_frame = polygon_annotator.annotate(
        annotated_frame, detections=detections)

    N_0 = sum(labels_k_means==0)
    N_1 = sum(labels_k_means == 1)
    annotated_frame = sv.draw_text(scene=annotated_frame, text="{} {} - {} {}".format(labels_k_means_names[0], N_0,labels_k_means_names[1], N_1), text_anchor=text_anchor, background_color=bg_color, text_color=fg_color)
    return trace_annotator.annotate(annotated_frame, detections=detections) if testing_tracking else annotated_frame

if kmeans_saved_path is None:
    sv.process_video(
        source_path=source_path,
        target_path="{}/result_detection_seg.mp4".format(path_out),
        callback=callback_training
    )
else:
    print('Will use existing KMeans from {}...'.format(kmeans_saved_path))
    kmeans = load(kmeans_saved_path)

tracker.reset()

sv.process_video(
    source_path=source_path,
    target_path="{}/result_ml6_challenge_hist_K2_f50_loaded.mp4".format(path_out),
    callback=callback_testing
)