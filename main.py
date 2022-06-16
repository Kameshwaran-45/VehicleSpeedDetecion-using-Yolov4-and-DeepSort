import warnings
import cv2
import json
import numpy as np

from timeit import time
from collections import deque, defaultdict

from yolo import YOLO
from deep_sort import preprocessing
from deep_sort import nn_matching
from deep_sort.detection import Detection
from deep_sort.tracker import Tracker
from counting.trackableobject import TrackableObject

pts = [deque(maxlen=30) for _ in range(9999)]
warnings.filterwarnings('ignore')

with open('./config/config.json') as json_file:
    config_file = json.load(json_file)

#  setting up axis (either topdown or leftright) (configure as per the requirement)
axis = config_file["axis"]

#  setting up the range for the threshold line (configure as per the requirement)
range1 = config_file["range"]["1"]
range2 = config_file["range"]["2"]
range3 = config_file["range"]["3"]
range4 = config_file["range"]["4"]
range5 = config_file["range"]["5"]
range6 = config_file["range"]["6"]

#  setting up the tolerance between the lines
tolerance = config_file["tolerance"]

objects = {}

time_track = defaultdict(dict)

trackers = []
trackableObjects = {}


def speedDetection(time):
    distance = 7.5
    speed = float(distance / time)
    return speed * 3.6


def speed_det(objects, frame, fps, inp_fps):
    # loop over the tracked objects
    for (objectID, centroid) in objects.items():

        to = trackableObjects.get(objectID, None)
        if to is None:
            to = TrackableObject(objectID, centroid)
        else:
            y = [c[axis] for c in to.centroids]
            direction = centroid[axis] - np.mean(y)
            to.centroids.append(centroid)

            if not to.counted:

                if direction < 0:
                    if centroid[axis] in range(range1 - tolerance,range1 + tolerance) and objectID not in time_track.keys():
                        time_track[objectID]["lane1"] = time.time()

                    elif centroid[axis] in range(range2 - tolerance, range2 + tolerance):
                        if "lane1" in time_track[objectID].keys() and "lane2" not in time_track[objectID].keys():
                            time_track[objectID]["lane2"] = time.time()
                            sec1 = float(time_track[objectID]["lane2"] - time_track[objectID]["lane1"])
                            fp = 1 / fps
                            inp = (1 / inp_fps)
                            sec1 = (inp * sec1) / fp
                            speed1 = round(speedDetection(sec1), 3)
                            time_track[objectID]["speed"] = speed1
                            to.counted = True
                        elif "lane1" not in time_track[objectID].keys() and "lane2" not in time_track[objectID].keys():
                            time_track[objectID]["lane2"] = time.time()

                    elif centroid[axis] in range(range3 - tolerance, range3 + tolerance):
                        if "lane2" in time_track[objectID].keys() and "speed" not in time_track[objectID].keys():
                            time_track[objectID]["lane3"] = time.time()
                            sec2 = float(time_track[objectID]["lane3"] - time_track[objectID]["lane2"])
                            fp = 1 / fps
                            inp = (1 / inp_fps)
                            sec2 = (inp * sec2) / fp
                            speed2 = round(speedDetection(sec2), 3)
                            time_track[objectID]["speed"] = speed2
                            to.counted = True

                elif direction > 0:
                    if centroid[axis] in range(range4 - tolerance,range4 + tolerance) and objectID not in time_track.keys():
                        time_track[objectID]["lane1"] = time.time()

                    elif centroid[axis] in range(range5 - tolerance, range5 + tolerance):
                        if "lane1" in time_track[objectID].keys() and "lane2" not in time_track[objectID].keys():
                            time_track[objectID]["lane2"] = time.time()
                            sec1 = float(time_track[objectID]["lane2"] - time_track[objectID]["lane1"])
                            fp = 1 / fps
                            inp = (1 / inp_fps)
                            sec1 = (inp * sec1) / fp
                            speed1 = round(speedDetection(sec1), 3)
                            time_track[objectID]["speed"] = speed1
                            to.counted = True
                        elif "lane1" not in time_track[objectID].keys() and "lane2" not in time_track[objectID].keys():
                            time_track[objectID]["lane2"] = time.time()

                    elif centroid[axis] in range(range6 - tolerance, range6 + tolerance):
                        if "lane2" in time_track[objectID].keys() and "speed" not in time_track[objectID].keys():
                            time_track[objectID]["lane3"] = time.time()
                            sec2 = float(time_track[objectID]["lane3"] - time_track[objectID]["lane2"])
                            fp = 1 / fps
                            inp = (1 / inp_fps)
                            sec2 = (inp * sec2) / fp
                            speed2 = round(speedDetection(sec2), 3)
                            time_track[objectID]["speed"] = speed2
                            to.counted = True

        trackableObjects[objectID] = to

def main(yolo):

    # Definition of the parameters
    max_cosine_distance = 0.5
    nn_budget = None
    nms_max_overlap = 0.3

    counter = []

    metric = nn_matching.NearestNeighborDistanceMetric("cosine", max_cosine_distance, nn_budget)
    tracker = Tracker(metric)

    if config_file["save_video"] == "true":
        writeVideo_flag = True
    else:
        writeVideo_flag = False

    video_capture = cv2.VideoCapture(config_file["input"])

    if writeVideo_flag:
        # Define the codec and create VideoWriter object
        w = int(video_capture.get(3))
        h = int(video_capture.get(4))
        fourcc = cv2.VideoWriter_fourcc(*'MJPG')
        out = cv2.VideoWriter('output1.avi', fourcc, 15, (w, h))

    fps = 0.0

    while True:
        inp_fps = video_capture.get(cv2.CAP_PROP_FPS)
        ret, frame = video_capture.read()  # frame shape 640*480*3
        if ret != True:
            break
        #  To draw ROI lines (configure as per the requirement)
        cv2.line(frame, (1027, 413), (772, 446), (0, 255, 255), 2)
        cv2.line(frame, (963, 340), (748, 363), (0, 255, 255), 2)
        cv2.line(frame, (925, 295), (733, 313), (0, 255, 255), 2)
        cv2.line(frame, (561, 454), (313, 453), (0, 255, 255), 2)
        cv2.line(frame, (573, 362), (367, 362), (0, 255, 255), 2)
        cv2.line(frame, (588, 308), (406, 308), (0, 255, 255), 2)
        t1 = time.time()

        boxs, class_names = yolo.detect_image(frame)
        detections = []

        for box, class_name in zip(boxs,  class_names):
            detections.append(Detection(box, class_name, [1]))

        # Load image and generate detections.
        detections = [d for d in detections if d.confidence >= 0.5]
        # Run non-maxima suppression.
        boxes = np.array([d.tlwh for d in detections])
        scores = np.array([d.confidence for d in detections])
        indices = preprocessing.non_max_suppression(boxes, nms_max_overlap, scores)
        detections = [detections[i] for i in indices]

        # Call the tracker
        tracker.predict()
        tracker.update(detections)

        i = int(0)
        indexIDs = []
        for det in detections:
            bbox = det.to_tlbr()

        for track in tracker.tracks:
            if not track.is_confirmed() or track.time_since_update > 1:
                continue
            indexIDs.append(int(track.track_id))
            counter.append(int(track.track_id))
            bbox = track.to_tlbr()
            color = (0, 255, 0)

            cv2.rectangle(frame, (int(bbox[0]), int(bbox[1])), (int(bbox[2]), int(bbox[3])), (color), 3)
            # cv2.putText(frame, str(track.track_id), (int(bbox[0]), int(bbox[1] - 50)), 0, 5e-3 * 150, (color), 2)

            i += 1
            center = (int(((bbox[0]) + (bbox[2])) / 2), int(((bbox[1]) + (bbox[3])) / 2))
            if track.track_id in time_track.keys() and "speed" in time_track[track.track_id].keys():
                speed = time_track[track.track_id]["speed"]
                cv2.putText(frame, f"{speed} km/h",
                            (int(((bbox[0]) + (bbox[2])) / 2) - 50, int(((bbox[1]) + (bbox[3])) / 2) - 50),
                            cv2.FONT_HERSHEY_SIMPLEX, 1,
                            (236, 240, 94), 2)

            pts[track.track_id].append(center)
            thickness = 5
            # center point
            cv2.circle(frame, (center), 1, color, thickness)
            objects[track.track_id] = center

            # draw motion path
            if config_file["motion_draw"] == "true":
                for j in range(1, len(pts[track.track_id])):
                    if pts[track.track_id][j - 1] is None or pts[track.track_id][j] is None:
                        continue
                    thickness = int(np.sqrt(64 / float(j + 1)) * 2)
                    cv2.line(frame, (pts[track.track_id][j - 1]), (pts[track.track_id][j]), (color), thickness)

        count = len(set(counter))
        speed_det(objects, frame, fps, inp_fps)
        cv2.putText(frame, "FPS: %f" % (fps), (int(20), int(60)), 0, 5e-3 * 200, (255, 0, 0), 2)
        cv2.imshow('Vehicle_speed_detection', frame)

        if writeVideo_flag:
            # save a frame
            out.write(frame)

        fps = (fps + (1. / (time.time() - t1))) / 2

        # Press Q to stop!
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    video_capture.release()

    if writeVideo_flag:
        out.release()
    cv2.destroyAllWindows()


if __name__ == '__main__':
    main(YOLO(config_file))
