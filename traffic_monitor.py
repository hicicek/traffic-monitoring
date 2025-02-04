import argparse
from typing import Dict, Iterable, List, Set, Tuple

import cv2
import numpy as np
from tqdm import tqdm

import supervision as sv
from inference import get_model

COLORS = sv.ColorPalette.from_hex(["#E6194B", "#3CB44B", "#FFE119", "#3C76D1"])

zoneInPointsArray = [[(110, 1080), (570, 820)],
                     [(1520, 700), (1920, 510)],
                     [(700, 340), (850, 110)]]
zoneOutPointsArray = [[(100, 800), (440, 510)],
                      [(1540, 950), (1920, 720)],
                      [(900, 340), (1080, 110)]]


def ConvertToNpArray(pointsArray):
    return np.array([[pointsArray[0][0], pointsArray[0][1]], [pointsArray[1][0], pointsArray[0][1]],
                     [pointsArray[1][0], pointsArray[1][1]], [pointsArray[0][0], pointsArray[1][1]]])


ZONE_IN_POLYGONS = []

for point in zoneInPointsArray:
    ZONE_IN_POLYGONS.append(ConvertToNpArray(point))

ZONE_OUT_POLYGONS = []

for point in zoneOutPointsArray:
    ZONE_OUT_POLYGONS.append(ConvertToNpArray(point))

modelNames = ["bus", "car", "truck", "van"]

myClasses = []
for i in range(100):
    myClasses.append(1)


class DetectionsManager:
    def __init__(self) -> None:
        self.tracker_id_to_zone_id: Dict[int, list] = {}
        self.counts: Dict[int, Dict[int, Dict[str, set[int]]]] = {}

    def update(
            self,
            detections_all: sv.Detections,
            detections_in_zones: List[sv.Detections],
            detections_out_zones: List[sv.Detections],
    ) -> sv.Detections:
        for i in range(len(detections_in_zones)):
            for x, tracker_id in enumerate(detections_in_zones[i].tracker_id):
                self.tracker_id_to_zone_id.setdefault(tracker_id, [modelNames[detections_in_zones[i].class_id[x]], i])

        for zone_out_id, detections_out_zone in enumerate(detections_out_zones):
            for tracker_id in detections_out_zone.tracker_id:
                if tracker_id in self.tracker_id_to_zone_id:
                    zone_in_id = self.tracker_id_to_zone_id[tracker_id][1]
                    self.counts.setdefault(zone_out_id, {})
                    self.counts[zone_out_id].setdefault(zone_in_id, {})
                    self.counts[zone_out_id][zone_in_id].setdefault(self.tracker_id_to_zone_id[tracker_id][0], set())
                    self.counts[zone_out_id][zone_in_id][self.tracker_id_to_zone_id[tracker_id][0]].add(tracker_id)

        for x, tracker_id in enumerate(detections_all.tracker_id):
            if tracker_id in self.tracker_id_to_zone_id.keys():
                myClasses[x] = detections_all.class_id[x]
                detections_all.class_id[x] = self.tracker_id_to_zone_id[tracker_id][1]
            else:
                detections_all.class_id[x] = -1
        return detections_all[detections_all.class_id != -1]


def initiate_polygon_zones(
        polygons: List[np.ndarray],
        frame_resolution_wh: Tuple[int, int],
        triggering_anchors: Iterable[sv.Position] = [sv.Position.CENTER],
) -> List[sv.PolygonZone]:
    return [
        sv.PolygonZone(
            polygon=polygon,
            frame_resolution_wh=frame_resolution_wh,
            triggering_anchors=triggering_anchors,
        )
        for polygon in polygons
    ]


class VideoProcessor:
    def __init__(
            self,
            source_video_path: str,
            target_video_path: str = None,
            confidence_threshold: float = 0.3,
            iou_threshold: float = 0.7,
    ) -> None:
        self.conf_threshold = confidence_threshold
        self.iou_threshold = iou_threshold
        self.source_video_path = source_video_path
        self.target_video_path = target_video_path

        self.detection_classes = np.zeros(1)

        self.model = get_model("traffic-1zhxh/2", api_key="N4waJrKwTzqem4cTzdS2")
        self.tracker = sv.ByteTrack()

        self.video_info = sv.VideoInfo.from_video_path(source_video_path)
        self.zones_in = initiate_polygon_zones(
            ZONE_IN_POLYGONS, self.video_info.resolution_wh, [sv.Position.CENTER]
        )
        self.zones_out = initiate_polygon_zones(
            ZONE_OUT_POLYGONS, self.video_info.resolution_wh, [sv.Position.CENTER]
        )

        self.bounding_box_annotator = sv.BoundingBoxAnnotator(color=COLORS)
        self.label_annotator = sv.LabelAnnotator(
            color=COLORS, text_color=sv.Color.BLACK
        )
        self.trace_annotator = sv.TraceAnnotator(
            color=COLORS, position=sv.Position.CENTER, trace_length=100, thickness=2
        )
        self.detections_manager = DetectionsManager()

    def process_video(self):
        frame_generator = sv.get_video_frames_generator(
            source_path=self.source_video_path
        )

        if self.target_video_path:
            with sv.VideoSink(self.target_video_path, self.video_info) as sink:
                for frame in tqdm(frame_generator, total=self.video_info.total_frames):
                    annotated_frame = self.process_frame(frame)
                    sink.write_frame(annotated_frame)
        else:
            for frame in tqdm(frame_generator, total=self.video_info.total_frames):
                annotated_frame = self.process_frame(frame)
                cv2.imshow("Processed Video", annotated_frame)
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break
            cv2.destroyAllWindows()

    def annotate_frame(
            self, frame: np.ndarray, detections: sv.Detections
    ) -> np.ndarray:
        annotated_frame = frame.copy()
        for i, (zone_in, zone_out) in enumerate(zip(self.zones_in, self.zones_out)):
            annotated_frame = sv.draw_polygon(
                annotated_frame, zone_in.polygon, COLORS.colors[i]
            )
            annotated_frame = sv.draw_polygon(
                annotated_frame, zone_out.polygon, COLORS.colors[i]
            )

        labels = [f"#{tracker_id} " for tracker_id in detections.tracker_id]
        for x in range(0, len(detections)):
            labels[x] = labels[x] + str(modelNames[myClasses[x]])
        annotated_frame = self.trace_annotator.annotate(annotated_frame, detections)
        annotated_frame = self.bounding_box_annotator.annotate(
            annotated_frame, detections
        )
        annotated_frame = self.label_annotator.annotate(
            annotated_frame, detections, labels
        )

        for zone_out_id, zone_out in enumerate(self.zones_out):
            x = 0
            zone_center = sv.get_polygon_center(polygon=zone_out.polygon)
            zone_center.y = zone_center.y - 120
            if zone_out_id in self.detections_manager.counts:
                counts = self.detections_manager.counts[zone_out_id]
                for zone_in_id in counts:
                    for className in counts[zone_in_id].keys():
                        x = x + 1
                        count = len(counts[zone_in_id][className])
                        text_anchor = sv.Point(x=zone_center.x, y=zone_center.y + 40 * x)
                        annotated_frame = sv.draw_text(
                            scene=annotated_frame,
                            text=str(count) + " " + className,
                            text_anchor=text_anchor,
                            background_color=COLORS.colors[zone_in_id],
                        )

        return annotated_frame

    def process_frame(self, frame: np.ndarray) -> np.ndarray:
        # results = self.model(
        #     frame, verbose=False, conf=self.conf_threshold, iou=self.iou_threshold
        # )[0]
        results = self.model.infer(frame, confidence=self.conf_threshold, iou_threshold=self.iou_threshold)
        # detections = sv.Detections.from_ultralytics(results)
        detections = sv.Detections.from_inference(results[0].dict(by_alias=True, exclude_none=True))

        detections = self.tracker.update_with_detections(detections)

        detections_in_zones = []
        detections_out_zones = []

        for zone_in, zone_out in zip(self.zones_in, self.zones_out):
            detections_in_zone = detections[zone_in.trigger(detections=detections)]
            detections_in_zones.append(detections_in_zone)
            detections_out_zone = detections[zone_out.trigger(detections=detections)]
            detections_out_zones.append(detections_out_zone)

        detections = self.detections_manager.update(
            detections, detections_in_zones, detections_out_zones
        )
        return self.annotate_frame(frame, detections)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Traffic Flow Analysis with YOLO and ByteTrack"
    )

    parser.add_argument(
        "--source_video_path",
        required=True,
        help="Path to the source video file",
        type=str,
    )
    parser.add_argument(
        "--target_video_path",
        default=None,
        help="Path to the target video file (output)",
        type=str,
    )
    parser.add_argument(
        "--confidence_threshold",
        default=0.3,
        help="Confidence threshold for the model",
        type=float,
    )
    parser.add_argument(
        "--iou_threshold", default=0.7, help="IOU threshold for the model", type=float
    )

    args = parser.parse_args()
    processor = VideoProcessor(
        source_video_path=args.source_video_path,
        target_video_path=args.target_video_path,
        confidence_threshold=args.confidence_threshold,
        iou_threshold=args.iou_threshold,
    )
    processor.process_video()