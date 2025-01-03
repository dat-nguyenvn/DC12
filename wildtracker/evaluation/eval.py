import motmetrics as mm
import pandas as pd

def load_mot_results(gt_path, tracker_path):
    """
    Load ground truth and tracker results into DataFrames.
    """
    gt = pd.read_csv(gt_path, header=None, names=[
        'FrameId', 'Id', 'X', 'Y', 'Width', 'Height', 'Confidence', 'Class', 'Visibility',"Attributes"
    ])#frame_id,object_id,xmin,ymin,width,height,confidence,-1,-1,-1

    tracker = pd.read_csv(tracker_path, header=None, names=[
        'FrameId', 'Id', 'X', 'Y', 'Width', 'Height', 'Confidence', 'Class', 'Visibility',"Attributes"
    ])
    gt['Id'] = pd.to_numeric(gt['Id'], errors='coerce')
    tracker['Id'] = pd.to_numeric(tracker['Id'], errors='coerce')

    # Remove rows where 'Id' is NaN (invalid entries)
    gt = gt.dropna(subset=['Id'])
    tracker = tracker.dropna(subset=['Id'])

    return gt, tracker

def evaluate_mot(gt_path, tracker_path):
    """
    Evaluate a tracker using MOT metrics.
    """
    gt, tracker = load_mot_results(gt_path, tracker_path)

    # Convert to MOTMetrics format
    acc = mm.MOTAccumulator(auto_id=True)
    for frame_id in sorted(gt['FrameId'].unique()):
        gt_frame = gt[gt['FrameId'] == frame_id]
        tr_frame = tracker[tracker['FrameId'] == frame_id]

        gt_ids = gt_frame['Id'].tolist()
        tr_ids = tr_frame['Id'].tolist()

        gt_boxes = gt_frame[['X', 'Y', 'Width', 'Height']].values
        tr_boxes = tr_frame[['X', 'Y', 'Width', 'Height']].values

        distances = mm.distances.iou_matrix(gt_boxes, tr_boxes, max_iou=0.9)
        acc.update(gt_ids, tr_ids, distances)

    # Compute metrics
    mh = mm.metrics.create()
    summary = mh.compute(acc, metrics=mm.metrics.motchallenge_metrics, name='summary')
    print(mm.io.render_summary(
        summary, formatters=mh.formatters, namemap=mm.io.motchallenge_metric_names
    ))

# Paths to ground truth and tracker result CSV files
gt_path = 'ground_truth.csv'  # Ground truth CSV
tracker_path = 'wildtrack.csv'  # Tracker's result CSV
gt_path = 'ground_truth_vlc-record-2025-01-03-14h37m50s-DJI_20240624153820_0001_V.csv'  # Ground truth CSV
tracker_path = 'output.csv'  # Tracker's result CSV

gt_path = 'ground_truth.csv'  # Ground truth CSV
tracker_path = 'wildtrack.csv'  # Tracker's result CSV
tracker_path = 'outputgood.csv'  # Tracker's result CSV

# Evaluate tracker
evaluate_mot(gt_path, tracker_path)
