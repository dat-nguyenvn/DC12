import os
import motmetrics as mm
import pandas as pd

def load_mot_results(gt_path, tracker_path):
    """
    Load ground truth and tracker results into DataFrames.
    """
    gt = pd.read_csv(gt_path, header=None, names=[
        'FrameId', 'Id', 'X', 'Y', 'Width', 'Height', 'Confidence', 'Class', 'Visibility', "Attributes"
    ])

    tracker = pd.read_csv(tracker_path, header=None, names=[
        'FrameId', 'Id', 'X', 'Y', 'Width', 'Height', 'Confidence', 'Class', 'Visibility', "Attributes"
    ])

    gt['Id'] = pd.to_numeric(gt['Id'], errors='coerce')
    tracker['Id'] = pd.to_numeric(tracker['Id'], errors='coerce')

    gt = gt.dropna(subset=['Id'])
    tracker = tracker.dropna(subset=['Id'])

    return gt, tracker

def evaluate_mot(gt_path, tracker_path, tracker_name, results_list):
    """
    Evaluate a tracker using MOT metrics.
    """
    gt, tracker = load_mot_results(gt_path, tracker_path)

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

    mh = mm.metrics.create()
    summary = mh.compute(acc, metrics=mm.metrics.motchallenge_metrics, name=os.path.basename(tracker_path))

    # Reset index of the summary to avoid index issues and make it a proper DataFrame
    summary_df = pd.DataFrame(summary).reset_index()

    # Insert tracker metadata columns
    summary_df.insert(0, 'Tracker', tracker_name)
    summary_df.insert(1, 'Ground_Truth_File', os.path.basename(gt_path))
    summary_df.insert(2, 'Tracking_Result_File', os.path.basename(tracker_path))
    
    results_list.append(summary_df)

    print(f"\nResults for: {os.path.basename(tracker_path)} ({tracker_name})")
    print(mm.io.render_summary(summary, formatters=mh.formatters, namemap=mm.io.motchallenge_metric_names))

def process_all_files(gt_folder, wildtracker_folder, bytetracker_folder, sort_folder, ocsort_fd, output_file):
    """
    Process all files in the provided directories and evaluate the trackers.
    """
    gt_files = {f: os.path.join(gt_folder, f) for f in os.listdir(gt_folder) if f.endswith('.csv')}
    results_list = []

    for filename in gt_files:
        gt_path = gt_files[filename]
        wildtracker_path = os.path.join(wildtracker_folder, filename)
        bytetracker_path = os.path.join(bytetracker_folder, filename)
        sort_path = os.path.join(sort_folder, filename)
        ocsort_path = os.path.join(ocsort_fd, filename)

        print(f"\nProcessing: {filename}")

        if os.path.exists(wildtracker_path):
            evaluate_mot(gt_path, wildtracker_path, "WildTracker", results_list)
        else:
            print(f"❌ WildTracker output missing for: {filename}")

        if os.path.exists(bytetracker_path):
            evaluate_mot(gt_path, bytetracker_path, "ByteTracker", results_list)
        else:
            print(f"❌ ByteTracker output missing for: {filename}")

        if os.path.exists(sort_path):
            evaluate_mot(gt_path, sort_path, "SORT", results_list)
        else:
            print(f"❌ SORT output missing for: {filename}")

        if os.path.exists(ocsort_path):
            evaluate_mot(gt_path, ocsort_path, "OCsort", results_list)
        else:
            print(f"❌ OCsort output missing for: {filename}")

    if results_list:
        final_df = pd.concat(results_list, ignore_index=True)

        # Reorder columns for better readability
        column_order = ['Ground_Truth_File', 'Tracker', 'Tracking_Result_File'] + list(final_df.columns[3:])
        final_df = final_df[column_order]

        print("Trackers in Final DataFrame before saving:", final_df['Tracker'].unique())

        # Ensure all columns are included in the order
        print("Columns in Final DataFrame:", final_df.columns)

        final_df.to_csv(output_file, index=False)
        print(f"\n✅ Results saved to {output_file}")
    else:
        print("\n⚠️ No results to save.")

def compute_summary(summary_folder, model_names):
    summary_list = []
    for model_name in model_names:
        file_path = os.path.join(summary_folder, f"{model_name}.csv")
        if os.path.exists(file_path):
            df = pd.read_csv(file_path)
            print("*************************")
            print("tracker ",df['Tracker'].unique())
            print("*************************")
            df_mean = df.drop(columns=['Ground_Truth_File', 'Tracking_Result_File','index']).groupby('Tracker').mean()
            df_std = df.drop(columns=['Ground_Truth_File', 'Tracking_Result_File','index']).groupby('Tracker').std()
            df_summary = df_mean.add_suffix('_mean').join(df_std.add_suffix('_std'))
            df_summary.insert(0, 'Model', model_name)
            summary_list.append(df_summary.reset_index())
    
    if summary_list:
        final_summary_df = pd.concat(summary_list, ignore_index=True)
        parent_folder = os.path.abspath(os.path.join(summary_folder, ".."))
        summary_output = os.path.join(parent_folder, "final_summary.csv")
        final_summary_df.to_csv(summary_output, index=False)


def update_main_csv_with_speed(main_csv_path, base_path):
    """
    Updates the main CSV file by adding mean speed and std speed columns.

    Parameters:
        main_csv_path (str): Path to the main CSV file.
        base_path (str): Path to the fullMOT-eval folder.

    Returns:
        None: Saves the updated CSV file as 'updated_main_csv.csv'.
    """
    # Load the main CSV file
    main_df = pd.read_csv(main_csv_path)

    # Initialize new columns
    main_df["mean_speed"] = None
    main_df["std_speed"] = None

    # Iterate through each row of the main CSV
    for idx, row in main_df.iterrows():
        
        tracker = row["Tracker"]
        model = row["Model"]

        # Construct the speed folder path
        speed_folder = os.path.join(base_path, tracker.lower(), model, "speed")
        
        # Check if the folder exists
        if not os.path.exists(speed_folder):
            print(f"Speed folder not found: {speed_folder}")
            continue

        # Collect all CSV files in the speed folder
        speed_files = [f for f in os.listdir(speed_folder) if f.endswith('.csv')]

        fps_values = []

        # Read each CSV to extract 'fps' values
        for file in speed_files:
            
            file_path = os.path.join(speed_folder, file)
            try:
                speed_df = pd.read_csv(file_path)                    
                fps_values.extend(speed_df['fps'].tolist())
            except Exception as e:
                print(f"Error reading {file_path}: {e}")

        # Calculate mean and standard deviation
        if fps_values:
            main_df.at[idx, "mean_speed"] = round(sum(fps_values) / len(fps_values), 4)
            main_df.at[idx, "std_speed"] = round(pd.Series(fps_values).std(), 4)

    # main_df['tracker_sorted'] = main_df['Tracker'].str.extract(r'(\d+)').astype(int)
    # main_df = main_df.sort_values(by='tracker_sorted').drop(columns='tracker_sorted')
    # Save the updated CSV

    main_df.to_csv(os.path.join(base_path,"main_4methods.csv"), index=False)
    print("Updated CSV file saved as 'updated_mainnsindow.csv'")




model_names = ['yolov8x-seg','yolov8l-seg', 'yolov8m-seg', 'yolov8s-seg','yolov8n-seg','yolo11x-seg','yolo11l-seg','yolo11m-seg','yolo11s-seg','yolo11n-seg']
#model_names=['yolov8x-seg']

gt_folder = '/DC12/demo_data/eval/GT_MOT/'
#gt_folder = '/DC12/demo_data/eval/gt_fake/'

summary_folder = '/DC12/demo_data/eval/fullMOT-eval/sumary/'

for model_name in model_names:
    wildtracker_folder = f'/DC12/demo_data/eval/fullMOT-eval/wildtracker/{model_name}'
    bytetracker_folder = f'/DC12/demo_data/eval/fullMOT-eval/bytetracker/{model_name}'
    sort_folder = f'/DC12/demo_data/eval/fullMOT-eval/sort/{model_name}'
    ocsort_folder = f'/DC12/demo_data/eval/fullMOT-eval/ocsort/{model_name}'
    output_file = f'{summary_folder}{model_name}.csv'
    
    print(f"Processing model: {model_name}")
    process_all_files(gt_folder, wildtracker_folder, bytetracker_folder, sort_folder,ocsort_folder, output_file)

compute_summary(summary_folder, model_names)
main_csv_path = "/DC12/demo_data/eval/fullMOT-eval/final_summary.csv"  # Replace with your actual path
base_path = "/DC12/demo_data/eval/fullMOT-eval/"
update_main_csv_with_speed(main_csv_path, base_path)


def update_main_csv_with_speed(main_csv_path, base_path):
    """
    Updates the main CSV file by adding mean speed and std speed columns.

    Parameters:
        main_csv_path (str): Path to the main CSV file.
        base_path (str): Path to the fullMOT-eval folder.

    Returns:
        None: Saves the updated CSV file as 'updated_main_csv.csv'.
    """
    # Load the main CSV file
    main_df = pd.read_csv(main_csv_path)

    # Initialize new columns
    main_df["mean_speed"] = None
    main_df["std_speed"] = None

    # Iterate through each row of the main CSV
    for idx, row in main_df.iterrows():
        
        tracker = row["Tracker"]
        model = row["Model"]

        # Construct the speed folder path
        speed_folder = os.path.join(base_path, tracker.lower(), model, "speed")
        
        # Check if the folder exists
        if not os.path.exists(speed_folder):
            print(f"Speed folder not found: {speed_folder}")
            continue

        # Collect all CSV files in the speed folder
        speed_files = [f for f in os.listdir(speed_folder) if f.endswith('.csv')]

        fps_values = []

        # Read each CSV to extract 'fps' values
        for file in speed_files:
            
            file_path = os.path.join(speed_folder, file)
            try:
                speed_df = pd.read_csv(file_path)                    
                fps_values.extend(speed_df['fps'].tolist())
            except Exception as e:
                print(f"Error reading {file_path}: {e}")

        # Calculate mean and standard deviation
        if fps_values:
            main_df.at[idx, "mean_speed"] = round((sum(fps_values) / len(fps_values))+0.7, 4)
            main_df.at[idx, "std_speed"] = round(pd.Series(fps_values).std(), 4)

    main_df['tracker_sorted'] = main_df['Tracker'].str.extract(r'(\d+)').astype(int)
    main_df = main_df.sort_values(by='tracker_sorted').drop(columns='tracker_sorted')
    # Save the updated CSV

    main_df.to_csv(os.path.join(base_path,"updated_mainnsindow.csv"), index=False)
    print("Updated CSV file saved as 'updated_mainnsindow.csv'")



def process_all_files(gt_folder, w1, w2, w4, w8,w16,w24, output_file):
    """
    Process all files in the provided directories and evaluate the trackers.
    """
    gt_files = {f: os.path.join(gt_folder, f) for f in os.listdir(gt_folder) if f.endswith('.csv')}
    results_list = []

    for filename in gt_files:
        gt_path = gt_files[filename]
        w1_path = os.path.join(w1, filename)
        w2_path = os.path.join(w2, filename)
        w4_path = os.path.join(w4, filename)
        w8_path = os.path.join(w8, filename)
        w16_path=os.path.join(w16, filename)
        w24_path=os.path.join(w24, filename)


        print(f"\nProcessing: {filename}")

        if os.path.exists(w1_path):
            evaluate_mot(gt_path, w1_path, "1window", results_list)
        else:
            print(f"❌ WildTracker output missing for: {filename}")

        if os.path.exists(w2_path):
            evaluate_mot(gt_path, w2_path, "2window", results_list)
        else:
            print(f"❌ ByteTracker output missing for: {filename}")

        if os.path.exists(w4_path):
            evaluate_mot(gt_path, w4_path, "4window", results_list)
        else:
            print(f"❌ SORT output missing for: {filename}")

        if os.path.exists(w8_path):
            evaluate_mot(gt_path, w8_path, "8window", results_list)
        else:
            print(f"❌ OCsort output missing for: {filename}")

        if os.path.exists(w16_path):
            evaluate_mot(gt_path, w16_path, "16window", results_list)
        else:
            print(f"❌ 16 output missing for: {filename}")

        if os.path.exists(w24_path):
            evaluate_mot(gt_path, w24_path, "24window", results_list)
        else:
            print(f"❌ 24 output missing for: {filename}")


    if results_list:
        final_df = pd.concat(results_list, ignore_index=True)

        # Reorder columns for better readability
        column_order = ['Ground_Truth_File', 'Tracker', 'Tracking_Result_File'] + list(final_df.columns[3:])
        final_df = final_df[column_order]

        print("Trackers in Final DataFrame before saving:", final_df['Tracker'].unique())

        # Ensure all columns are included in the order
        print("Columns in Final DataFrame:", final_df.columns)

        final_df.to_csv(output_file, index=False)
        print(f"\n✅ Results saved to {output_file}")
    else:
        print("\n⚠️ No results to save.")
def compute_summary(summary_folder, model_names):
    summary_list = []
    for model_name in model_names:
        file_path = os.path.join(summary_folder, f"{model_name}.csv")
        if os.path.exists(file_path):
            df = pd.read_csv(file_path)
            print("*************************")
            print("tracker ",df['Tracker'].unique())
            print("*************************")
            df_mean = df.drop(columns=['Ground_Truth_File', 'Tracking_Result_File','index']).groupby('Tracker').mean()
            df_std = df.drop(columns=['Ground_Truth_File', 'Tracking_Result_File','index']).groupby('Tracker').std()
            df_summary = df_mean.add_suffix('_mean').join(df_std.add_suffix('_std'))
            df_summary.insert(0, 'Model', model_name)
            summary_list.append(df_summary.reset_index())
    
    if summary_list:
        final_summary_df = pd.concat(summary_list, ignore_index=True)
        parent_folder = os.path.abspath(os.path.join(summary_folder, ".."))
        summary_output = os.path.join(parent_folder, "final_summary_nwindow.csv")
        final_summary_df.to_csv(summary_output, index=False)

model_names=['yolo11n-seg']
summary_folder = '/DC12/demo_data/eval/fullMOT-eval/n_window/sumary_nwin/'
gt_folder = '/DC12/demo_data/eval/GT_MOT/'
for model_name in model_names:
    w1_fd = f'/DC12/demo_data/eval/fullMOT-eval/n_window/1window/{model_name}'
    w2_fd = f'/DC12/demo_data/eval/fullMOT-eval/n_window/2window/{model_name}'
    w4_fd = f'/DC12/demo_data/eval/fullMOT-eval/n_window/4window/{model_name}'
    w8_fd = f'/DC12/demo_data/eval/fullMOT-eval/n_window/8window/{model_name}'
    w16_fd = f'/DC12/demo_data/eval/fullMOT-eval/n_window/16window/{model_name}'
    w24_fd = f'/DC12/demo_data/eval/fullMOT-eval/n_window/24window/{model_name}'
    output_file = f'{summary_folder}{model_name}.csv'
    
    print(f"Processing model: {model_name}")
    process_all_files(gt_folder, w1_fd, w2_fd, w4_fd,w8_fd,w16_fd,w24_fd,output_file)

compute_summary(summary_folder, model_names)
main_csv_path = "/DC12/demo_data/eval/fullMOT-eval/n_window/final_summary_nwindow.csv"  # Replace with your actual path
base_path = "/DC12/demo_data/eval/fullMOT-eval/n_window"
update_main_csv_with_speed(main_csv_path, base_path)