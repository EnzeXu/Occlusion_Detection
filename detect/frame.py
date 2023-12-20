import cv2
import os
import shutil
import numpy as np
import json
import pickle
import datetime
import matplotlib.pyplot as plt
from PIL import Image, ImageDraw, ImageFont
from collections import defaultdict
from tqdm import tqdm

INPUT_FOLDER = "data/raw/"
OUTPUT_FOLDER = "data/processed/"
MASK_BLACK_FILE = "data/black/black.png"
SIZE_720 = (1280, 720)
SIZE_360 = (640, 360)


def get_now_string(time_string="%Y%m%d_%H%M%S_%f"):
    return datetime.datetime.now().strftime(time_string)


def extract_frames(video_path, frame_speed=5, video_speed=30, output_folder=None, period=None):
    if period:
        start, end = period
        start_index = int(start * video_speed)
        end_index = int(end * video_speed)
    else:
        start_index = -1e6
        end_index = 1e6
    if output_folder is None:
        output_folder = video_path.replace("input", "output").replace(".MP4", "").replace(".mov", "")
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print("Error: Could not open video file.")
        return
    frame_count = 0
    frame_filename_list = []
    while True:
        ret, frame = cap.read()

        if not ret:
            break
        if frame_count % frame_speed == 0 and start_index <= frame_count < end_index:
            # print(f"frame_count: {frame_count}")
            frame_filename = os.path.join(output_folder, f"{os.path.splitext(os.path.basename(video_path))[0]}_{frame_count:06d}.jpg")
            frame_filename_list.append(frame_filename)
            cv2.imwrite(frame_filename, frame)
        frame_count += 1

    cap.release()
    cv2.destroyAllWindows()
    return frame_filename_list


def copy_png_file_multiple_times(source_file, output_folder, name_list):
    os.makedirs(output_folder, exist_ok=True)
    for new_name in name_list:
        destination_file = os.path.join(output_folder, new_name)
        shutil.copyfile(source_file, destination_file)


def batch_compress_images(input_folder, output_folder):

    os.makedirs(output_folder, exist_ok=True)
    for filename in os.listdir(input_folder):
        if filename.endswith('.jpg'):
            with Image.open(os.path.join(input_folder, filename)) as img:
                img = img.resize(SIZE_360, Image.BILINEAR)
                img.save(os.path.join(output_folder, filename), 'JPEG')


def one_time_process_video(video_name, frame_speed=5, video_speed=30, period=None, timestamp=None):
    if not timestamp:
        timestamp = get_now_string()
    job_name = f"{timestamp}_{video_name.replace('.MP4', '').replace('.mov', '')}"
    print(f"job name: {job_name} time_stamp: {timestamp}")
    input_path = os.path.join(INPUT_FOLDER, video_name)
    output_folder = os.path.join(OUTPUT_FOLDER, job_name)
    output_folder_cut = os.path.join(output_folder, "cut")
    output_folder_cut_raw = os.path.join(output_folder, "cut_raw")
    output_folder_mask = os.path.join(output_folder, "mask")
    output_folder_result = os.path.join(output_folder, "results")
    output_folder_dict = os.path.join(output_folder, "dict")
    output_folder_plot = os.path.join(output_folder, "plot")
    output_folder_edge = os.path.join(output_folder, "edge")
    output_folder_video = os.path.join(output_folder, "video")
    output_folder_video_all = os.path.join(OUTPUT_FOLDER, "all_videos")

    print(f"[Step 1] Cutting from {input_path} to {output_folder_cut_raw}...")
    if not os.path.exists(output_folder_cut_raw):
        os.makedirs(output_folder_cut_raw)
        cut_filename_list = extract_frames(input_path, frame_speed, video_speed, output_folder_cut_raw, period=period)
        cut_filename_list = [os.path.basename(item) for item in cut_filename_list]
        print("Done")
    else:
        cut_filename_list = os.listdir(output_folder_cut_raw)
        cut_filename_list = [os.path.basename(item) for item in cut_filename_list if ".jpg" in item]
        print("Ready")
    mask_filename_list = [os.path.basename(item).replace("jpg", "png") for item in cut_filename_list]
    # print(f"frame_filename_list [length={len(cut_filename_list)}]: {cut_filename_list}")

    print(f"[Step 2] Compressing from {output_folder_cut_raw} to {output_folder_cut}...")
    if not os.path.exists(output_folder_cut):
        os.makedirs(output_folder_cut)
        batch_compress_images(output_folder_cut_raw, output_folder_cut)
        print("Done")
    else:
        print("Ready")

    print(f"[Step 3] Creating masks (n={len(mask_filename_list)}) to {output_folder_mask}...")

    if not os.path.exists(output_folder_mask):
        os.makedirs(output_folder_mask)
        copy_png_file_multiple_times(MASK_BLACK_FILE, output_folder_mask, mask_filename_list)
        print("Done")
    else:
        print("Ready")

    meta_file_path = os.path.join(output_folder, "meta.json")
    print(f"[Step 4] Making meta file to {meta_file_path}...")
    make_meta(folder_path=output_folder_cut, output_file=meta_file_path)

    print(f"[Step 5] Detecting...")
    if not os.path.exists(output_folder_result):
        cmd = f"python cfbi/eval_net.py --config configs.resnet101_cfbi --dataset od --job {job_name} --ckpt_path ./pretrain_models/resnet101_cfbi.pth --gpu_id -1"
        os.system(cmd)
        print("Done")
    else:
        print("Ready")

    print(f"[Step 6] Save IOU dict from {output_folder_result} to {os.path.join(output_folder_dict, 'dict_list.pt')}...")
    detected_file_names = [os.path.join(output_folder_result, item.replace(".jpg", ".npy")) for item in cut_filename_list]
    detected_file_names = sorted(detected_file_names)
    if not os.path.exists(output_folder_dict):
        os.makedirs(output_folder_dict)
        # print(detected_file_names)
        dic_list = [dict()] + [matrix_to_dic(np.load(one_path)) for one_path in tqdm(detected_file_names[1:])]
        with open(os.path.join(output_folder_dict, "dict_list.pkl"), "wb") as f:
            pickle.dump(dic_list, f)
        print("Done")
    else:
        with open(os.path.join(output_folder_dict, "dict_list.pkl"), "rb") as f:
            dic_list = pickle.load(f)
        print("Ready")
    iou_list = [0.0] + [calculate_iou(dic_list[i - 1], dic_list[i]) for i in tqdm(range(1, len(dic_list)))]
    # for one_name, one_iou in zip(detected_file_names, iou_list):
    #     print(one_name, one_iou)

    print(f"[Step 7] Detect edge nodes...")
    if not os.path.exists(output_folder_edge):
        os.makedirs(output_folder_edge)
        edge_node_list_all = [[]] + [edge_nodes(np.load(one_path)) for one_path in tqdm(detected_file_names[1:])]
        with open(os.path.join(output_folder_edge, "edge.pkl"), "wb") as f:
            pickle.dump(edge_node_list_all, f)
        print("Done")
    else:
        with open(os.path.join(output_folder_edge, "edge.pkl"), "rb") as f:
            edge_node_list_all = pickle.load(f)
        print("Ready")

    print(f"[Step 8] Plotting from {output_folder_cut} to {output_folder_plot}...")
    iou_threshold = 0.75
    occlusion_time_threshold = 2.0#5.0
    t_unit = 1.0 / video_speed * frame_speed
    plot_file_names = [os.path.join(output_folder_plot, item) for item in cut_filename_list]
    plot_file_names = sorted(plot_file_names)
    if not os.path.exists(output_folder_plot):
        os.makedirs(output_folder_plot)
    occlusion_flag = False
    save_detected_flag = False  # for save only
    save_occlusion_flag = False  # for save only
    time_accumulated = 0.0
    time_accumulated_list = []
    for i in tqdm(range(len(iou_list))):
        if iou_list[i] > iou_threshold:
            save_detected_flag = True
            time_accumulated += t_unit
            if time_accumulated >= occlusion_time_threshold:
                occlusion_flag = True
                save_occlusion_flag = True
        else:
            time_accumulated = 0.0
        time_accumulated_list.append(time_accumulated)

        plot_list = [
            ("blue", f"   Max IoU: {iou_list[i]:.6f}  "),
            ("blue", f"Accum Time: {time_accumulated:.6f}s "),
            ("green", f"    Status: Safe      ") if not occlusion_flag else (("red", f"    Status: Occlusion!") if i % 1 == 0 else ("red", f"    Status:           ")) # 2
        ]
        plot_text(plot_file_names[i].replace("/plot/", "/cut/"), plot_file_names[i], plot_list, edge_node_list_all[i])
    print("Done")
    # else:
    #     print("Ready")

    print(f"[Step 9] Creating video from {output_folder_plot} to {output_folder_video}...")
    video_filename = f"{job_name}_{'Detected' if save_detected_flag else 'Non-detected'}_{'Occlusion' if save_occlusion_flag else 'Non-occlusion'}.mp4"
    if not os.path.exists(output_folder_video):
        os.makedirs(output_folder_video)
        creat_video_from_images(plot_file_names, os.path.join(output_folder_video, video_filename), SIZE_360, int(video_speed / frame_speed))
        plot_max_iou(iou_list, [i * t_unit for i in range(len(cut_filename_list))], os.path.join(output_folder_video, f"{job_name}_iou.png"))
        print("Done")
    else:
        print("Ready")

    print(f"[Step 10] Copy video from {output_folder_video} to {output_folder_video_all}...")
    if not os.path.exists(output_folder_video_all):
        os.makedirs(output_folder_video_all)
    if not os.path.exists(os.path.join(output_folder_video_all, video_filename)):
        shutil.copy(os.path.join(output_folder_video, video_filename), os.path.join(output_folder_video_all, video_filename))
        print("Done")
    else:
        print("Ready")


def plot_max_iou(iou_list, t, save_path):
    plt.figure(figsize=(8, 6))
    plt.plot(t, iou_list, label="Max IoU")
    # plt.plot(t, ta_list, label="Accum Time")
    plt.xlabel("$t$", fontsize=20)
    plt.ylabel("$IoU$", fontsize=20)
    plt.xticks(fontsize=20)
    plt.yticks(fontsize=20)
    plt.grid(True)
    plt.tight_layout()
    plt.legend()
    plt.savefig(save_path, dpi=400)
    plt.close()


def creat_video_from_images(full_file_names, output_path, fig_size, fps=5):
    # file_dir = input_dir
    # file_names = ["Turing2D_Fourier_Tau_20230408_173656_366942_epoch={}_train.png".format(i * 10) for i in range(1, 40)]

    # print("{} files: {}".format(len(file_names), file_names))
    video = cv2.VideoWriter(output_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, fig_size)
    # video = cv2.VideoWriter(output_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, fig_size) # mp4
    # video = cv2.VideoWriter(output_path, cv2.VideoWriter_fourcc(*'MJPG'), fps, fig_size) # avi
    # video = cv2.VideoWriter('outputs/test.mp4', fourcc, 5, (16000, 4800))
    for i in tqdm(range(len(full_file_names))):
        img = cv2.imread(full_file_names[i])
        img = cv2.resize(img, fig_size)
        video.write(img)
    video.release()



def plot_text(input_filepath, output_filepath, text_list, node_location_list):
    with Image.open(input_filepath) as img:
        draw = ImageDraw.Draw(img)
        try:
            font = ImageFont.truetype("CourierNew.ttf", 30)
            # print(11111)
        except IOError:
            font = ImageFont.load_default()

        text_x = img.width - 30
        text_y = 10
        for node in node_location_list:
            node = (int(node[0]), int(node[1]))
            draw.point(node, fill="red")

        for color, text in text_list:
            bbox = draw.textbbox((text_x, text_y), text, font=font)
            text_height = bbox[3] - bbox[1]

            draw.text((text_x - bbox[2] + bbox[0], text_y), text, font=font, fill=color)
            text_y += 10 + text_height

            # Save the image
        img.save(output_filepath)


def edge_nodes(map):
    rows, cols = map.shape
    edge_node_list = []
    for i in range(rows):
        for j in range(cols):
            current_class_id = map[i, j]
            neighbors = []
            if i > 0:  # Up
                neighbors.append(map[i-1, j])
            if i < rows - 1:  # Down
                neighbors.append(map[i+1, j])
            if j > 0:  # Left
                neighbors.append(map[i, j-1])
            if j < cols - 1:  # Right
                neighbors.append(map[i, j+1])

            # Check if any neighbor's class_id is different
            if any(neighbor_class_id != current_class_id for neighbor_class_id in neighbors):
                edge_node_list.append((j, i))

    return edge_node_list


def plot_node_list(input_filepath, output_filepath, node_location_list):
    # Open the image file
    with Image.open(input_filepath) as img:
        # Create a drawing object
        draw = ImageDraw.Draw(img)

        # Set the color for the nodes
        node_color = 'blue'

        # Plot each node location
        for node in node_location_list:
            # Ensure the node coordinates are integers
            node = (int(node[0]), int(node[1]))
            # Draw a single pixel for each node
            draw.point(node, fill=node_color)

        # Save the image
        img.save(output_filepath)


def matrix_to_dic(data):
    # dic = dict()
    # for i in range(data.shape[0]):
    #     for j in range(data.shape[1]):
    #         if int(data[i][j]) not in dic:
    #             dic[int(data[i][j])] = [(i, j)]
    #         else:
    #             dic[int(data[i][j])] += [(i, j)]
    # return dic
    dic = defaultdict(list)
    for value in np.unique(data):
        indices = np.argwhere(data == value)
        dic[int(value)].extend(map(tuple, indices))
    return dict(dic)


def calculate_iou(dic1, dic2):
    #  2 npy files
    # data1 = np.load(filepath1)
    # data2 = np.load(filepath2)
    # dic1 = matrix_to_dic(data1)
    # dic2 = matrix_to_dic(data2)
    res_dic = dict()
    all_keys = list(set(list(dic1.keys()) + list(dic2.keys())))
    for one_key in all_keys:
        if one_key == 0:
            continue
        if one_key not in dic1 or one_key not in dic2:
            res_dic[one_key] = 0.0
            continue
        node_list_1, node_list_2 = dic1[one_key], dic2[one_key]
        n_union = len(list(set(node_list_1 + node_list_2)))
        n_overlap = len(node_list_1) + len(node_list_2) - n_union
        res_dic[one_key] = n_overlap / n_union
    if len(res_dic.keys()) == 0:
        max_iou = 0.0
    else:
        max_iou = max([res_dic.get(item) for item in res_dic])
    # print(res_dic)
    return max_iou


def make_meta(folder_path, output_file):
    name_list = sorted(list(os.listdir(folder_path)))
    name_list = [item.replace(".jpg", "") for item in name_list]
    dic = {
        "videos": {
            folder_path.split("/")[-1]: {
                "objects": {
                    "1": {
                        "category": None,
                        "frames": name_list
                    },
                    "2": {
                        "category": None,
                        "frames": name_list
                    },
                    "3": {
                        "category": None,
                        "frames": name_list
                    }
                }
            }
        }
    }
    with open(output_file, "w") as f:
        json.dump(dic, f, indent=4)




def one_time_job():
    job_list = [
        # ["NO20231218-170418-001549F.MP4", 0],
        # ["NO20231218-170418-001549F.MP4", 20],
        # ["NO20231218-170418-001549F.MP4", 40],
        #
        # ["NO20231218-170518-001550F.MP4", 0],
        # ["NO20231218-170518-001550F.MP4", 20],
        # ["NO20231218-170518-001550F.MP4", 40],
        #
        # ["NO20231218-170618-001551F.MP4", 0],
        # ["NO20231218-170618-001551F.MP4", 20],
        # ["NO20231218-170618-001551F.MP4", 40],
        #
        # ["NO20231218-170718-001552F.MP4", 0],
        # ["NO20231218-170718-001552F.MP4", 20],
        # ["NO20231218-170718-001552F.MP4", 40],
        #
        # ["NO20231218-170818-001553F.MP4", 0],
        # ["NO20231218-170818-001553F.MP4", 20],
        ["NO20231218-170818-001553F.MP4", 40],

        ["NO20231218-170918-001554F.MP4", 0],
        ["NO20231218-170918-001554F.MP4", 20],
        ["NO20231218-170918-001554F.MP4", 40],

        ["NO20231218-171018-001555F.MP4", 0],
        ["NO20231218-171018-001555F.MP4", 20],
        ["NO20231218-171018-001555F.MP4", 40],

        ["NO20231218-171118-001556F.MP4", 0],
        ["NO20231218-171118-001556F.MP4", 20],
        ["NO20231218-171118-001556F.MP4", 40],

        ["NO20231218-171218-001557F.MP4", 0],
        ["NO20231218-171218-001557F.MP4", 20],
        ["NO20231218-171218-001557F.MP4", 40],

        ["NO20231218-171318-001558F.MP4", 0],
        ["NO20231218-171318-001558F.MP4", 20],
        ["NO20231218-171318-001558F.MP4", 40],

        ["NO20231218-171418-001559F.MP4", 0],
        ["NO20231218-171418-001559F.MP4", 20],
        ["NO20231218-171418-001559F.MP4", 40],

        ["NO20231218-171518-001560F.MP4", 0],
        ["NO20231218-171518-001560F.MP4", 20],
        ["NO20231218-171518-001560F.MP4", 40],

        ["NO20231218-171618-001561F.MP4", 0],
        ["NO20231218-171618-001561F.MP4", 20],
        ["NO20231218-171618-001561F.MP4", 40],

        ["NO20231218-171718-001562F.MP4", 0],
        ["NO20231218-171718-001562F.MP4", 20],
        ["NO20231218-171718-001562F.MP4", 40],

        ["NO20231218-171818-001563F.MP4", 0],
        ["NO20231218-171818-001563F.MP4", 20],
        ["NO20231218-171818-001563F.MP4", 40],

        ["NO20231218-171918-001564F.MP4", 0],
    ]
    length = 20
    for i, one_pair in tqdm(enumerate(job_list)):
        period = [one_pair[1], one_pair[1] + length]
        timestamp = get_now_string()
        print(f"\n\n######################################################################### [{i+1:04d} / {len(job_list):04d}] Processing {one_pair[0]}: period = {period}, timestamp = {timestamp} #########################################################################")
        one_time_process_video(one_pair[0], frame_speed=5, video_speed=30, period=period, timestamp=timestamp)


if __name__ == "__main__":
    # one_time_process_video("NO20231118-170905-000569F.MP4", frame_speed=1, period=[5, 7])
    # one_time_process_video("C0018.MP4", frame_speed=3, video_speed=60, period=[8.5, 10.5])
    # one_time_process_video("NO20231118-163917-000538F.MP4", frame_speed=5, video_speed=30, period=[20, 26])
    # one_time_process_video("NO20231118-164417-000543F.MP4", frame_speed=5, video_speed=30, period=[40, 60])
    one_time_process_video("NO20231118-164417-000543F.MP4", frame_speed=5, video_speed=30, period=[42, 48])
    # one_time_process_video("NO20231118-164217-000541F.MP4", frame_speed=5, video_speed=30, period=[9, 29])
    # one_time_process_video("NO20231118-165217-000551F.MP4", frame_speed=5, video_speed=30, period=[40, 60])
    # one_time_process_video("NO20231118-172246-000584F.MP4", frame_speed=5, video_speed=30, period=[40, 60])
    # one_time_process_video("NO20231118-170705-000567F.MP4", frame_speed=5, video_speed=30, period=[40, 60])
    # one_time_process_video("NO20231218-170418-001549F.MP4", frame_speed=5, video_speed=30, period=[7, 27])
    # one_time_job()
    # one_time_process_video("d466f0e0-c9b6897c.mov", frame_speed=5, video_speed=30, period=[0, 20])
    # map = np.array([[1, 1, 2, 2],
    #                 [1, 1, 2, 2],
    #                 [3, 3, 4, 4],
    #                 [3, 3, 4, 4]])
    # print(edge_nodes(map))
    # plot_text("test/test_1.jpg", "test/test_1_text.jpg", ["hello", "world"])
    # file_names = ["data/processed/NO20231118-164417-000543F/cut/NO20231118-164417-000543F_00{}.jpg".format(1200 + i * 5) for i in range(0, 30)]
    # creat_video_from_images(file_names, "test/cover.mp4", SIZE_360, 6)
    pass