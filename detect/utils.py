from PIL import Image, ImageOps


def merge_pictures(input_path_list, row, col, save_path, border_width=5):
    if len(input_path_list) != row * col:
        raise ValueError("The number of input images does not match the specified grid size.")

    images = [Image.open(path) for path in input_path_list]
    image_width, image_height = images[0].size
    merged_width = (image_width + border_width) * col - border_width
    merged_height = (image_height + border_width) * row - border_width

    merged_image = Image.new("RGB", (merged_width, merged_height), color="white")

    for i in range(row):
        for j in range(col):
            x_offset = j * (image_width + border_width)
            y_offset = i * (image_height + border_width)
            merged_image.paste(images[i * col + j], (x_offset, y_offset))

    merged_image.save(save_path)


if __name__ == "__main__":
    input_path_list = [f"data/processed/20231219_220829_359206_NO20231118-164417-000543F/plot/NO20231118-164417-000543F_00{1310 + 5 * i}.jpg" for i in range(24)]
    merge_pictures(input_path_list, 4, 6, "test/test_plot.png")
    input_path_list = [f"data/processed/20231219_220829_359206_NO20231118-164417-000543F/results/NO20231118-164417-000543F_00{1310 + 5 * i}.png" for i in range(24)]
    merge_pictures(input_path_list, 4, 6, "test/test_results.png")
    # input_path_list = [f"data/processed/20231219_220829_359206_NO20231118-164417-000543F/cut_raw/NO20231118-164417-000543F_00{1310 + 5 * i}.jpg" for i in range(24)]
    # merge_pictures(input_path_list, 2, 12, "test/test_raw.png")
    input_path_list = [
        f"data/processed/20231219_220829_359206_NO20231118-164417-000543F/cut/NO20231118-164417-000543F_00{1310 + 5 * i}.jpg"
        for i in range(24)]
    merge_pictures(input_path_list, 4, 6, "test/test_cut.png")