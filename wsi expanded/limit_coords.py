tumor_train_patch_path = "../PATCHES_TUMOR_TRAIN/list.txt"
tumor_valid_patch_path = "../PATCHES_TUMOR_VALID/list.txt"
normal_train_patch_path = "../PATCHES_NORMAL_TRAIN/list.txt"
normal_valid_patch_path = "../PATCHES_NORMAL_VALID/list.txt"

tumor_train_list = ["Tumor_001"]
tumor_valid_list = ["Tumor_101"]
normal_train_list = ["Normal_001"]
normal_valid_list = ["Normal_141"]


def limit_coords_def(wsi_names, patch_path):
    f = open(patch_path, "r")
    to_write = ""
    for line in f:
        pid, x_center, y_center = line.strip('\n').split(',')[0:3]
        if pid in wsi_names:
            to_write += line
    f.close()
    f = open(patch_path, "w")
    f.write(to_write.strip())
    f.close()


limit_coords_def(tumor_train_list, tumor_train_patch_path)
limit_coords_def(tumor_valid_list, tumor_valid_patch_path)
limit_coords_def(normal_train_list, normal_train_patch_path)
limit_coords_def(normal_valid_list, normal_valid_patch_path)