tumor_train_patch_path = "../PATCHES_TUMOR_TRAIN/list.txt"
tumor_valid_patch_path = "../PATCHES_TUMOR_VALID/list.txt"
normal_train_patch_path = "../PATCHES_NORMAL_TRAIN/list.txt"
normal_valid_patch_path = "../PATCHES_NORMAL_VALID/list.txt"

tumor_train_list_server = ["tumor_001", "tumor_002", "tumor_003", "tumor_004", "tumor_005", "tumor_006", "tumor_007", "tumor_008"]
tumor_valid_list_server = ["tumor_101", "tumor_102"]
normal_train_list_server = ["normal_001", "normal_002", "normal_003", "normal_004", "normal_005", "normal_006", "normal_007", "normal_008"]
normal_valid_list_server = ["normal_141", "normal_142"]
all_list_server = tumor_train_list_server + tumor_valid_list_server + normal_train_list_server + normal_valid_list_server

tumor_train_list = ["tumor_001"]
tumor_valid_list = ["tumor_101"]
normal_train_list = ["normal_001"]
normal_valid_list = ["normal_141"]
all_list = tumor_train_list + tumor_valid_list + normal_train_list + normal_valid_list


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


limit_coords_def(all_list, tumor_train_patch_path)
limit_coords_def(all_list, tumor_valid_patch_path)
limit_coords_def(all_list, normal_train_patch_path)
limit_coords_def(all_list, normal_valid_patch_path)
