import copy
import json

import random
import os


def process_splice(file_intput_root, file_name, file_output_root):
    df = open(os.path.join(file_intput_root, file_name), encoding="utf-8")  # 打开原始的麻将文件数据
    data = json.load(df)  # 使用json读取原始的麻将文件数据
    discard_step = data["discard_step"]
    tmp_step = []
    for item in discard_step:
        tmp_step.append(item)
        round = item["round"]
        tmp_data = copy.deepcopy(data)
        tmp_data["discard_step"] = tmp_step

        file_label_root = os.path.join(file_output_root, str(round))

        if not os.path.exists(file_label_root):  # 标签文件目录不存在
            os.mkdir(file_label_root)  # 创建标签文件目录

        file_num = file_num_dict.get(str(round))  # 获取存好的文件数量

        if file_num is None:  # 若没有就获取一下文件数量
            file_num = len(os.listdir(file_label_root))
            file_num_dict[str(round)] = file_num

        if file_num > 20000:
            return file_num_dict

        with open(os.path.join(file_label_root, f"{file_num}.json"), "w", encoding="utf-8") as file_json:  # 将数据写入文件中
            file_json.write(json.dumps(tmp_data, indent=4))

        file_num_dict[str(round)] = file_num_dict[str(round)] + 1  # 文件数量加一

    return file_num_dict

if __name__ == '__main__':
    # # print(get_action_label()[1])
    file_root = "/home/tonnn/xiu/data/discard_total"
    dir_name = os.listdir(file_root)
    file_num_dict = {}  # 控制文件数量
    for item in dir_name:
        file_label_path = os.path.join(file_root, item)
        file_name = os.listdir(file_label_path)
        for it in file_name:
            file_num_dict = process_splice(file_label_path, it, r"/home/tonnn/xiu/data/discard_total_splice")
        print(f"{item}文件完成！")

