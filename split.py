import os
import random
import csv


def split_data(data_dir, train_percent=0.8):
    train_list = []
    valid_list = []

    dirs = os.listdir(data_dir)
    os.chdir(data_dir)
    class_txt = open('class.txt', 'w')
    i = 0
    for dir in dirs:
        if dir in ['train.txt', 'valid.txt', 'class.txt', 'train.csv', 'valid.csv']:
            continue
        temp_list = []
        files = os.listdir(dir)
        for file in files:
            if file[0] == '.':
                continue
            temp_list.append(dir + '/' + file + '!' + str(i))
        random.shuffle(temp_list)
        threshold = int(len(temp_list) * train_percent)
        train_list += temp_list[0:threshold]
        valid_list += temp_list[threshold::]

        class_txt.write(dir + ' ' + str(i) + '\n')
        i += 1
    random.shuffle(train_list)
    random.shuffle(valid_list)
    class_txt.close()

    with open('./train.txt', 'w') as f:
        for path in train_list:
            f.write(path.strip().split('!')[0] + ' ' + path.strip().split('!')[1] + '\n')
    with open('./valid.txt', 'w') as f:
        for path in valid_list:
            f.write(path.strip().split('!')[0] + ' ' + path.strip().split('!')[1] + '\n')

    with open('./train.csv', 'w') as f:
        f_csv = csv.writer(f)
        f_csv.writerow(['img_name', 'label'])
        for path in train_list:
            f_csv.writerow(path.strip().split('!'))
    with open('./valid.csv', 'w') as f:
        f_csv = csv.writer(f)
        f_csv.writerow(['img_name', 'label'])
        for path in valid_list:
            f_csv.writerow(path.strip().split('!'))

    print('Finished writing ')


def read_data(dir):
    x_train, y_train, x_valid, y_valid = [], [], [], []
    with open(dir + 'train.txt', 'r') as f:
        for line in f.readlines():
            x_train.append(line.strip().split(' ')[0])
            y_train.append(line.strip().split(' ')[1])
    with open(dir + 'valid.txt', 'r') as f:
        for line in f.readlines():
            x_valid.append(line.strip().split(' ')[0])
            y_valid.append(line.strip().split(' ')[1])
    return x_train, y_train, x_valid, y_valid


if __name__ == '__main__':
    Gb_data_dir = "/home/hsq/DeepLearning/volume/gstreamer/process/img/"
    if os.path.exists(os.path.join(Gb_data_dir, 'class.txt')) or os.path.exists(
            os.path.join(Gb_data_dir, 'train.txt')) or os.path.exists(
        os.path.join(Gb_data_dir, 'valid.txt')) or os.path.exists(os.path.join(Gb_data_dir,
                                                                               'train.csv')) or os.path.exists(
        os.path.join(Gb_data_dir, 'valid.csv')):
        print('路径:', str(Gb_data_dir), '下class.txt train.txt valid.txt文件存在，请手动删除后再运行该程序')
        exit()

    split_data(Gb_data_dir, train_percent=0.8)
    # x_train, y_train, x_valid, y_valid = read_data(Gb_data_dir)
    exit()
