import random
import glob



def split_data(dir_root):
    random.seed(0)
    ori_img = glob.glob(dir_root + '/*.png')
    k = 0.5
    train_ori_imglist = []
    val_ori_imglist = []
    sample_data = random.sample(population=ori_img, k=int(k * len(ori_img)))
    for img in ori_img:
        if img in sample_data:
            val_ori_imglist.append(img)
        train_ori_imglist.append(img)
    return train_ori_imglist, val_ori_imglist


def split_data_real(input, output, output_2):
    random.seed(0)
    ori_img = input
    out_img = output
    out_img_10 = output_2
    train_ori_imglist = []
    train_ori_imglist_out = []
    val_ori_imglist = []
    val_ori_imglist_out = []
    for index, img in enumerate(ori_img):
        if index % 6 == 0:
            val_ori_imglist.append(img)
            val_ori_imglist_out.append([out_img[index], out_img_10[index]])
        else:
            train_ori_imglist.append(img)
            train_ori_imglist_out.append([out_img[index], out_img_10[index]])
    return train_ori_imglist, train_ori_imglist_out, val_ori_imglist, val_ori_imglist_out


if __name__ == '__main__':
    a, b = split_data('../base')
