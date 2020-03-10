import os
import pandas as pd
from sklearn.utils import shuffle


label_list = []
image_list = []

image_dir = '/root/private/ColorImage/'
label_dir = '/root/private/Gray_Label/'

"""
cd ColorImage_road2
mv ColoeImage /root/private/ColorImage/Road2
"""
"""
   ColorImage/
     road02/
       Record002/
         Camera 5/
           ...
         Camera 6
       Record003
       ....
     road03
     road04
   Label/
     Label_road02/
      Label
       Record002/
         Camera 5/
          ...
         Camera 6
       Record003
       ....
     Label_road03
     Label_road04     
     
"""
# ColorImage
for s1 in os.listdir(image_dir): #os.listdir()返回1个list，list里面是image_dir文件下面的文件名
    # image_dir/road02
    image_sub_dir1 = os.path.join(image_dir, s1) #按照路径连接的方式连接到一起
    # label_dir/label_road02/label
    label_sub_dir1 = os.path.join(label_dir, 'Label_' + str.lower(s1), 'Label')

    # road2
    for s2 in os.listdir(image_sub_dir1):
        # image_dir/road02/record001
        image_sub_dir2 = os.path.join(image_sub_dir1, s2)
        # label_dir / label_road02 /label/record001
        label_sub_dir2 = os.path.join(label_sub_dir1, s2)

        # Record001
        for s3 in os.listdir(image_sub_dir2):
            #image_dir/road02/record001/camera 5
            image_sub_dir3 = os.path.join(image_sub_dir2, s3) #sub3目录下对应的就是我们真实的图片了
            label_sub_dir3 = os.path.join(label_sub_dir2, s3)

            # Camera 5
            for s4 in os.listdir(image_sub_dir3):
                s44 = s4.replace('.jpg','_bin.png') #名字后缀进行转换，对应label的名字
                #
                image_sub_dir4 = os.path.join(image_sub_dir3, s4) #每一张真实图片的位置，
                label_sub_dir4 = os.path.join(label_sub_dir3, s44) #每一个label的真实的位置
                if not os.path.exists(image_sub_dir4):
                    print(image_sub_dir4)
                    continue
                if not os.path.exists(label_sub_dir4):
                    print(label_sub_dir4)

                #建立原image和label的一一对应的关系，并且传递到list中
                image_list.append(image_sub_dir4)
                label_list.append(label_sub_dir4)

assert len(image_list) == len(label_list) #image_list 和 label_list 的长度必须是一致的
print("The length of image dataset is {}, and label is {}".format(len(image_list), len(label_list)))

#找到image的总长度
total_length = len(image_list)

#划分dataset为 tarin：validate：test=6:2:2
sixth_part = int(total_length*0.6)  #取60%的图片进行训练
eighth_part = int(total_length*0.8) #取20%的图片进行验证

all = pd.DataFrame({'image':image_list, 'label':label_list})
all_shuffle = shuffle(all)

train_dataset = all_shuffle[:sixth_part]
val_dataset = all_shuffle[sixth_part:eighth_part]
test_dataset = all_shuffle[eighth_part:]

#讲这三个数据集存入到csv的文件中
train_dataset.to_csv('../data_list/train.csv', index=False)
val_dataset.to_csv('../data_list/val.csv', index=False)
test_dataset.to_csv('../data_list/test.csv', index=False)