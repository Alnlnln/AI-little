#CNN模型训练

import tensorflow as tf
from tensorflow.keras import layers
from tensorflow import keras
import matplotlib.pyplot as plt
from time import *

# 加载数据集
def data_load(data_dir, img_height, img_width, batch_size):     #四个参数：图像数据路径，图像高度数据，图像宽度数据，每批次图像数量
    
    #创建训练数据集   
    train_ds = tf.keras.preprocessing.image_dataset_from_directory(
        data_dir,
        label_mode='categorical',
        validation_split=0.1,   #指定从训练集中划分出10%的数据作为验证集
        subset="training",
        seed=123,
        image_size=(img_height, img_width),
        batch_size=batch_size)
    
    #创建验证数据集

    val_ds = tf.keras.preprocessing.image_dataset_from_directory(
        data_dir,
        label_mode='categorical',
        validation_split=0.2,   #指定从训练集中划分出20%的数据作为验证集
        subset="validation",
        seed=123,
        image_size=(img_height, img_width),
        batch_size=batch_size)
    
    class_names = train_ds.class_names  #存储所有类别名称

    return train_ds, val_ds, class_names

# 模型加载，指定图片处理的大小和是否进行迁移学习
def model_load(IMG_SHAPE=(224, 224, 3), class_num=6):   #图像参数（224,224,3），类别数6
    model = tf.keras.models.Sequential([
        tf.keras.layers.experimental.preprocessing.Rescaling(1. / 255, input_shape=IMG_SHAPE),
        tf.keras.layers.Conv2D(32, (3, 3), activation='relu'), #添加卷积层
        tf.keras.layers.MaxPooling2D(2, 2),

        tf.keras.layers.Conv2D(64, (3, 3), activation='relu'),  #再次添加卷积层
        tf.keras.layers.MaxPooling2D(2, 2),
        tf.keras.layers.Flatten(),

        tf.keras.layers.Dense(128, activation='relu'),
        tf.keras.layers.Dense(class_num, activation='softmax')  #添加Dense层做输出层
    ])
    model.summary()
    # 模型训练
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    return model


# 创建函数展示训练过程的曲线
def show_loss_acc(history):
    acc = history.history['accuracy']           #提取训练集准确率
    val_acc = history.history['val_accuracy']   #提取验证集准确率

    loss = history.history['loss']              #提取训练集损失
    val_loss = history.history['val_loss']      #提取验证集损失

    #绘制准确率曲线

    plt.figure(figsize=(8, 8))
    plt.subplot(2, 1, 1)
    plt.plot(acc, label='Training Accuracy')
    plt.plot(val_acc, label='Validation Accuracy')
    plt.legend(loc='lower right')
    plt.ylabel('Accuracy')
    plt.ylim([min(plt.ylim()), 1])
    plt.title('Training and Validation Accuracy')

    #绘制损失曲线

    plt.subplot(2, 1, 2)
    plt.plot(loss, label='Training Loss')
    plt.plot(val_loss, label='Validation Loss')
    plt.legend(loc='upper right')
    plt.ylabel('Cross Entropy')
    plt.title('Training and Validation Loss')
    plt.xlabel('epoch')
    plt.savefig('results_cnn.png', dpi=100)

#训练模型，并绘制曲线
    
def train(epochs):
    begin_time = time()
    train_ds, val_ds, class_names = data_load("C:/Users/JZX/Desktop/trash classification (1)/data/trash_jpg", 224, 224, 16)
    print(class_names)
    model = model_load(class_num=len(class_names))
    history = model.fit(train_ds, validation_data=val_ds, epochs=epochs)
    model.save("models/CNN_model")
    end_time = time()
    run_time = end_time - begin_time
    print('该循环程序运行时间：', run_time, "s")  # 打印循环时间，该循环程序运行时间： 1.4201874732
    show_loss_acc(history)      #绘制曲线



if __name__ == '__main__':
    train(epochs=30)          #训练30个周期