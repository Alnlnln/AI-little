本例程使用的环境及其版本号如下：(requirement.txt)
tensorflow-cpu == 2.3.0;
pyqt5;
pillow;
opencv-python;
matplotlib;

基于tensorflow2.3的垃圾分类系统

代码结构
images 目录主要是放置一些图片，包括测试的图片和ui界面使用的图片
models 目录下放置训练好的两组模型，分别是cnn模型和mobilenet的模型
results 目录下放置的是训练的训练过程的一些可视化的图，两个txt文件是训练过程中的输出，两个图是两个模型训练过程中训练集和验证集准确率和loss变化曲线
utils 是主要是我测试的时候写的一些文件，对这个项目没有实际的用途
mainwindow.py 是界面文件，主要是利用pyqt5完成的界面，通过上传图片可以对图片种类进行预测
testmodel.py 是测试文件，主要是用于测试两组模型在验证集上的准确率，这个信息你从results的txt的输出中也能获取
train_cnn.py 是训练cnn模型的代码
train_mobilenet.py 是训练mobilenet模型的代码
