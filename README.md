# MNIST_train
使用对比学习对MNIST数据集进行预训练和分类
功能：1. 使用Keras构建模型
      2. 自定义DataGenerater生成minibatch的样本图片和标签进行训练
      3. 自定义对比损失函数代码
      4. 预训练结束后使用TSNE可视化处理
      5. 预训练结束后训练MLP层进行分类并告知准确率(accuracy)
———————————————————————————————————————————

Pretraining and Classifying the MNIST Dataset Using Contrastive Learning
Funcation:1. Use Keras to build the model
          2. Customize the DataGenerator to generate minibatch sample pictures and labels for training
          3. Custom contrastive loss function code
          4. Use TSNE visualization after pre-training
          5. After the pre-training is over, train the MLP layer to classify and inform the accuracy (accuracy)
