Bilibili: [手把手教你机器学习Kaggle比赛冲到全球 Top 2%](https://www.bilibili.com/video/av9399358/?from=search&seid=9477089273895124407)

Github: [ypwhs/dogs_vs_cats](https://github.com/ypwhs/dogs_vs_cats)

Kaggle: [Dogs vs. Cats Redux: Kernels Edition](https://www.kaggle.com/c/dogs-vs-cats-redux-kernels-edition/data) This is new version of compitition

blog: [杨培文](https://ypw.io/)

keras 中文文档: [面向小数据集构建图像分类模型](http://keras-cn.readthedocs.io/en/latest/blog/image_classification_using_very_little_data/)


## 更多提升方式

* y_pred = y_pred.clip(min=0.005, max=0.995) 注意这个小技巧，来处理比赛评分的问题
* 采用更新的更多的网络加进来，提取特征向量，进行测试。
* 对模型进行进一步微调，对选择的卷积网络只提取前面的卷积层部分。对最后的层进行修改来让网络更加专注的应用于猫狗的分类。可以关注keras的文章，[面向小数据集构建图像分类模型](http://keras-cn.readthedocs.io/en/latest/blog/image_classification_using_very_little_data/)
* 为了尽量利用我们有限的训练数据，我们将通过一系列随机变换堆数据进行提升，这样我们的模型将看不到任何两张完全相同的图片，这有利于我们抑制过拟合，使得模型的泛化能力更好。在Keras中，这个步骤可以通过keras.preprocessing.image.ImageGenerator来实现。
