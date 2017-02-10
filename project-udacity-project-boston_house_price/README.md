# udacity-project-one-
predict the boston house price

>[boston_housing.ipynb](./boston_housing.ipynb) 是我的作业提交。完成了基本的数据分析和处理工作。

>[review_one.md](./review_one.md) 是作业第一次结果，有几处修改意见，最后作业中都做了一一修改。

# 项目1：模型评估与验证
## 波士顿房价预测

### 准备工作

这个项目需要安装**Python 2.7**和以下的Python函数库：

- [NumPy](http://www.numpy.org/)
- [matplotlib](http://matplotlib.org/)
- [scikit-learn](http://scikit-learn.org/stable/)

你还需要安装一个软件，以运行和编辑[ipynb](http://jupyter.org/)文件。

优达学城推荐学生安装 [Anaconda](https://www.continuum.io/downloads)，这是一个常用的Python集成编译环境，且已包含了本项目中所需的全部函数库。我们在P0项目中也有讲解[如何搭建学习环境](https://github.com/udacity/machine-learning/blob/master/projects_cn/titanic_survival_exploration/README.md)。

### 编码

代码的模版已经在`boston_housing.ipynb`文件中给出。你还会用到`visuals.py`和名为`housing.csv`的数据文件来完成这个项目。我们已经为你提供了一部分代码，但还有些功能需要你来实现才能以完成这个项目。

### 运行

在终端或命令行窗口中，选定`boston_housing/`的目录下（包含此README文件），运行下方的命令：

```jupyter notebook boston_housing.ipynb```

这样就能够启动jupyter notebook软件，并在你的浏览器中打开文件。

### Data

经过编辑的波士顿房价数据集有490个数据点，每个点有三个特征。这个数据集编辑自[加州大学欧文分校机器学习数据集库](https://archive.ics.uci.edu/ml/datasets/Housing).

**特征**

1. `RM`: 住宅平均房间数量
2. `LSTAT`: 区域中被认为是低收入阶层的比率
3. `PTRATIO`: 镇上学生与教师数量比例

**目标变量**

4. `MEDV`: 房屋的中值价格


### 项目提交

波士顿房屋市场竞争异常激烈，您想成为当地最好的房地产中介。为了与同行竞争，您决定使用几个基本的机器学习概念来帮助您和客户找到其房屋的最佳销售价格。幸运的是，您遇到了波士顿房屋数据集，其中包含大波士顿社区的房屋的各种特征的累积数据，包括其中各个地区的房屋价格的中值。您的任务是利用可用工具基于统计分析来构建一个最佳模型。然后使用该模型估算客户房屋的最佳销售价格。

项目文件
你可以在机器学习项目 GitHub 中 projects 下找到 boston_housing 文件夹。你可以下载这个纳米学位项目的整个目录。请确保使用最新的项目文件完成并提交项目！

评估
优达学城的项目评审师会依据此预测波士顿房价项目要求来评审您的提交。提交前请务必确保你已经仔细查看该要求，自己也依此对照检查过了所做的项目。所有要求必须被标注为“合格”，项目才能通过。

须提交文件
提交的文件中应包含以下文件并且可以打包为单个 .zip 文件：

包含完整实现且可正常运行的代码的 “boston_housing.ipynb” 文件，并已执行所有代码块和显示了输出。
一个由jupyter notebook 导出的 HTML 文件，重命名为 report.html。此文件需同 ipynb 文件一起提交才能被审阅。
准备好了！
当你准备好了提交项目，点击页面下方的项目提交按钮。

提交
如果有任何与提交项目相关的疑问，或想查看提交状态，请向我们发送电子邮件 (support@youdaxue.com) 或访问我们的论坛。

后续步骤
在项目导师给出反馈后，您会立即收到电子邮件。在此期间，请继续学习后面的项目和内容！
