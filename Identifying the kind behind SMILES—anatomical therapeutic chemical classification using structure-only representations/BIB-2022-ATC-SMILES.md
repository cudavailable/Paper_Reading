## 标题：
- Identifying the kind behind SMILES—anatomical therapeutic chemical classification using structure-only representations
- 识别SMILES背后的类型-只使用结构表示的解剖治疗化学分类

## 关键词：
Anatomical Therapeutic Chemical, ATC Classification, Drug Development, Deep Learning

## 术语：

### Anatomical Therapeutic Chemical：
	该分类法由世界卫生组织（WHO）开发，ATC分类法主要通过药物的解剖学、治疗学和化学属性来对药物进行分类，使医疗专业人员和研究人员能够更好地理解和比较不同药物之间的特征和用途。

### Rdkit：
	一种用于化学信息学和分子模拟的开源软件库，它提供了丰富的工具和算法，用于处理分子结构、化学反应、药物设计等领域。

### domain knowledge injector：领域知识注入器
	一种工具、方法或系统，用于将特定领域专业知识或规则注入到计算机系统中，以改善系统在该领域的表现和决策能力。

### Word2Vec：
	Word2Vec的主要思想是通过训练一个神经网络模型，根据单词在上下文中的出现情况来学习单词的分布式表示。这种表示方式允许单词之间的语义关系通过它们在向量空间中的距禂来表示。


### Skip-gram：
	Skip-gram是Word2Vec模型的一种架构，用于学习词嵌入（Word Embeddings）。Skip-gram模型的主要思想是通过给定一个单词来预测其周围的上下文单词。Skip-gram模型训练时会从文本数据中提取出训练样本对，其中一个单词作为输入（中心词），而其周围的上下文单词则作为输出。模型的目标是学习一个神经网络，使得给定一个中心词时，可以预测出它周围的上下文单词。Skip-gram模型通过学习单词在文本中的分布情况，将单词表示为密集的低维向量。这些向量在向量空间中被设计成使得语义相似的单词在空间中彼此接近。
	
### kernel pyramid：
	在计算机视觉领域，特别是在图像处理和图像识别任务中，"Kernel Pyramid"是一种用于多尺度特征提取的技术。这个概念涉及使用不同大小的卷积核（Kernels）来处理图像，以便从不同尺度上获取信息。
	一种常见的做法是通过使用不同尺寸的卷积核来构建金字塔，每一层都会从图像中提取不同尺度的特征。这种多尺度特征提取有助于在处理图像时捕获不同尺度下的对象和结构。

## 内容：

### 1 Introduction：

**新趋势**：
	1. 深度学习的应用。
	2. 新数据源的不断丰富。

**限制：**
	1. 新资源的纳入导致新的问题，比如模型和表征复杂度的提升、计算资源的要求提高、需要额外的外部查询工具rdkit。
	2. 新资源依赖于STITCH，ATC预测来源于实验数据，因此难以预测未见的药物类型。

**贡献：**
	1. 收集了一个新数据集，可用于只基于结构的方法或者对于基线的补充。
	2. 数据集、源代码和web服务器将公开。
	3. 提出了一个轻量级的DL模型(ATC-CNN)，优于SOTA方法。

### 2 Materials and Methods

#### Benchmark Dataset

#### Problem Formulation

**立意**：找到同时具备统计意义和物化意义的token，构建有效、高效的模型，并提高模型的可解释性。

**tokenization过程有三步：**
	1. Token Extractor提供候选token和相应的置信分数。
	2. 领域知识注入器：人类专家标定出高置信度候选token的物化意义，并构建token字典。这将为后面的提取产生规则。
	3. Sequence Validator：对于一个给定的序列，从所有token组合中找到最好的partition。

#### Tokenization and Representation Generation：

##### Text Extractor：

- 用平均tf-idf作为置信分数，查询top-20候选token，并且传送给Human Knowledge Injector。

##### Human Knowledge Injector：

- Injector以top-20的token作为输入，并且需要人类专家的介入。专家标定出有物化意义的token并加入token字典。

- 将token字典里的token根据物化性质进行分类。
- 每组设定选择标准，使结果可解释。

##### Sequence Validator：

- The validator works together with the Token Extractor to split a sequence into actual tokens based the token dictionary D and transition matrix T.

- Validator会在k+1步时，选出使得过去到现在概率最大token t，这是一个常见的Merkov链。

##### Generating the Representation x using Word Embedding：

- Once the tokens are extracted from sequences, we pool them for word embedding.

- token -> vector  (token在词向量空间中会与上下文相关的token更靠近。而token的向量化可以使用Word2Vec模型)

#### Model f and Parameters θ：

*ATC-CNN is a seven-stream and light-weight CNN.*

##### Convolutional Layers：

- 卷积层作为提取器，提取2m个相邻的token并总结它们之间的联系。

- 随着kernel size的增大，CNN能都“看到”更大范围的分子分支。

##### FC Layers and Predictions ˆy：
	
- Dropout layer (dropout rate = 0.2)

##### Loss Function L(ˆy, y) and Parameter Learning:

- Loss function : BCE (logits loss function)
- optimizer : Adam
- learning rate : 0.001
- batch_size = 16

### 3 Results and Discussion：

#### Metrics for Multi-label ATC Classification：
	Aiming / Coverage / Accuracy / Absolute True / Absolute False

#### Cross-validation：
	jackknife test[47] for cross-validation

#### Comparison with SOTA Methods：
	ATC-CNN outperforms the SOTA methods by 1.62%, 6.40%,7.15%, 7.68% and 0.22% on Chen-2012[4] in Aiming, Coverage, Accuracy, Absolute True and Absolute False, respectively.

- **优点**：在14label中有13个label的accurancy超过80%，CGATCPred只有在仅仅4类上能够与之一比。而且ATC-CNN更加稳定。

- **缺点**：两个方法都在类别B上的准确率低于80%。因为类别B有着远超其他类别的无机盐占比，SIMLIES长度（3-7）远远小于标准表征的长度787，故其中有大量的0填充，导致信息含量低于其他类别。

#### Web Server：

- Web server at http://www.aimars.net:8090/ATC_SMILES/

### 4 Conclusion：

- Construct a **new benchmark ATC-SMILES for ATC classification** which is with larger scale than transitional benchmarks and eliminates the reliance on STITCH database.

- Propose a **new tokenization process** which extracts and embeds statistically and physicochemically meaningful tokens.

- Propose a **molecular structure-only deep learning method** which is with better explainability.

- The proposed method **outperforms** the state-of-the-art methods.

### 5 Code and data availability：

 - https://github.com/lookwei/ATC_CNN