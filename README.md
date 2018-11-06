# keras_tought
学习一下keras并简单进行记录学习过程

## bilstm_attention.py
目前正在尝试使用bilstm+attention进行ner，但是attention这层目前存在一些问题，需要再研究一下。
目前实现了，bilstm进行序列标注，对文本进行命名实体识别，数据集使用的还是人民日报语料。

## val.py
直接运行就好，模型已经训练好的，之后会加入测试集检测结果

## case

其中检测一下效果：如下所示：
sentence = ‘中华人民共和国国务院总理周恩来和付子玉在外交部长陈毅的陪同下，连续访问了埃塞俄比亚等非洲10国以及阿尔巴尼亚’
![](https://ws3.sinaimg.cn/large/006tNbRwly1fwy6x6tmy4j30wx01wdhv.jpg)

## work to do
最近有些忙，之后会将内容进行修改，把attlay层的bug调整一下，再做一下测试集检测结果。
