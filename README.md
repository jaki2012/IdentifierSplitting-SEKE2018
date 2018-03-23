## Dear reviewers and readers,
## Sorry to show a messy project right now.. We will tidy it up as soon as possible, before March 25. But it still works well, basically.

---
### Source code for identifier splitting via CNN-BiLSTM-CRF 
#### (implemented by ```Tensorflow 1.4```)

### User Guides:
1. **Prerequisites**: You must have *Tensorflow* and *Python 3+* installed.
<br /> With regard to Tensorflow, we strongly recommend you install *Tensorflow 1.4* to avoid any imcompatibility.
2. If you want to infer a identifier splitting with our **ready-made package**, just use the command line below:
```
python is_inference.py identifier
```
for more than one identifiers, use command line in the form of 
```
python is_inference.py -i identifier1,identifier2,...,identifierN
```
or for a csv file consisting of plenty of identifiers
```
python is_inference.py -f idenntifiers.csv
```
3. To **adpot or revise our code**, we provide you some indications of related files:
* ```Model/model.ckpt``` stores the trained-model, in a Tensorflow model(. ckpt) format
* ```Oracles/``` stores all the oracles used in our study, namely *Binkley*, *BT11*, *Jhotdraw* and *Lynx*
* ```is_modeltrainning.py``` is the most important file, which contains the core code to train our CNN-BiLSTM-CRF Model.

---
### Training Enviroment:
1. Macbook pro 2015, 2.5 GHz Intel Core i7, CPU built tensorflow 1.4 - Jaki's PC
2. Nvidia GTX GEForce 1060, Ubuntu 16.04, docker of GPU built tensorflow 1.4 - Rainlf's PC
3. Nvidia GTX 970X, CUDA 8.0, Ubuntu 14.04, cNN6.1, GPU built tensorflow 1.4 - Robbie's pC
4. Alibaba ecs.gn5-c4g1.xlarge（1 * NVIDIA P100, 4核 30GB，GPU计算型 gn5）	

### Sub-Project related to this project
1. TRIS（有向图、最短路径—Dijkstra算法）
2. LIDS（Perl语言 CPAN）
3. CodeNet（常用程序字典、应用领域、缩写大小写）
4. [Latex TABLE Generator](http://www.tablesgenerator.com/)：支持latextable生成，导出tgn，根据latex代码生成表格等


### The running details are described below:
* GenTest
	We implement GenTest by invocating the [web service on-line](http://splitit.cs.loyola.edu/web-service.html) provided by Binkley et al. The programming language of each identifier sample in the Binkley dataset is explicitly recorded. Further, all the identifier samples in the BT11 dataset are known to be extracted from Java projects. Thus, we can optimally use GenTest by indicating the language of each identifier when we invoke the http requests. 
* LIDS
	We implement LIDS by using its [command-line interface tool](https://github.com/nunorc/Lingua-IdSplitter) written in Perl language. The dependent Perl module is also available in the cpan repository\footnote{http://search.cpan.org/}.
* INTT
	INTT is actually a [Java library](http://oro.open.ac.uk/28352/) that implements an approach to enable the automated tokenization of identifier names. It is implemented by Butler et al. and also made available in the [maven repository](http://mvnrepository.com/artifact/uk.org.facetus/intt).
