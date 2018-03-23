# Identifier splitting via CNN-BiLSTM-CRF 
### (The source code is implemented with Tensorflow 1.4)
---
### User Guides:
1. **Prerequisites**: You must have *Tensorflow* and *Python 3+* installed.
<br/> With regard to Tensorflow, we strongly recommend you install *Tensorflow 1.4* to avoid any imcompatibility.
2. If you want to infer a identifier splitting with our **ready-made package**, just use the command line below:
```
python is_inference.py -i identifier --save_path Model/model.ckpt
```
for more than one identifiers, use command line in the form of: (no blank between concatentated comma)
```
python is_inference.py -i identifier1,identifier2,...,identifierN --save_path Model/model.ckpt
```
or for a csv file consisting of plenty of identifiers
```
python is_inference.py -f idenntifiers.csv --save_path Model/model.ckpt
```
--save_path specify the location of trained model file. We provide a netdisk download-link of [our model](https://pan.baidu.com/s/1p8UvdL2MPq9sDwY3oH2eWg), because it exceeds the file size limit of Github. The download password is *s923* <br/>
3. An example test case is shown below:
```
python is_inference.py -i treenode,sfile,colspan,printluck --save_path Model/model.ckpt
```
with the corresponding output:
```
Splitting results are shown below:
treenode                        ==>  tree-node                      
sfile                           ==>  s-file                         
colspan                         ==>  col-span                       
printluck                       ==>  print-luck    
```
4. To **adpot or revise our code**, we provide you some indications of related files:
* ```Model/model.ckpt``` stores the trained-model, in a Tensorflow model(. ckpt) format
* ```Oracles/``` stores all the oracles used in our study, namely *Binkley*, *BT11*, *Jhotdraw* and *Lynx*
* ```is_modeltrainning.py``` is the most important file, which contains the core code to train our CNN-BiLSTM-CRF Model.

### Training Enviroment:
1. Macbook pro 2015, 2.5 GHz Intel Core i7, CPU built tensorflow 1.4 - Jaki's PC
2. Nvidia GTX GEForce 1060, Ubuntu 16.04, docker of GPU built tensorflow 1.4 - Rainlf's PC
3. Nvidia GTX 970X, CUDA 8.0, Ubuntu 14.04, cNN6.1, GPU built tensorflow 1.4 - Robbie's pC
4. Alibaba ecs.gn5-c4g1.xlarge（1 * NVIDIA P100, 4核 30GB，GPU计算型 gn5）	

### The running details of other benchmarking techniques are described below:
* GenTest
	We implement GenTest by invocating the [web service on-line](http://splitit.cs.loyola.edu/web-service.html) provided by Binkley et al. The programming language of each identifier sample in the Binkley dataset is explicitly recorded. Further, all the identifier samples in the BT11 dataset are known to be extracted from Java projects. Thus, we can optimally use GenTest by indicating the language of each identifier when we invoke the http requests. 
* LIDS
	We implement LIDS by using its [command-line interface tool](https://github.com/nunorc/Lingua-IdSplitter) written in Perl language. The dependent Perl module is also available in the cpan repository\footnote{http://search.cpan.org/}.
* INTT
	INTT is actually a [Java library](http://oro.open.ac.uk/28352/) that implements an approach to enable the automated tokenization of identifier names. It is implemented by Butler et al. and also made available in the [maven repository](http://mvnrepository.com/artifact/uk.org.facetus/intt).


### To-do:
Use [**tensorflow serving**](https://www.tensorflow.org/serving/serving_basic) to provide continuous grpc or restful service, so that it won't cost us the model-loading time every time.

### About the authors:
```
The project is maintained by Xlab and Software Engineering R&D Centre, Tongji University.
```
If you have any question, welcome to contact us. <br/>
Email: lijiechu@qq.com 
Website: [Xlab](www.x-lab.ac)
