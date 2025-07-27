# Agriculture-Farmer-kit

PROBLEM STATEMENT 
Agriculture plays a crucial role in the global economy and is vital for food security. 
However, farmers face numerous challenges in optimizing their agricultural practices, 
managing resources effectively, and maximizing crop yields. Traditional farming 
methods often lack access to real-time data, advanced analytics, and technology-driven 
solutions, resulting in inefficient resource allocation and suboptimal decision-making. 

OBJECTIVE 
The primary objective of the E-AGRI KIT is to revolutionize agriculture through 
innovative technology. Our focus lies in developing a robust deep learning algorithm 
for precise crop identification, bridging the gap between farmers and investors. We 
aspire to create a seamless platform where farmers can showcase their crops, investors 
can make informed decisions on crop funding, and buyers can access crops at fair 
market prices. Through this, we aim to optimize agricultural processes, promote 
financial inclusivity, and fortify the agricultural supply chain. 

SCOPE 
The scope of the E-AGRI KIT extends across the agricultural landscape, offering a 
comprehensive solution to multiple challenges. From leveraging deep learning for accurate 
crop identification to establishing an investment platform for agricultural ventures, our 
scope encompasses technological advancements that benefit farmers, investors, and buyers 
alike. By facilitating efficient crop selling processes, our application aims to contribute to 
connected and empowered agricultural community.

PROJECT INTRODUCTION 
According to a study by the Associated Chambers of Commerce and Industry of India, 
annual crops losses due to pests amount to Rs.50,000 crore ($500 billion), which is 
tantamount to a country where at least 200 million go to bed hungry every night. 
Agriculture being a vital sector has a majority of the rural population in developing 
countries relying on it. 
The sector is faced by major challenges like unprecedented pest attack and unforeseen 
weather conditions affecting their produce leading to major loss of food and effort. 
Technology plays a vital role in uplifting the livelihoods of the rural populace which can 
be done by using a simple agro-android application system. 
Plant can affect vast produce of crops posing a major menace to food security as well as 
leading to major losses to farmers. An extensive review of existing research was conducted 
by us on this domain and in an effort to help farmers overcome this problem, we have 
designed an android application,  
In this work, several approaches were tried and tested to form an automated plant detection 
system which would later be integrated with an agricultural aid application. Section 2 
describes the application and its overall working and architecture using UML diagrams. 
Section 3 describes the dataset used for training and testing the models and dataset 
parameters along with a snapshot of the images in the dataset. 

BLOCK DIAGRAM 

<img width="1171" height="1086" alt="image" src="https://github.com/user-attachments/assets/3cd9150a-475a-46cb-9612-3b2e0afda452" />

PROJECT FLOW

<img width="1295" height="1278" alt="image" src="https://github.com/user-attachments/assets/8311ea4f-6f8f-4eda-9222-8e45c3b82277" />


ARCHITECTURE

<img width="1323" height="1051" alt="image" src="https://github.com/user-attachments/assets/2dd69f56-797f-46ad-9f73-d1bd8d607a02" />


METHODOLOGY AND ALGORITHMS 

CNN WITH MOBILENET: 
• We are using the pre-processed training dataset to train our model using 
CNN algorithm. 
• CNN algorithm consists of 4 layers: Input layer, Convolution Layer, pooling layer, 
Flatten layer and dense layer. 
• In input layer we consider images as input. 
• In Convolution layer, we convert image into matric format. Here matrix size is 1024 
X 1024 (rows X columns). 
• In the pooling layer the numerical values will be stored. To change the numerical 
data to binary data, we use machine learning algorithm named SoftMax (supervised 
learning algorithm). In SoftMax layer we will convert the numerical data to binary. 
• In flatten layer and dense the classes of total dataset (17 types) are stored which 
will be in the binary data format. 
• We use fit generator method for saving the data in the form of .h5. Here model is a 
format for storing the binary data. 
CNN WITH RESNET50: 
• We are using the pre-processed training dataset to train our model using CNN 
algorithm with Resnet50 model. 
• CNN algorithm consists of 4 layers: Input layer, Convolution Layer, pooling layer, 
Flatten layer and dense layer. 
• In input layer we consider images as input. 
13 
• In Convolution layer, we convert image into matric format. Here matrix size is 1024 
X 1024 (rows X columns). 
• In the pooling layer the numerical values will be stored. To change the 
numerical data to binary data, we use machine learning algorithm 
named SoftMax (supervised learning algorithm). In SoftMax layer we 
will convert the numerical data to binary. 
• In flatten layer and dense the classes of total dataset (17 types) is stored 
which will be in the binary data format. 
• We use fit method for saving the data in the form of .h5. Here model is 
a format for storing the binary data. 

RANDOM FOREST REGRESSION: 
• Irrespective of the algorithm we select the training is the same for every algorithm. 
• Given a dataset we split the data into two parts training and testing, the reason 
behind doing this is to test our model/algorithm performance just like the exams for 
a student the testing is also exam for the model. 
• We can split data into anything we want but it is just good practice to split the data 
such that the training has more data than the testing data, we generally split the data 
into    70% training and 30% testing. 
• And for training and testing there are two variables X and Y in each of them, the X 
is the features that we use to predict the Y target and same for the testing also. 
• Then we call the .fit ( ) method on any given algorithm which takes two parameters 
i.e., X and Y for calculating the math and after that when we call the .predict ( ) 
giving our testing X as parameter and checking it with the accuracy score giving 
the testing Y and predicted X as the two parameters will get our accuracy score and 
same steps for the classification report and the confusion matrix, these are just 
checking for how good our model performed on a given dataset

ACCURACY

<img width="983" height="658" alt="image" src="https://github.com/user-attachments/assets/309df6cd-efb2-42d0-af7e-38e12f0cae0d" />

REFERENCES

1. Sharada P. Mohanty David P. Hughes and Marcel Salathé.” Using Deep Learning for 
Image-Based Plant Disease Detection.” Front. Plant Sci., 22 September 2016  
2. Carsten Rother, Vladimir Kolmogorov, and Andrew Blake. 2004. “GrabCut”: 
interactive foreground extraction using iterated graph cuts. In ACM SIGGRAPH 2004 
Papers (SIGGRAPH ’04). Association for Computing Machinery, New York, NY, 
USA, 309–314 
3. R. Chapaneri, M. Desai, A. Goyal, S. Ghose and S. Das, "Plant Disease Detection: A 
Comprehensive Survey," 2020 3rd International Conference on Communication 
System, Computing and IT Applications (CSCITA), Mumbai, India, 2020, pp. 220
225, doi: 10.1109/CSCITA47329.2020.9137779.  
4. Raghavendra, B. K. (2019, March). Diseases Detection of Various Plant Leaf Using 
Image Processing Techniques: A Review. In 2019 5th International Conference on 
Advanced Computing & Communication Systems (ICACCS), (pp. 313-316). IEEE.  
5. Malathi, M., Aruli, K., Nizar, S. M., & Selvaraj, A. S. (2015). A Survey on Plant Leaf 
Disease Detection Using Image Processing Techniques. International Research Journal 
of Engineering and Technology (IRJET), 2(09)  
6. Kaur, S., Pandey, S., & Goel, S. (2018). Semi-automatic leaf disease detection and 
classification system for soybean culture. IET Image Processing, 12(6), 1038-1048.  
7. Patil, S., & Chandavale, A. (2015). A survey on methods of plant disease detection. 
International journal of Science and Research (IJSR), 4(2), 1392-1396.  
8. [10] Rathod, A. N., Tanawala, B. A., & Shah, V. H. (2014). Leaf disease detection 
using image processing and neural network. International Journal of Advance 
Engineering and Research Development (IJAERD), 1(6).  
9. "Computational Vision and Bio-Inspired Computing", Springer Science and Business, 
Media LLC, 2020.  
10. Glorot, Xavier & Bordes, Antoine & Bengio, Y. (2010), “Deep Sparse Rectifier Neural 
Networks,” Journal of Machine Learning Research. 15.  
40 
11. K. He, X. Zhang, S. Ren and J. Sun, "Deep Residual Learning for Image Recognition," 
2016 IEEE Conference on Computer Vision and Pattern Recognition (CVPR), Las 
Vegas, NV, 2016, pp. 770- 778, doi: 10.1109/CVPR.2016.90.  
12. Szegedy, Christian et al. “Going deeper with convolutions.” 2015 IEEE Conference on 
Computer Vision and Pattern Recognition (CVPR) (2015): 1-9




