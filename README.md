# NN_Study
The learning process of deep learning
│  pytorchLearning
│  
├─CNN
│      backward.py
│      cat1.jpg
│      CNN_01.py
│      DatasetAndLoader.py
│      graph_aug.py
│      InceptionA.py
│      kaggle_OTTO.py
│      linearModel.py
│      LinearWithPytorch.py
│      Logistic.py
│      mnist.py
│      mnist_CNN.py
│      multipleDimensionInput.py
│      README.md
│      ResNet.py
│      test.py
│      
├─dataset
│  │  otto-group-product-classification-challenge.zip
│  │  
│  ├─eg
│  │      diabetes.csv.gz
│  │      diabetes_data.csv.gz
│  │      diabetes_target.csv.gz
│  │      
│  ├─mnist
│  │  ├─FashionMNIST
│  │  │  └─raw
│  │  │          t10k-images-idx3-ubyte
│  │  │          t10k-images-idx3-ubyte.gz
│  │  │          t10k-labels-idx1-ubyte
│  │  │          t10k-labels-idx1-ubyte.gz
│  │  │          train-images-idx3-ubyte
│  │  │          train-images-idx3-ubyte.gz
│  │  │          train-labels-idx1-ubyte
│  │  │          train-labels-idx1-ubyte.gz
│  │  │          
│  │  └─MNIST
│  │      └─raw
│  │              t10k-images-idx3-ubyte
│  │              t10k-images-idx3-ubyte.gz
│  │              t10k-labels-idx1-ubyte
│  │              t10k-labels-idx1-ubyte.gz
│  │              train-images-idx3-ubyte
│  │              train-images-idx3-ubyte.gz
│  │              train-labels-idx1-ubyte
│  │              train-labels-idx1-ubyte.gz
│  │              
│  ├─otto-group-product-classification-challenge
│  │      sampleSubmission.csv
│  │      test.csv
│  │      train.csv
│  │      
│  └─Titanic
│          train.csv
│          
├─GNN
│  │  build_datasets.py
│  │  defineGCN.py
│  │  initial.py
│  │  myGNN.py
│  │  tst.py
│  │  
│  ├─REshow
│  │  │  1.py
│  │  │  
│  │  └─dataset
│  │      ├─test
│  │      │      0.json
│  │      │      1.json
│  │      │      10.json
│  │      │      11.json
│  │      │      12.json
│  │      │      13.json
│  │      │      14.json
│  │      │      15.json
│  │      │      16.json
│  │      │      17.json
│  │      │      18.json
│  │      │      19.json
│  │      │      2.json
│  │      │      20.json
│  │      │      21.json
│  │      │      22.json
│  │      │      23.json
│  │      │      24.json
│  │      │      25.json
│  │      │      26.json
│  │      │      27.json
│  │      │      28.json
│  │      │      29.json
│  │      │      3.json
│  │      │      30.json
│  │      │      31.json
│  │      │      32.json
│  │      │      33.json
│  │      │      34.json
│  │      │      35.json
│  │      │      36.json
│  │      │      37.json
│  │      │      38.json
│  │      │      39.json
│  │      │      4.json
│  │      │      40.json
│  │      │      41.json
│  │      │      42.json
│  │      │      43.json
│  │      │      44.json
│  │      │      45.json
│  │      │      46.json
│  │      │      47.json
│  │      │      48.json
│  │      │      49.json
│  │      │      5.json
│  │      │      6.json
│  │      │      7.json
│  │      │      8.json
│  │      │      9.json
│  │      │      
│  │      └─train
│  │              0.json
│  │              1.json
│  │              10.json
│  │              11.json
│  │              12.json
│  │              13.json
│  │              14.json
│  │              15.json
│  │              16.json
│  │              17.json
│  │              18.json
│  │              19.json
│  │              2.json
│  │              20.json
│  │              21.json
│  │              22.json
│  │              23.json
│  │              24.json
│  │              25.json
│  │              26.json
│  │              27.json
│  │              28.json
│  │              29.json
│  │              3.json
│  │              30.json
│  │              31.json
│  │              32.json
│  │              33.json
│  │              34.json
│  │              35.json
│  │              36.json
│  │              37.json
│  │              38.json
│  │              39.json
│  │              4.json
│  │              40.json
│  │              41.json
│  │              42.json
│  │              43.json
│  │              44.json
│  │              45.json
│  │              46.json
│  │              47.json
│  │              48.json
│  │              49.json
│  │              5.json
│  │              6.json
│  │              7.json
│  │              8.json
│  │              9.json
│  │              
│  └─similarity
│      │  layers.py
│      │  main.py
│      │  param_parser.py
│      │  simgnn.py
│      │  text.py
│      │  utils.py
│      │  
│      ├─dataset
│      │  ├─test
│      │  │      0.json
│      │  │      1.json
│      │  │      10.json
│      │  │      11.json
│      │  │      12.json
│      │  │      13.json
│      │  │      14.json
│      │  │      15.json
│      │  │      16.json
│      │  │      17.json
│      │  │      18.json
│      │  │      19.json
│      │  │      2.json
│      │  │      20.json
│      │  │      21.json
│      │  │      22.json
│      │  │      23.json
│      │  │      24.json
│      │  │      25.json
│      │  │      26.json
│      │  │      27.json
│      │  │      28.json
│      │  │      29.json
│      │  │      3.json
│      │  │      30.json
│      │  │      31.json
│      │  │      32.json
│      │  │      33.json
│      │  │      34.json
│      │  │      35.json
│      │  │      36.json
│      │  │      37.json
│      │  │      38.json
│      │  │      39.json
│      │  │      4.json
│      │  │      40.json
│      │  │      41.json
│      │  │      42.json
│      │  │      43.json
│      │  │      44.json
│      │  │      45.json
│      │  │      46.json
│      │  │      47.json
│      │  │      48.json
│      │  │      49.json
│      │  │      5.json
│      │  │      6.json
│      │  │      7.json
│      │  │      8.json
│      │  │      9.json
│      │  │      
│      │  └─train
│      │          0.json
│      │          1.json
│      │          10.json
│      │          11.json
│      │          12.json
│      │          13.json
│      │          14.json
│      │          15.json
│      │          16.json
│      │          17.json
│      │          18.json
│      │          19.json
│      │          2.json
│      │          20.json
│      │          21.json
│      │          22.json
│      │          23.json
│      │          24.json
│      │          25.json
│      │          26.json
│      │          27.json
│      │          28.json
│      │          29.json
│      │          3.json
│      │          30.json
│      │          31.json
│      │          32.json
│      │          33.json
│      │          34.json
│      │          35.json
│      │          36.json
│      │          37.json
│      │          38.json
│      │          39.json
│      │          4.json
│      │          40.json
│      │          41.json
│      │          42.json
│      │          43.json
│      │          44.json
│      │          45.json
│      │          46.json
│      │          47.json
│      │          48.json
│      │          49.json
│      │          5.json
│      │          6.json
│      │          7.json
│      │          8.json
│      │          9.json
│      │          
│      └─__pycache__
│              layers.cpython-36.pyc
│              param_parser.cpython-36.pyc
│              simgnn.cpython-36.pyc
│              utils.cpython-36.pyc
│              
├─pytorch_train
│      basic_operation.py
│      get_dataset.py
│      linear.py
│      test.py
│      
└─RNN
    │  create.py
    │  eg1.py
    │  
    └─name
        │  name_classifier.py
        │  test.py
        │  
        └─data
                names_test.csv.gz
                names_train.csv.gz
