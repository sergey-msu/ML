using System;
using ML.DeepMethods.Models;
using ML.Core;
using ML.Core.Registry;
using ML.DeepMethods.Algorithms;
using ML.DeepMethods.Registry;
using System.Reflection;

namespace ML.DeepTests
{
  public static class Examples
  {
    #region MNIST

    /// <summary>
    /// Error = 0.92
    /// </summary>
    public static BackpropAlgorithm CreateMNISTSimpleDemo_SEALED()
    {
      Console.WriteLine("init CreateMNISTSimpleDemo_SEALED");
      var activation = Activation.LeakyReLU();
      var net = new ConvNet(1, 28) { IsTraining=true };

      net.AddLayer(new ConvLayer(outputDepth: 12, windowSize: 5, padding: 2));
      net.AddLayer(new ConvLayer(outputDepth: 12, windowSize: 5, padding: 2));
      net.AddLayer(new MaxPoolingLayer(windowSize: 2, stride: 2, activation: activation));
      net.AddLayer(new ConvLayer(outputDepth: 24, windowSize: 5, padding: 2));
      net.AddLayer(new MaxPoolingLayer(windowSize: 2, stride: 2, activation: activation));
      net.AddLayer(new FlattenLayer(outputDim: 32, activation: activation));
      net.AddLayer(new DropoutLayer(rate: 0.5D));
      net.AddLayer(new DenseLayer(outputDim: 10, activation: activation));

      net._Build();

      net.RandomizeParameters(seed: 0);

      var lrate = 0.001D;
      var alg = new BackpropAlgorithm(net)
      {
        EpochCount = 500,
        LearningRate = lrate,
        BatchSize = 4,
        UseBatchParallelization = true,
        MaxBatchThreadCount = 4,
        LossFunction = Loss.Euclidean,
        Optimizer = Optimizer.RMSProp,
        Regularizator = Regularizator.L2(0.0001D),
        LearningRateScheduler = LearningRateScheduler.DropBased(lrate, 5, 0.5D)
      };

      alg.Build();

      return alg;
    }

    /// <summary>
    ///
    /// </summary>
    public static BackpropAlgorithm CreateMNISTSimpleDemoWithBatching()
    {
      Console.WriteLine("init CreateMNISTSimpleDemoWithBatching");
      var activation = Activation.ReLU;
      var net = new ConvNet(1, 28) { IsTraining=true };

      net.AddLayer(new ConvLayer(outputDepth: 16, windowSize: 5));
      net.AddLayer(new MaxPoolingLayer(windowSize: 3, stride: 2, activation: activation));
      net.AddLayer(new ConvLayer(outputDepth: 24, windowSize: 5));
      net.AddLayer(new MaxPoolingLayer(windowSize: 3, stride: 2, activation: activation));
      net.AddLayer(new FlattenLayer(outputDim: 10, activation: activation));

      net._Build();

      net.RandomizeParameters(seed: 0);

      var lrate = 0.0001D;
      var alg = new BackpropAlgorithm(net)
      {
        EpochCount = 50,
        LearningRate = lrate,
        BatchSize = 8,
        UseBatchParallelization = true,
        MaxBatchThreadCount = 8,
        LossFunction = Loss.Euclidean,
        Optimizer = Optimizer.RMSProp,
        LearningRateScheduler = LearningRateScheduler.DropBased(lrate, 5, 0.5D)
      };

      alg.Build();

      return alg;
    }

    public static BackpropAlgorithm CreateMNISTHardDemo()
    {
      Console.WriteLine("init CreateMNISTHardDemo");
      var activation = Activation.ReLU;
      var net = new ConvNet(1, 28) { IsTraining=true };

      net.AddLayer(new ConvLayer(outputDepth: 32, windowSize: 3, padding: 1, activation: activation));
      net.AddLayer(new ConvLayer(outputDepth: 64, windowSize: 3, padding: 1, activation: activation));
      net.AddLayer(new MaxPoolingLayer(windowSize: 2, stride: 2));
      net.AddLayer(new DropoutLayer(0.25));
      net.AddLayer(new FlattenLayer(outputDim: 128, activation: activation));
      net.AddLayer(new DropoutLayer(0.5));
      net.AddLayer(new FlattenLayer(outputDim: 10, activation: Activation.Exp));

      net._Build();

      net.RandomizeParameters(seed: 0);

      var lrate = 0.005D;
      var alg = new BackpropAlgorithm(net)
      {
        EpochCount = 50,
        LearningRate = lrate,
        BatchSize = 8,
        UseBatchParallelization = true,
        MaxBatchThreadCount = 8,
        LossFunction = Loss.CrossEntropySoftMax,
        Optimizer = Optimizer.RMSProp,
        LearningRateScheduler = LearningRateScheduler.DropBased(lrate, 5, 0.5D)
      };

      alg.Build();

      return alg;
    }

    #endregion

    #region CIFAR10

    /// <summary>
    /// Creates CNN for CIFAR-10 training (from keras)
    /// </summary>
    public static BackpropAlgorithm CreateCIFAR10Demo1()
    {
      Console.WriteLine("init CreateCIFAR10Demo1");
      var activation = Activation.ReLU;
      var net = new ConvNet(3, 32) { IsTraining=true };

      net.AddLayer(new ConvLayer(outputDepth: 32, windowSize: 3, padding: 1, activation: activation));
      net.AddLayer(new ConvLayer(outputDepth: 32, windowSize: 3, padding: 1, activation: activation));
      net.AddLayer(new MaxPoolingLayer(windowSize: 2, stride: 2));
      net.AddLayer(new DropoutLayer(0.25));

      net.AddLayer(new ConvLayer(outputDepth: 64, windowSize: 3, padding: 1, activation: activation));
      net.AddLayer(new ConvLayer(outputDepth: 64, windowSize: 3, padding: 1, activation: activation));
      net.AddLayer(new MaxPoolingLayer(windowSize: 2, stride: 2));
      net.AddLayer(new DropoutLayer(0.25));

      net.AddLayer(new FlattenLayer(outputDim: 512, activation: Activation.ReLU));
      net.AddLayer(new DropoutLayer(0.5));
      net.AddLayer(new DenseLayer(outputDim: 10, activation: Activation.Logistic(1)));

      net._Build();

      net.RandomizeParameters(seed: 0);

      var lrate = 0.01D;
      var alg = new BackpropAlgorithm(net)
      {
        LossFunction = Loss.CrossEntropySoftMax,
        EpochCount = 50,
        LearningRate = lrate,
        BatchSize = 8,
        UseBatchParallelization = true,
        MaxBatchThreadCount = 8,
        Optimizer = Optimizer.Adadelta,
        LearningRateScheduler = LearningRateScheduler.DropBased(lrate, 5, 0.5D)
      };

      alg.Build();

      return alg;
    }

    // https://code.google.com/archive/p/cuda-convnet/   - CIFAR archtectures+errors

    /// <summary>
    /// Creates CNN for CIFAR-10 training (from https://cs.stanford.edu/people/karpathy/convnetjs/demo/cifar10.html)
    /// </summary>
    public static BackpropAlgorithm CreateCIFAR10Demo2()
    {
      Console.WriteLine("init CreateCIFAR10Demo2");
      var activation = Activation.LeakyReLU();
      var net = new ConvNet(3, 32) { IsTraining=true };

      net.AddLayer(new ConvLayer(outputDepth: 32, windowSize: 5, padding: 2, activation: activation));
      net.AddLayer(new MaxPoolingLayer(windowSize: 2, stride: 2));
      net.AddLayer(new ConvLayer(outputDepth: 40, windowSize: 5, padding: 2, activation: activation));
      net.AddLayer(new MaxPoolingLayer(windowSize: 2, stride: 2));
      net.AddLayer(new ConvLayer(outputDepth: 60, windowSize: 5, padding: 2, activation: activation));
      net.AddLayer(new MaxPoolingLayer(windowSize: 2, stride: 2));
      net.AddLayer(new FlattenLayer(outputDim: 1024, activation: activation));
      net.AddLayer(new DropoutLayer(0.5));
      net.AddLayer(new DenseLayer(outputDim: 1024, activation: activation));
      net.AddLayer(new DropoutLayer(0.25));
      net.AddLayer(new DenseLayer(outputDim: 10, activation: activation));

      net._Build();

      net.RandomizeParameters(seed: 0);

      var lrate = 0.05D;
      var alg = new BackpropAlgorithm(net)
      {
        LossFunction = Loss.Euclidean,
        EpochCount = 500,
        LearningRate = lrate,
        BatchSize = 8,
        UseBatchParallelization = true,
        MaxBatchThreadCount = 8,
        Optimizer = Optimizer.Adadelta,
        LearningRateScheduler = LearningRateScheduler.TimeBased(lrate, 0.005D)
      };

      alg.Build();

      return alg;
    }

    /// <summary>
    /// Creates CNN for CIFAR-10 training
    /// (from http://machinelearningmastery.com/object-recognition-convolutional-neural-networks-keras-deep-learning-library/)
    /// </summary>
    public static BackpropAlgorithm CreateCIFAR10Demo3()
    {
      Console.WriteLine("init CreateCIFAR10Demo3");
      var activation = Activation.ReLU;
      var net = new ConvNet(3, 32) { IsTraining=true };

      net.AddLayer(new ConvLayer(outputDepth: 32, windowSize: 3, padding: 1, activation: activation));
      net.AddLayer(new DropoutLayer(0.2D));
      net.AddLayer(new ConvLayer(outputDepth: 32, windowSize: 3, padding: 1, activation: activation));
      net.AddLayer(new MaxPoolingLayer(windowSize: 2, stride: 2));
      net.AddLayer(new FlattenLayer(outputDim: 512, activation: Activation.ReLU));
      net.AddLayer(new DropoutLayer(0.5));
      net.AddLayer(new DenseLayer(outputDim: 10, activation: Activation.Logistic(1)));

      net._Build();

      net.RandomizeParameters(seed: 0);

      var lrate = 0.01D;
      var alg = new BackpropAlgorithm(net)
      {
        LossFunction = Loss.CrossEntropySoftMax,
        EpochCount = 50,
        LearningRate = lrate,
        BatchSize = 8,
        UseBatchParallelization = true,
        MaxBatchThreadCount = 8,
        Optimizer = Optimizer.Momentum,
        LearningRateScheduler = LearningRateScheduler.TimeBased(lrate, 0.0005D)
      };

      alg.Build();

      return alg;
    }

    #endregion

    #region CIFAR10 2 classes truncated

    /// <summary>
    /// Error = 22.5 Epoch = 62
    /// </summary>
    public static BackpropAlgorithm CreateCIFAR10Trunc2ClassesDemo1()
    {
      Console.WriteLine("init CreateCIFAR10Trunc2ClassesDemo1");
      var activation = Activation.ReLU;
      var net = new ConvNet(3, 32) { IsTraining=true };

      net.AddLayer(new ConvLayer(outputDepth: 16, windowSize: 3, padding: 2, activation: activation));
      net.AddLayer(new MaxPoolingLayer(windowSize: 2, stride: 2));
      net.AddLayer(new ConvLayer(outputDepth: 40, windowSize: 3, padding: 2, activation: activation));
      net.AddLayer(new MaxPoolingLayer(windowSize: 2, stride: 2));
      net.AddLayer(new ConvLayer(outputDepth: 60, windowSize: 3, padding: 2, activation: activation));
      net.AddLayer(new MaxPoolingLayer(windowSize: 2, stride: 2));
      net.AddLayer(new FlattenLayer(outputDim: 256, activation: activation));
      net.AddLayer(new DropoutLayer(0.5));
      net.AddLayer(new DenseLayer(outputDim: 2, activation: activation));

      net._Build();

      net.RandomizeParameters(seed: 0);

      var lrate = 0.01D;
      var alg = new BackpropAlgorithm(net)
      {
        LossFunction = Loss.Euclidean,
        EpochCount = 500,
        LearningRate = lrate,
        BatchSize = 8,
        UseBatchParallelization = true,
        MaxBatchThreadCount = 8,
        Optimizer = Optimizer.Adadelta,
        //Regularizator = Regularizator.L1(0.01D),
        LearningRateScheduler = LearningRateScheduler.DropBased(lrate, 5, 0.5D)
      };

      alg.Build();

      return alg;
    }

    /// <summary>
    /// Error 21.65
    /// </summary>
    public static BackpropAlgorithm CreateCIFAR10Trunc2ClassesDemo2_SEALED()
    {
      Console.WriteLine("init CreateCIFAR10Trunc2ClassesDemo2");

      var activation = Activation.ReLU;
      var net = new ConvNet(3, 32) { IsTraining=true };

      net.AddLayer(new ConvLayer(outputDepth: 16, windowSize: 3, padding: 1, activation: activation));
      net.AddLayer(new ConvLayer(outputDepth: 16, windowSize: 3, padding: 1, activation: activation));
      net.AddLayer(new MaxPoolingLayer(windowSize: 3, stride: 2));
      net.AddLayer(new DropoutLayer(0.25));

      net.AddLayer(new ConvLayer(outputDepth: 32, windowSize: 3, padding: 1, activation: activation));
      net.AddLayer(new ConvLayer(outputDepth: 32, windowSize: 3, padding: 1, activation: activation));
      net.AddLayer(new MaxPoolingLayer(windowSize: 3, stride: 2));
      net.AddLayer(new DropoutLayer(0.25));

      net.AddLayer(new FlattenLayer(outputDim: 256, activation: activation));
      net.AddLayer(new DropoutLayer(0.5));
      net.AddLayer(new DenseLayer(outputDim: 2, activation: Activation.Exp));

      net._Build();

      net.RandomizeParameters(seed: 0);

      var lrate = 0.01D;
      var alg = new BackpropAlgorithm(net)
      {
        LossFunction = Loss.CrossEntropySoftMax,
        EpochCount = 500,
        LearningRate = lrate,
        BatchSize = 4,
        UseBatchParallelization = true,
        MaxBatchThreadCount = 8,
        Optimizer = Optimizer.Adadelta,
        Regularizator = Regularizator.L2(0.001D),
        LearningRateScheduler = LearningRateScheduler.DropBased(lrate, 5, 0.5D)
      };

      alg.Build();

      return alg;
    }

    #endregion

    #region KAGGLE Cat or Dog

    /// <summary>
    ///
    /// </summary>
    public static BackpropAlgorithm CreateKaggleCatOrDogDemo1()
    {
      Console.WriteLine("init CreateKaggleCatOrDogDemo1");

      var activation = Activation.ReLU;
      var net = new ConvNet(3, 32) { IsTraining=true };

      net.AddLayer(new ConvLayer(outputDepth: 16, windowSize: 3, padding: 1, activation: activation));
      net.AddLayer(new ConvLayer(outputDepth: 16, windowSize: 3, padding: 1, activation: activation));
      net.AddLayer(new MaxPoolingLayer(windowSize: 3, stride: 2));
      net.AddLayer(new DropoutLayer(0.25));

      net.AddLayer(new ConvLayer(outputDepth: 32, windowSize: 3, padding: 1, activation: activation));
      net.AddLayer(new MaxPoolingLayer(windowSize: 3, stride: 2));
      net.AddLayer(new DropoutLayer(0.25));

      net.AddLayer(new FlattenLayer(outputDim: 256, activation: activation));
      net.AddLayer(new DropoutLayer(0.5));
      net.AddLayer(new DenseLayer(outputDim: 2, activation: Activation.Exp));

      net._Build();

      net.RandomizeParameters(seed: 0);

      var lrate = 0.1D;
      var alg = new BackpropAlgorithm(net)
      {
        LossFunction = Loss.CrossEntropySoftMax,
        EpochCount = 500,
        LearningRate = lrate,
        BatchSize = 4,
        UseBatchParallelization = true,
        MaxBatchThreadCount = 8,
        Optimizer = Optimizer.Adadelta,
        Regularizator = Regularizator.L2(0.001D),
        LearningRateScheduler = LearningRateScheduler.DropBased(lrate, 5, 0.5D)
      };

      alg.Build();

      return alg;
    }

    /// <summary>
    /// Error: 19.1
    /// </summary>
    public static BackpropAlgorithm CreateKaggleCatOrDogDemo_Pretrained()
    {
      Console.WriteLine("init CreateKaggleCatOrDogDemo_Pretrained");

      ConvNet net;
      var assembly = Assembly.GetExecutingAssembly();
      using (var stream = assembly.GetManifestResourceStream("ML.DeepTests.Pretrained.cat-dog-19.1.mld"))
      {
        net = ConvNet.Deserialize(stream);
        net.IsTraining = true;
      }

      var lrate = 0.01D;
      var alg = new BackpropAlgorithm(net)
      {
        LossFunction = Loss.CrossEntropySoftMax,
        EpochCount = 500,
        LearningRate = lrate,
        BatchSize = 4,
        UseBatchParallelization = true,
        MaxBatchThreadCount = 8,
        Optimizer = Optimizer.Adadelta,
        Regularizator = Regularizator.L2(0.001D),
        LearningRateScheduler = LearningRateScheduler.DropBased(lrate, 5, 0.5D)
      };

      alg.Build();

      return alg;
    }

    #endregion
  }
}
