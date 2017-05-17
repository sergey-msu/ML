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
    public static BackpropAlgorithm CreateMNISTSimpleDemo_SEALED(ClassifiedSample<double[][,]> training)
    {
      Console.WriteLine("init CreateMNISTSimpleDemo_SEALED");
      var activation = Activation.LeakyReLU();
      var net = new ConvNet(1, 28) { IsTraining=true };

      net.AddLayer(new ConvLayer(outputDepth: 12, windowSize: 5, padding: 2));
      net.AddLayer(new MaxPoolingLayer(windowSize: 2, stride: 2, activation: activation));
      net.AddLayer(new ConvLayer(outputDepth: 24, windowSize: 5, padding: 2));
      net.AddLayer(new MaxPoolingLayer(windowSize: 2, stride: 2, activation: activation));
      net.AddLayer(new FlattenLayer(outputDim: 64, activation: activation));
      net.AddLayer(new DropoutLayer(rate: 0.5D));
      net.AddLayer(new DenseLayer(outputDim: 10, activation: activation));

      net._Build();

      net.RandomizeParameters(seed: 0);

      var lrate = 0.001D;
      var alg = new BackpropAlgorithm(training, net)
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

      return alg;
    }

    /// <summary>
    ///
    /// </summary>
    public static BackpropAlgorithm CreateMNISTSimpleDemoWithBatching(ClassifiedSample<double[][,]> training)
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
      var alg = new BackpropAlgorithm(training, net)
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

      return alg;
    }

    public static BackpropAlgorithm CreateMNISTHardDemo(ClassifiedSample<double[][,]> training)
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
      var alg = new BackpropAlgorithm(training, net)
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

      return alg;
    }

    #endregion

    #region CIFAR10

    /// <summary>
    /// Creates CNN for CIFAR-10 training (from keras)
    /// </summary>
    public static BackpropAlgorithm CreateCIFAR10Demo1(ClassifiedSample<double[][,]> training)
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
      var alg = new BackpropAlgorithm(training, net)
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

      return alg;
    }

    // https://code.google.com/archive/p/cuda-convnet/   - CIFAR archtectures+errors

    /// <summary>
    /// Creates CNN for CIFAR-10 training (from https://cs.stanford.edu/people/karpathy/convnetjs/demo/cifar10.html)
    /// </summary>
    public static BackpropAlgorithm CreateCIFAR10Demo2(ClassifiedSample<double[][,]> training)
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
      var alg = new BackpropAlgorithm(training, net)
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

      return alg;
    }

    /// <summary>
    /// Creates CNN for CIFAR-10 training
    /// (from http://machinelearningmastery.com/object-recognition-convolutional-neural-networks-keras-deep-learning-library/)
    /// </summary>
    public static BackpropAlgorithm CreateCIFAR10Demo3(ClassifiedSample<double[][,]> training)
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
      var alg = new BackpropAlgorithm(training, net)
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

      return alg;
    }

    #endregion

    #region CIFAR10 2 classes truncated

    /// <summary>
    /// Error = 22.5 Epoch = 62
    /// </summary>
    public static BackpropAlgorithm CreateCIFAR10Trunc2ClassesDemo1(ClassifiedSample<double[][,]> training)
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
      var alg = new BackpropAlgorithm(training, net)
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

      return alg;
    }

    /// <summary>
    /// Error 21.65
    /// </summary>
    public static BackpropAlgorithm CreateCIFAR10Trunc2ClassesDemo2_SEALED(ClassifiedSample<double[][,]> training)
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
      var alg = new BackpropAlgorithm(training, net)
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

      return alg;
    }

    #endregion

    #region KAGGLE Cat or Dog

    /// <summary>
    ///
    /// </summary>
    public static BackpropAlgorithm CreateKaggleCatOrDogDemo1(ClassifiedSample<double[][,]> training)
    {
      Console.WriteLine("init CreateKaggleCatOrDogDemo1");

      var activation = Activation.ReLU;
      var net = new ConvNet(3, 32) { IsTraining=true };

      net.AddLayer(new ConvLayer(outputDepth: 16, windowSize: 3, padding: 1, activation: activation));
      net.AddLayer(new ConvLayer(outputDepth: 16, windowSize: 3, padding: 1, activation: activation));
      net.AddLayer(new MaxPoolingLayer(windowSize: 3, stride: 2));
      net.AddLayer(new DropoutLayer(0.25));

      net.AddLayer(new ConvLayer(outputDepth: 32, windowSize: 3, padding: 1, activation: activation));
      //net.AddLayer(new ConvLayer(outputDepth: 32, windowSize: 3, padding: 1, activation: activation));
      net.AddLayer(new MaxPoolingLayer(windowSize: 3, stride: 2));
      net.AddLayer(new DropoutLayer(0.25));

      net.AddLayer(new FlattenLayer(outputDim: 256, activation: activation));
      net.AddLayer(new DropoutLayer(0.5));
      net.AddLayer(new DenseLayer(outputDim: 2, activation: Activation.Exp));

      net._Build();

      net.RandomizeParameters(seed: 0);

      var lrate = 0.1D;
      var alg = new BackpropAlgorithm(training, net)
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

      return alg;
    }

    /// <summary>
    /// Error: 19.1
    /// </summary>
    public static BackpropAlgorithm CreateKaggleCatOrDogDemo_Pretrained(ClassifiedSample<double[][,]> training)
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
      var alg = new BackpropAlgorithm(training, net)
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

      return alg;
    }

    /// <summary>
    /// Error: from 19.1 to
    /// </summary>
    public static BackpropAlgorithm CreateKaggleCatOrDogDemo_Pretrained_LiftTo64Size(ClassifiedSample<double[][,]> training)
    {
      Console.WriteLine("init CreateKaggleCatOrDogDemo_Pretrained_LiftTo64Size");

      // pretrained
      ConvNet pnet;
      var assembly = Assembly.GetExecutingAssembly();
      using (var stream = assembly.GetManifestResourceStream("ML.DeepTests.Pretrained.cat-dog-19.1.mld"))
      {
        pnet = ConvNet.Deserialize(stream);
        pnet.IsTraining = true;
      }

      // larger net
      var activation = Activation.ReLU;
      var net = new ConvNet(3, 64) { IsTraining=true };

      net.AddLayer(new ConvLayer(outputDepth: 16, windowSize: 5, padding: 2, activation: activation));
      net.AddLayer(new ConvLayer(outputDepth: 16, windowSize: 5, padding: 2, activation: activation));
      net.AddLayer(new MaxPoolingLayer(windowSize: 3, stride: 2));
      net.AddLayer(new DropoutLayer(0.25));

      net.AddLayer(new ConvLayer(outputDepth: 32, windowSize: 5, padding: 2, activation: activation));
      net.AddLayer(new ConvLayer(outputDepth: 32, windowSize: 5, padding: 2, activation: activation));
      net.AddLayer(new MaxPoolingLayer(windowSize: 3, stride: 2));
      net.AddLayer(new DropoutLayer(0.25));

      net.AddLayer(new FlattenLayer(outputDim: 256, activation: activation));
      net.AddLayer(new DropoutLayer(0.5));
      net.AddLayer(new DenseLayer(outputDim: 2, activation: Activation.Exp));

      net._Build();

      // convert weights

      var pl1 = (ConvLayer)pnet[0];
      var l1  = (ConvLayer)net[0];
      for (int q=0; q<l1.OutputDepth; q++)
      {
        l1.SetBias(q, pl1.GetBias(q));
        for (int p=0; p<l1.InputDepth; p++)
        for (int x=0; x<5; x++)
        for (int y=0; y<5; y++)
        {
          var pw = pl1.GetKernel(q, p, y/2, x/2);
          l1.SetKernel(q, p, y, x, pw);
        }
      }

      var pl2 = (ConvLayer)pnet[1];
      var l2  = (ConvLayer)net[1];
      for (int q=0; q<l2.OutputDepth; q++)
      {
        l2.SetBias(q, pl2.GetBias(q));
        for (int p=0; p<l2.InputDepth; p++)
        for (int x=0; x<5; x++)
        for (int y=0; y<5; y++)
        {
          var pw = pl2.GetKernel(q, p, y/2, x/2);
          l2.SetKernel(q, p, y, x, pw);
        }
      }

      var pl5 = (ConvLayer)pnet[4];
      var l5  = (ConvLayer)net[4];
      for (int q=0; q<l5.OutputDepth; q++)
      {
        l5.SetBias(q, pl5.GetBias(q));
        for (int p=0; p<l5.InputDepth; p++)
        for (int x=0; x<5; x++)
        for (int y=0; y<5; y++)
        {
          var pw = pl5.GetKernel(q, p, y/2, x/2);
          l5.SetKernel(q, p, y, x, pw);
        }
      }

      var pl8 = (ConvLayer)pnet[4];
      var l8  = (ConvLayer)net[4];
      for (int q=0; q<l8.OutputDepth; q++)
      {
        l8.SetBias(q, pl8.GetBias(q));
        for (int p=0; p<l8.InputDepth; p++)
        for (int x=0; x<5; x++)
        for (int y=0; y<5; y++)
        {
          var pw = pl8.GetKernel(q, p, y/2, x/2);
          l8.SetKernel(q, p, y, x, pw);
        }
      }

      var pl9 = (FlattenLayer)pnet[8];
      var l9  = (FlattenLayer)net[8];
      for (int q=0; q<l9.OutputDepth; q++)
      {
        l9.SetBias(q, pl9.GetBias(q));
        for (int p=0; p<l9.InputDepth; p++)
        for (int x=0; x<15; x++)
        for (int y=0; y<15; y++)
        {
          var pw = pl9.GetKernel(q, p, Math.Min(y/2, 6), Math.Min(x/2, 6));
          l9.SetKernel(q, p, y, x, pw);
        }
      }

      // build algorithm
      var lrate = 0.01D;
      var alg = new BackpropAlgorithm(training, net)
      {
        LossFunction = Loss.CrossEntropySoftMax,
        EpochCount = 500,
        LearningRate = lrate,
        BatchSize = 8,
        UseBatchParallelization = true,
        MaxBatchThreadCount = 8,
        Optimizer = Optimizer.Adadelta,
        Regularizator = Regularizator.L2(0.001D),
        LearningRateScheduler = LearningRateScheduler.DropBased(lrate, 5, 0.5D)
      };

      return alg;
    }

    #endregion
  }
}
