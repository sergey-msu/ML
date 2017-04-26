using System;
using ML.NeuralMethods.Models;
using ML.DeepMethods.Models;
using ML.Contracts;
using ML.Core;
using ML.Core.Registry;
using ML.DeepMethods.Algorithms;
using ML.DeepMethods.Registry;

namespace ML.Utils
{
  public static class Examples
  {
    /// <summary>
    /// Error = 0.92
    /// </summary>
    public static BackpropAlgorithm CreateMNISTSimpleDemo(ClassifiedSample<double[][,]> training)
    {
      var activation = Activation.ReLU;
      var net = new ConvNet(1, 28) { IsTraining=true };

      net.AddLayer(new ConvLayer(outputDepth: 8, windowSize: 5));
      net.AddLayer(new MaxPoolingLayer(windowSize: 2, stride: 2, activation: activation));
      net.AddLayer(new ConvLayer(outputDepth: 18, windowSize: 5));
      net.AddLayer(new MaxPoolingLayer(windowSize: 2, stride: 2, activation: activation));
      net.AddLayer(new FlattenLayer(outputDim: 10, activation: activation));

      net._Build();

      net.RandomizeParameters(seed: 0);

      var lrate = 0.005D;
      var alg = new BackpropAlgorithm(training, net)
      {
        EpochCount = 15,
        LearningRate = lrate,
        BatchSize = 1,
        LossFunction = Loss.Euclidean,
        Optimizer = Optimizer.SGD,
        LearningRateScheduler = LearningRateScheduler.DropBased(lrate, 5, 0.5D)
      };

      return alg;
    }

    /// <summary>
    ///
    /// </summary>
    public static BackpropAlgorithm ____CreateMNISTSimpleDemo(ClassifiedSample<double[][,]> training)
    {
      var activation = Activation.ReLU;
      var net = new ConvNet(1, 28) { IsTraining=true };

      net.AddLayer(new ConvLayer(outputDepth: 8, windowSize: 5));
      net.AddLayer(new MaxPoolingLayer(windowSize: 2, stride: 2, activation: activation));
      net.AddLayer(new ConvLayer(outputDepth: 18, windowSize: 5));
      net.AddLayer(new MaxPoolingLayer(windowSize: 2, stride: 2, activation: activation));
      net.AddLayer(new FlattenLayer(outputDim: 10, activation: activation));

      net._Build();

      net.RandomizeParameters(seed: 0);

      var lrate = 0.0001D;
      var alg = new BackpropAlgorithm(training, net)
      {
        EpochCount = 50,
        LearningRate = lrate,
        BatchSize = 1,
        LossFunction = Loss.Euclidean,
        Optimizer = Optimizer.Adam,
        LearningRateScheduler = LearningRateScheduler.DropBased(lrate, 5, 0.5D)
      };

      return alg;
    }

    public static BackpropAlgorithm CreateMNISTHardDemo(ClassifiedSample<double[][,]> training)
    {
      var activation = Activation.ReLU;
      var net = new ConvNet(1, 28) { IsTraining=true };

      net.AddLayer(new ConvLayer(outputDepth: 32, windowSize: 3, activation: activation));
      net.AddLayer(new ConvLayer(outputDepth: 64, windowSize: 3, activation: activation));
      net.AddLayer(new MaxPoolingLayer(windowSize: 2, stride: 2));
      net.AddLayer(new DropoutLayer(0.25));
      net.AddLayer(new FlattenLayer(outputDim: 128, activation: activation));
      net.AddLayer(new DropoutLayer(0.5));
      net.AddLayer(new FlattenLayer(outputDim: 10, activation: Activation.Logistic(1)));

      net._Build();

      net.RandomizeParameters(seed: 0);

      var lrate = 0.005D;
      var alg = new BackpropAlgorithm(training, net)
      {
        LossFunction = Loss.Euclidean,
        EpochCount = 50,
        LearningRate = lrate,
        BatchSize = 1,
        LearningRateScheduler = LearningRateScheduler.Constant(lrate)
      };

      return alg;
    }

    /// <summary>
    /// Creates CNN for CIFAR-10 training
    /// </summary>
    public static BackpropAlgorithm CreateCIFAR10Demo(ClassifiedSample<double[][,]> training)
    {
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

      net.Build();

      net.RandomizeParameters(seed: 0);

      var lrate = 0.005D;
      var alg = new BackpropAlgorithm(training, net)
      {
        LossFunction = Loss.CrossEntropySoftMax,
        EpochCount = 50,
        LearningRate = lrate,
        BatchSize = 1,
        LearningRateScheduler = LearningRateScheduler.Constant(lrate)
      };

      return alg;
    }

  }
}
