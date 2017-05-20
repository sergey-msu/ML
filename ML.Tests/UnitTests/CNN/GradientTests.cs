using System;
using System.Linq;
using Microsoft.VisualStudio.TestTools.UnitTesting;
using ML.Core;
using ML.DeepMethods.Algorithms;
using ML.DeepMethods.Registry;
using ML.Core.Registry;
using ML.DeepMethods.Models;

namespace ML.Tests.UnitTests.CNN
{
  [TestClass]
  public class GradientTests : TestBase
  {
     private readonly Class[]    CLASSES  = new Class[] { new Class("a", 0), new Class("b", 1) };
     private readonly double[][] EXPECTED = new double[2][] { new[] { 1.0D, 0.0D }, new[] { 0.0D, 1.0D } };

    #region Iter

    [TestMethod]
    public void Gradient_1ConvLayer_1Iter_Euclidean()
    {
      // arrange

      var net = new ConvNet(3, 1, 1) { IsTraining=true };
      net.AddLayer(new ConvLayer(outputDepth: 2, windowSize: 1, activation: Activation.Atan));
      net._Build();
      net.RandomizeParameters(seed: 0);

      var point1 = RandomPoint(3,1,1);
      var point2 = RandomPoint(3,1,1); // just for 2 dim output
      var sample = new ClassifiedSample<double[][,]>();
      sample[point1] = CLASSES[0];
      sample[point2] = CLASSES[1];

      var alg = new BackpropAlgorithm(net)
      {
        LearningRate = 0.1D,
        LossFunction = Loss.Euclidean
      };
      alg.Build();

      // act
      alg.RunIteration(point1, EXPECTED[0]);

      // assert
      AssertNetGradient(alg, point1, EXPECTED[0]);
    }

    [TestMethod]
    public void Gradient_SimpleDropout_1Iter_Euclidean()
    {
      // arrange

      var net = new ConvNet(3, 1) { IsTraining=true };
      net.AddLayer(new DenseLayer(outputDim: 10, activation: Activation.Atan));
      net.AddLayer(new DropoutLayer(rate: 0.5D));
      net.AddLayer(new DenseLayer(outputDim: 2, activation: Activation.Atan));
      net._Build();
      net.RandomizeParameters(seed: 0);

      var point1 = RandomPoint(3, 1, 1);
      var point2 = RandomPoint(3, 1, 1); // just for 2 dim output
      var sample = new ClassifiedSample<double[][,]>();
      sample[point1] = CLASSES[0];
      sample[point2] = CLASSES[1];

      var alg = new BackpropAlgorithm(net)
      {
        LearningRate = 0.1D,
        LossFunction = Loss.Euclidean
      };
      alg.Build();

      // act
      alg.RunIteration(point1, EXPECTED[0]);
      ((DropoutLayer)alg.Net[1]).ApplyCustomMask=true;

      // assert
      AssertNetGradient(alg, point1, EXPECTED[0]);
    }

    [TestMethod]
    public void Gradient_2ConvLayer1Flatten_1Iter_Euclidean()
    {
      // arrange

      var net = new ConvNet(3, 2, 2) { IsTraining=true };
      net.AddLayer(new ConvLayer(outputDepth: 3, windowSize: 2, padding: 1, activation: Activation.Tanh));
      net.AddLayer(new ConvLayer(outputDepth: 2, windowSize: 2, padding: 0, activation: Activation.Exp));
      net.AddLayer(new FlattenLayer(outputDim: 2, activation: Activation.Logistic(1)));
      net._Build();
      net.RandomizeParameters(seed: 0);

      var point1 = RandomPoint(3,2,2);
      var point2 = RandomPoint(3,2,2); // just for 2 dim output
      var sample = new ClassifiedSample<double[][,]>();
      sample[point1] = CLASSES[0];
      sample[point2] = CLASSES[1];

      var alg = new BackpropAlgorithm(net)
      {
        LearningRate = 0.1D,
        LossFunction = Loss.Euclidean
      };
      alg.Build();

      // act
      alg.RunIteration(point1, EXPECTED[0]);

      // assert
      AssertNetGradient(alg, point1, EXPECTED[0]);
    }

    [TestMethod]
    public void Gradient_MNISTSimple_1Iter()
    {
      // arrange

      var activation = Activation.ReLU;
      var net = new ConvNet(1, 14) { IsTraining=true };

      net.AddLayer(new ConvLayer(outputDepth: 4, windowSize: 5));
      net.AddLayer(new MaxPoolingLayer(windowSize: 2, stride: 2, activation: activation));
      net.AddLayer(new ConvLayer(outputDepth: 8, windowSize: 5));
      net.AddLayer(new MaxPoolingLayer(windowSize: 2, stride: 2, activation: activation));
      net.AddLayer(new FlattenLayer(outputDim: 10, activation: activation));

      net._Build();

      Randomize(net.Weights, -1.0D, 1.0D);

      var sample = new ClassifiedSample<double[][,]>();
      for (int i=0; i<10; i++)
      {
        var point = RandomPoint(1,14,14);
        sample[point] = new Class(i.ToString(), i);
      }

      var alg = new BackpropAlgorithm(net)
      {
        LearningRate = 0.005D,
        LossFunction = Loss.Euclidean
      };
      alg.Build();

      // act
      var data = sample.First();
      var expected = new double[10] { 1.0D, 0.0D, 0.0D, 0.0D, 0.0D, 0.0D, 0.0D, 0.0D, 0.0D, 0.0D };
      alg.RunIteration(data.Key, expected);

      // assert
      AssertNetGradient(alg, data.Key, expected);
    }

    [TestMethod]
    public void Gradient_DifferentLayers_1Iter_Euclidean()
    {
      // arrange

      var activation = Activation.ReLU;
      var net = new ConvNet(1, 5) { IsTraining=true };

      net.AddLayer(new ConvLayer(outputDepth: 2, windowSize: 3, padding: 1));
      net.AddLayer(new MaxPoolingLayer(windowSize: 3, stride: 2, activation: Activation.Exp));
      net.AddLayer(new ActivationLayer(activation: Activation.Tanh));
      net.AddLayer(new FlattenLayer(outputDim: 10, activation: activation));
      net.AddLayer(new DropoutLayer(rate: 0.5D));
      net.AddLayer(new DenseLayer(outputDim: 3, activation: activation));

      net._Build();

      net.RandomizeParameters(seed: 0);

      var sample = new ClassifiedSample<double[][,]>();
      for (int i=0; i<3; i++)
      {
        var point = RandomPoint(1,5,5);
        sample[point] = new Class(i.ToString(), i);
      }

      var alg = new BackpropAlgorithm(net)
      {
        LearningRate = 0.1D,
        LossFunction = Loss.Euclidean
      };
      alg.Build();

      // act
      var data = sample.First();
      var expected = new double[3] { 1.0D, 0.0D, 0.0D };
      alg.RunIteration(data.Key, expected);
      ((DropoutLayer)alg.Net[4]).ApplyCustomMask=true;

      // assert
      AssertNetGradient(alg, data.Key, expected);
    }

    [TestMethod]
    public void Gradient_DifferentLayers_1Iter_CrossEntropy_Regularization()
    {
      // arrange

      var activation = Activation.ReLU;
      var net = new ConvNet(1, 5) { IsTraining=true };

      net.AddLayer(new ConvLayer(outputDepth: 2, windowSize: 3, padding: 1));
      net.AddLayer(new MaxPoolingLayer(windowSize: 3, stride: 2, activation: Activation.Exp));
      net.AddLayer(new ActivationLayer(activation: Activation.Tanh));
      net.AddLayer(new FlattenLayer(outputDim: 10, activation: activation));
      net.AddLayer(new DropoutLayer(rate: 0.5D));
      net.AddLayer(new DenseLayer(outputDim: 3, activation: Activation.Exp));

      net._Build();

      net.RandomizeParameters(seed: 0);

      var sample = new ClassifiedSample<double[][,]>();
      for (int i=0; i<3; i++)
      {
        var point = RandomPoint(1,5,5);
        sample[point] = new Class(i.ToString(), i);
      }

      var regularizator = Regularizator.Composite(Regularizator.L1(0.1D), Regularizator.L2(0.3D));
      var alg = new BackpropAlgorithm(net)
      {
        LearningRate  = 0.1D,
        LossFunction  = Loss.CrossEntropySoftMax,
        Regularizator = regularizator
      };
      alg.Build();

      // act
      var data = sample.First();
      var expected = new double[3] { 1.0D, 0.0D, 0.0D };
      alg.RunIteration(data.Key, expected);
      regularizator.Apply(alg.Gradient, alg.Net.Weights);
      ((DropoutLayer)alg.Net[4]).ApplyCustomMask=true;

      // assert
      AssertNetGradient(alg, data.Key, expected);
    }

    #endregion

    #region Batch Synchronous

    #endregion

    #region .pvt

    private void AssertNetGradient(BackpropAlgorithm alg, double[][,] data, double[] expected)
    {
      var weights = alg.Net.Weights;

      for (int i=0; i<weights.Length; i++)
      {
        var w = weights[i];
        if (w==null) continue;
        var g = alg.Gradient[i];

        for (int j=0; j<w.Length; j++)
        {
          var prev = w[j];
          var actg = g[j];

          AssertDerivative(x =>
          {
            w[j] = x;
            var loss = alg.FeedForward(data, expected);
            w[j] = prev;
            return loss;
          }, prev, actg);
        }
      }
    }

    #endregion
  }
}
