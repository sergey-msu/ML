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
    #region Iter

    [TestMethod]
    public void Gradient_1ConvLayer_1Iter_Euclidean()
    {
      // arrange

      var net = new ConvNet(3, 1, 1) { IsTraining=true };
      net.AddLayer(new ConvLayer(outputDepth: 2, windowSize: 1, activation: Activation.Atan));
      net._Build();
      net.RandomizeParameters(seed: 0);

      var sample = new ClassifiedSample<double[][,]>();
      var point1 = RandomPoint(3,1,1);
      var point2 = RandomPoint(3,1,1); // just for 2 dim output
      var cls1 = new Class("a", 0);
      var cls2 = new Class("b", 1);
      sample[point1] = cls1;
      sample[point2] = cls2;

      var alg = new BackpropAlgorithm(sample, net)
      {
        LearningRate = 0.1D,
        LossFunction = Loss.Euclidean
      };
      alg.Build();

      // act
      alg.RunIteration(point1, cls1);

      // assert
      AssertNetGradient(alg, point1, cls1);
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

      var sample = new ClassifiedSample<double[][,]>();
      var point1 = RandomPoint(3, 1, 1);
      var point2 = RandomPoint(3, 1, 1); // just for 2 dim output
      var cls1 = new Class("a", 0);
      var cls2 = new Class("b", 1);
      sample[point1] = cls1;
      sample[point2] = cls2;

      var alg = new BackpropAlgorithm(sample, net)
      {
        LearningRate = 0.1D,
        LossFunction = Loss.Euclidean
      };
      alg.Build();

      // act
      alg.RunIteration(point1, cls1);
      ((DropoutLayer)alg.Net[1]).ApplyCustomMask=true;

      // assert
      AssertNetGradient(alg, point1, cls1);
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

      var sample = new ClassifiedSample<double[][,]>();
      var point1 = RandomPoint(3,2,2);
      var point2 = RandomPoint(3,2,2); // just for 2 dim output
      var cls1 = new Class("a", 0);
      var cls2 = new Class("b", 1);
      sample[point1] = cls1;
      sample[point2] = cls2;

      var alg = new BackpropAlgorithm(sample, net)
      {
        LearningRate = 0.1D,
        LossFunction = Loss.Euclidean
      };
      alg.Build();

      // act
      alg.RunIteration(point1, cls1);

      // assert
      AssertNetGradient(alg, point1, cls1);
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
        var cls   = new Class(i.ToString(), i);
        sample[point] = cls;
      }

      var alg = new BackpropAlgorithm(sample, net)
      {
        LearningRate = 0.005D,
        LossFunction = Loss.Euclidean
      };
      alg.Build();

      // act
      var data = sample.First();
      alg.RunIteration(data.Key, data.Value);

      // assert
      AssertNetGradient(alg, data.Key, data.Value);
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
        var cls   = new Class(i.ToString(), i);
        sample[point] = cls;
      }

      var alg = new BackpropAlgorithm(sample, net)
      {
        LearningRate = 0.1D,
        LossFunction = Loss.Euclidean
      };
      alg.Build();

      // act
      var data = sample.First();
      alg.RunIteration(data.Key, data.Value);
      ((DropoutLayer)alg.Net[4]).ApplyCustomMask=true;

      // assert
      AssertNetGradient(alg, data.Key, data.Value);
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
        var cls   = new Class(i.ToString(), i);
        sample[point] = cls;
      }

      var regularizator = Regularizator.Composite(Regularizator.L1(0.1D), Regularizator.L2(0.3D));
      var alg = new BackpropAlgorithm(sample, net)
      {
        LearningRate  = 0.1D,
        LossFunction  = Loss.CrossEntropySoftMax,
        Regularizator = regularizator
      };
      alg.Build();

      // act
      var data = sample.First();
      alg.RunIteration(data.Key, data.Value);
      regularizator.Apply(alg.Gradient, alg.Net.Weights);
      ((DropoutLayer)alg.Net[4]).ApplyCustomMask=true;

      // assert
      AssertNetGradient(alg, data.Key, data.Value);
    }

    #endregion

    #region Batch Synchronous

    #endregion

    #region .pvt

    private void AssertNetGradient(BackpropAlgorithm alg, double[][,] data, Class cls)
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
            var loss = alg.FeedForward(data, cls);
            w[j] = prev;
            return loss;
          }, prev, actg);
        }
      }
    }

    #endregion
  }
}
