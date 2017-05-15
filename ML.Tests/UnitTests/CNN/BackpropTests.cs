using System;
using System.Linq;
using Microsoft.VisualStudio.TestTools.UnitTesting;
using ML.Core;
using ML.DeepMethods.Algorithms;
using ML.DeepMethods.Registry;

namespace ML.Tests.UnitTests.CNN
{
  [TestClass]
  public class BackpropTests : TestBase
  {
    [TestMethod]
    public void SimpleNet_Euclidean_OneIter()
    {
      // arrange

      var net = Mocks.SimpleLinearNetwork();

      var sample = new ClassifiedSample<double[][,]>();
      var point = new double[1][,] { new[,] { { 1.0D } } };
      var cls = new Class("a", 0);
      sample[point] = cls;

      var alg = new BackpropAlgorithm(sample, net);
      alg.LearningRate = 2.0D;
      alg.LossFunction = Loss.Euclidean;
      alg.Build();

      // act
      alg.RunIteration(point, cls);

      // assert

      Assert.AreEqual( 12, alg.Values[0][0][0,0]);
      Assert.AreEqual( 33, alg.Values[1][0][0,0]);
      Assert.AreEqual(-62, alg.Values[2][0][0,0]);

      Assert.AreEqual(3, net[0].ActivationFunction.DerivativeFromValue(alg.Values[0][0][0,0]));
      Assert.AreEqual(3, net[1].ActivationFunction.DerivativeFromValue(alg.Values[1][0][0,0]));
      Assert.AreEqual(2, net[2].ActivationFunction.DerivativeFromValue(alg.Values[2][0][0,0]));

      Assert.AreEqual(-126, alg.Errors[2][0][0,0]);
      Assert.AreEqual( 378, alg.Errors[1][0][0,0]);
      Assert.AreEqual(1134, alg.Errors[0][0][0,0]);

      Assert.AreEqual(-126*33, alg.Gradient[2][0]);
      Assert.AreEqual(-126,    alg.Gradient[2][1]);
      Assert.AreEqual(378*12,  alg.Gradient[1][0]);
      Assert.AreEqual(378,     alg.Gradient[1][1]);
      Assert.AreEqual(1134*1,  alg.Gradient[0][0]);
      Assert.AreEqual(1134,    alg.Gradient[0][1]);

      alg.FlushGradient();

      Assert.AreEqual(-1 + 2*126*33,    net[2].Weights[0]);
      Assert.AreEqual( 2 + 2*126,       net[2].Weights[1]);
      Assert.AreEqual( 1 + 2*(-378*12), net[1].Weights[0]);
      Assert.AreEqual(-1 + 2*(-378),    net[1].Weights[1]);
      Assert.AreEqual( 3 + 2*(-1134*1), net[0].Weights[0]);
      Assert.AreEqual( 1 + 2*(-1134),   net[0].Weights[1]);
    }

    [TestMethod]
    public void SimpleNet_OneIter_Dropout()
    {
      // arrange

      var drate = 0.5D;
      var dseed = 1;
      var net = Mocks.SimpleLinearNetworkWithDropout(drate, dseed);

      var sample = new ClassifiedSample<double[][,]>();
      var point = new double[1][,] { new[,] { { 1.0D } } };
      var cls = new Class("a", 0);
      sample[point] = cls;

      var alg = new BackpropAlgorithm(sample, net);
      alg.LearningRate = 2.0D;
      alg.LossFunction = Loss.Euclidean;
      alg.Build();

      // act
      alg.RunIteration(point, cls);

      // assert

      Assert.AreEqual(  12, alg.Values[0][0][0,0]);
      Assert.AreEqual(  33, alg.Values[1][0][0,0]);
      Assert.AreEqual(  66, alg.Values[2][0][0,0]);
      Assert.AreEqual(-128, alg.Values[3][0][0,0]);

      Assert.AreEqual(3, net[0].ActivationFunction.DerivativeFromValue(alg.Values[0][0][0,0]));
      Assert.AreEqual(3, net[1].ActivationFunction.DerivativeFromValue(alg.Values[1][0][0,0]));
      Assert.AreEqual(2, net[3].ActivationFunction.DerivativeFromValue(alg.Values[3][0][0,0]));

      Assert.AreEqual(-129*2,      alg.Errors[3][0][0,0]);
      Assert.AreEqual(-258*(-1),   alg.Errors[2][0][0,0]);
      Assert.AreEqual(258*3/drate, alg.Errors[1][0][0,0]);
      Assert.AreEqual(1548*3,      alg.Errors[0][0][0,0]);

      Assert.AreEqual(-258*66, alg.Gradient[3][0]);
      Assert.AreEqual(-258,    alg.Gradient[3][1]);
      Assert.AreEqual(0,       alg.Gradient[2].Length);
      Assert.AreEqual(0,       alg.Gradient[2].Length);
      Assert.AreEqual(1548*12, alg.Gradient[1][0]);
      Assert.AreEqual(1548,    alg.Gradient[1][1]);
      Assert.AreEqual(4644*1,  alg.Gradient[0][0]);
      Assert.AreEqual(4644,    alg.Gradient[0][1]);

      // act
      alg.FlushGradient();

      // assert

      Assert.AreEqual( 2 + 2*258,        net[3].Weights[1]);
      Assert.AreEqual(-1 + 2*258*66,     net[3].Weights[0]);
      Assert.AreEqual(-1 + 2*(-1548),    net[1].Weights[1]);
      Assert.AreEqual( 1 + 2*(-1548*12), net[1].Weights[0]);
      Assert.AreEqual( 1 + 2*(-4644),    net[0].Weights[1]);
      Assert.AreEqual( 3 + 2*(-4644*1),  net[0].Weights[0]);
    }
  }
}
