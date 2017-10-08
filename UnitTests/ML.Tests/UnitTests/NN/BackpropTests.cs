using System;
using Microsoft.VisualStudio.TestTools.UnitTesting;
using ML.Core;
using ML.Registry;
using ML.NeuralMethods.Algorithms;
using ML.DeepMethods;
using ML.DeepMethods.Registry;

namespace ML.Tests.UnitTests.NN
{
  [TestClass]
  public class BackpropTests : TestBase
  {
    [TestMethod]
    public void SimpleNet_OneIter()
    {
      // arrange

      var net = Mocks.SimpleLinearNetwork();

      var sample = new ClassifiedSample<double[]>();
      var point = new[] { 1.0D };
      sample[point] = new Class("a", 0);

      var alg = new BackpropAlgorithm(net);
      alg.LearningRate = 2.0D;
      alg.LossFunction = Loss.Euclidean;

      // act
      alg.RunIteration(point, new double[] { 1.0D });

      // assert

      Assert.AreEqual( 12, net[0][0].Value);
      Assert.AreEqual( 33, net[1][0].Value);
      Assert.AreEqual(-62, net[2][0].Value);

      Assert.AreEqual(3, net[0][0].Derivative);
      Assert.AreEqual(3, net[1][0].Derivative);
      Assert.AreEqual(2, net[2][0].Derivative);

      Assert.AreEqual(-126, alg.Errors[2][0]);
      Assert.AreEqual( 378, alg.Errors[1][0]);
      Assert.AreEqual(1134, alg.Errors[0][0]);

      Assert.AreEqual(-126*33, alg.Gradient[2][0,0]);
      AssertNetGradient(alg, point, 2, 0, 0);

      Assert.AreEqual(   -126, alg.Gradient[2][0,1]);
      AssertNetGradient(alg, point, 2, 0, 1);

      Assert.AreEqual( 378*12, alg.Gradient[1][0,0]);
      AssertNetGradient(alg, point, 1, 0, 0);

      Assert.AreEqual(    378, alg.Gradient[1][0,1]);
      AssertNetGradient(alg, point, 1, 0, 1);

      Assert.AreEqual( 1134*1, alg.Gradient[0][0,0]);
      AssertNetGradient(alg, point, 0, 0, 0);

      Assert.AreEqual(   1134, alg.Gradient[0][0,1]);
      AssertNetGradient(alg, point, 0, 0, 1);


      // act
      alg.FlushGradient();

      // assert

      Assert.AreEqual( 2 + 2*126,       net[2][0].Bias);
      Assert.AreEqual(-1 + 2*126*33,    net[2][0][0]);
      Assert.AreEqual(-1 + 2*(-378),    net[1][0].Bias);
      Assert.AreEqual( 1 + 2*(-378*12), net[1][0][0]);
      Assert.AreEqual( 1 + 2*(-1134),   net[0][0].Bias);
      Assert.AreEqual( 3 + 2*(-1134*1), net[0][0][0]);
    }

    [TestMethod]
    public void SimpleNet_OneIter_Dropout()
    {
      // arrange

      var drate = 0.5D;
      var dseed = 1;
      var net = Mocks.SimpleLinearNetworkWithDropout(drate, dseed);

      var sample = new ClassifiedSample<double[]>();
      var point = new[] { 1.0D };
      sample[point] = new Class("a", 0);

      var alg = new BackpropAlgorithm(net);
      alg.LearningRate = 2.0D;
      alg.LossFunction = Loss.Euclidean;

      // act
      alg.RunIteration(point, new double[] { 1.0D });

      // assert

      Assert.AreEqual( 12,  net[0][0].Value);
      Assert.AreEqual( 66,  net[1][0].Value);
      Assert.AreEqual(-128, net[2][0].Value);

      Assert.AreEqual(3,       net[0][0].Derivative);
      Assert.AreEqual(3/drate, net[1][0].Derivative);
      Assert.AreEqual(2,       net[2][0].Derivative);

      Assert.AreEqual(           -129*2, alg.Errors[2][0]);
      Assert.AreEqual(-1*(-258)*3/drate, alg.Errors[1][0]);
      Assert.AreEqual(           1548*3, alg.Errors[0][0]);

      Assert.AreEqual(-258*66, alg.Gradient[2][0,0]);
      AssertNetGradient(alg, point, 2, 0, 0);

      Assert.AreEqual(-258, alg.Gradient[2][0,1]);
      AssertNetGradient(alg, point, 2, 0, 1);

      Assert.AreEqual(1548*12, alg.Gradient[1][0,0]);
      AssertNetGradient(alg, point, 1, 0, 0);

      Assert.AreEqual(1548, alg.Gradient[1][0,1]);
      AssertNetGradient(alg, point, 1, 0, 1);

      Assert.AreEqual(4644*1, alg.Gradient[0][0,0]);
      AssertNetGradient(alg, point, 0, 0, 0);

      Assert.AreEqual(4644, alg.Gradient[0][0,1]);
      AssertNetGradient(alg, point, 0, 0, 1);


      // act
      alg.FlushGradient();

      // assert

      Assert.AreEqual( 2 + 2*258,        net[2][0].Bias);
      Assert.AreEqual(-1 + 2*258*66,     net[2][0][0]);
      Assert.AreEqual(-1 + 2*(-1548),    net[1][0].Bias);
      Assert.AreEqual( 1 + 2*(-1548*12), net[1][0][0]);
      Assert.AreEqual( 1 + 2*(-4644),    net[0][0].Bias);
      Assert.AreEqual( 3 + 2*(-4644*1),  net[0][0][0]);
    }


    #region .pvt

    private void AssertNetGradient(BackpropAlgorithm alg, double[] point, int lidx, int nidx, int widx)
    {
      var net  = alg.Net;
      var loss = alg.LossFunction;
      var prev = net[lidx][nidx][widx];
      var grad = alg.Gradient[lidx][nidx, widx];

      AssertDerivative(x =>
      {
        net[lidx][nidx][widx] = x;
        var res = net.Calculate(point)[0];
        net[lidx][nidx][widx] = prev;
        return loss.Value(new[] { res }, new[] { 1.0D });
      }, prev, grad);
    }

    #endregion
  }
}
