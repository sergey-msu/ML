using System;
using Microsoft.VisualStudio.TestTools.UnitTesting;
using ML;
using ML.Core;
using ML.BayesianMethods.Algorithms;
using ML.Core.Kernels;

namespace ML.Tests.UnitTests.Bayesian
{
  [TestClass]
  public class BayesianTests : TestBase
  {
    [TestMethod]
    public void NaiveBayesianAlgorithm_Predict()
    {
      // arrange
      var kernel = new TriangularKernel();
      var alg = new NaiveBayesianKernelAlgorithm(kernel, 0.3D);
      var sample = new ClassifiedSample<double[]>
      {
        { new[] { 0.2, 0.2 }, new Class("A", 0) },
        { new[] { 0.4, 0.6 }, new Class("A", 0) },
        { new[] { 0.6, 0.4 }, new Class("A", 0) },
        { new[] { 0.8, 0.6 }, new Class("B", 1) },
        { new[] { 0.8, 0.8 }, new Class("B", 1) }
      };

      // act
      alg.Train(sample);
      var res1 = alg.Predict(new[] { 0.4, 0.4 });
      var res2 = alg.Predict(new[] { 0.6, 0.6 });
      var res3 = alg.Predict(new[] { 0.9, 0.7 });

      // assert
      Assert.AreEqual(new Class("A", 0), res1);
      Assert.AreEqual(new Class("A", 0), res2);
      Assert.AreEqual(new Class("B", 1), res3);
    }

    [TestMethod]
    public void NaiveBayesianAlgorithm_CalculateClassScore()
    {
      // arrange
      var kernel = new TriangularKernel();
      var alg = new NaiveBayesianKernelAlgorithm(kernel, 2.0D);
      var sample = new ClassifiedSample<double[]>
      {
        { new[] { 2.0, 1.0 }, new Class("A", 0) },
        { new[] { 0.0, 3.0 }, new Class("A", 0) },
        { new[] { 4.0, 3.0 }, new Class("B", 1) }
      };

      // act
      alg.Train(sample);
      var s11 = alg.CalculateClassScore(new[] { 1.0, 2.0 }, new Class("A", 0));
      var s12 = alg.CalculateClassScore(new[] { 1.0, 2.0 }, new Class("B", 1));
      var s21 = alg.CalculateClassScore(new[] { 2.0, 2.0 }, new Class("A", 0));
      var s22 = alg.CalculateClassScore(new[] { 2.0, 2.0 }, new Class("B", 1));
      var s31 = alg.CalculateClassScore(new[] { 3.0, 2.0 }, new Class("A", 0));
      var s32 = alg.CalculateClassScore(new[] { 3.0, 2.0 }, new Class("B", 1));

      // assert
      Assert.AreEqual(Math.Log(1/24.0D), s11, EPS);
      Assert.AreEqual(double.NegativeInfinity, s12);
      Assert.AreEqual(Math.Log(1/24.0D), s21, EPS);
      Assert.AreEqual(double.NegativeInfinity, s22);
      Assert.AreEqual(Math.Log(1/48.0D), s31, EPS);
      Assert.AreEqual(Math.Log(1/48.0D), s32, EPS);
    }
  }
}
