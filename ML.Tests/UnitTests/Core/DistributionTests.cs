using System;
using Microsoft.VisualStudio.TestTools.UnitTesting;
using ML.Core.Kernels;
using ML.Core.Distributions;

namespace ML.Tests.UnitTests.Core
{
  [TestClass]
  public class DistributionTests : TestBase
  {
    [ClassInitialize]
    public static void ClassInit(TestContext context)
    {
      BaseClassInit(context);
    }

    [TestMethod]
    public void NormalDistribution_Value()
    {
      // arrange
      var distr = new NormalDistribution(theta: 1, sigma: 2);

      // act
      var v0 = distr.Value(0);
      var v1 = distr.Value(1);
      var v2 = distr.Value(2);

      // assert
      Assert.AreEqual(0.17603266338D, v0, EPS);
      Assert.AreEqual(0.1994711402,   v1, EPS);
      Assert.AreEqual(0.17603266338D, v2, EPS);
    }

    [TestMethod]
    public void NormalDistribution_MaximumLikelihood_FromSample()
    {
      // arrange
      var distr = new NormalDistribution(theta: 1, sigma: 2);
      var sample = new[] { -1.0D, 2.0D, 3.0D, 3.5D };

      // act
      distr.MaximumLikelihood(sample);

      // assert
      Assert.AreEqual(1.875D, distr.Theta);
      Assert.AreEqual(1.74553000547D, distr.Sigma, EPS);
    }

    [TestMethod]
    public void NormalDistribution_MaximumLikelihood_FromClassifiedSample()
    {
      // arrange
      var distr = new NormalDistribution(theta: 1, sigma: 2);
      var sample = new[] { -1.0D, 2.0D, 3.0D, 3.5D };

      // act
      distr.MaximumLikelihood(sample);

      // assert
      Assert.AreEqual(1.875D, distr.Theta);
      Assert.AreEqual(1.74553000547D, distr.Sigma, EPS);
    }
  }
}
