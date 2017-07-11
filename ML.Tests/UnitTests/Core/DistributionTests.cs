using System;
using Microsoft.VisualStudio.TestTools.UnitTesting;
using ML.Core.Distributions;
using ML.Core;

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
      var distr = new NormalDistribution(mu: 1, sigma: 2);

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
      var distr = new NormalDistribution();
      var sample = new[] { -1.0D, 2.0D, 3.0D, 3.5D };

      // act
      distr.MaximumLikelihood(sample);

      // assert
      Assert.AreEqual(1.875D, distr.Mu);
      Assert.AreEqual(1.74553000547D, distr.Sigma, EPS);
    }

    [TestMethod]
    public void NormalDistribution_MaximumLikelihood_FromClassifiedSample()
    {
      // arrange
      var distr = new NormalDistribution();
      var sample = new ClassifiedSample<double[]>
      {
        { new[] { -1.0D, 1.0D, 2.0D }, new Class("A", 1) },
        { new[] {  2.0D, 2.0D, 2.5D }, new Class("A", 1) },
        { new[] {  3.0D, 3.0D, 2.6D }, new Class("B", 2) },
        { new[] {  3.5D, 4.0D, 2.8D }, new Class("B", 2) },
      };

      // act
      var res = distr.MaximumLikelihood(sample);
      var dA1 = res[new ClassFeatureKey(new Class("A", 1), 0)];
      var dA2 = res[new ClassFeatureKey(new Class("A", 1), 1)];
      var dA3 = res[new ClassFeatureKey(new Class("A", 1), 2)];
      var dB1 = res[new ClassFeatureKey(new Class("B", 2), 0)];
      var dB2 = res[new ClassFeatureKey(new Class("B", 2), 1)];
      var dB3 = res[new ClassFeatureKey(new Class("B", 2), 2)];

      // assert
      Assert.AreEqual(0.5D,  dA1.Mu, EPS);
      Assert.AreEqual(1.5D,  dA1.Sigma, EPS);
      Assert.AreEqual(1.5D,  dA2.Mu, EPS);
      Assert.AreEqual(0.5D,  dA2.Sigma, EPS);
      Assert.AreEqual(2.25D, dA3.Mu, EPS);
      Assert.AreEqual(0.25D, dA3.Sigma, EPS);
      Assert.AreEqual(3.25D, dB1.Mu, EPS);
      Assert.AreEqual(0.25D, dB1.Sigma, EPS);
      Assert.AreEqual(3.5D,  dB2.Mu, EPS);
      Assert.AreEqual(0.5D,  dB2.Sigma, EPS);
      Assert.AreEqual(2.7D,  dB3.Mu, EPS);
      Assert.AreEqual(0.1D,  dB3.Sigma, EPS);
    }
  }
}
