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

    #region Normal Distribution

    [TestMethod]
    public void NormalDistribution_Value()
    {
      // arrange
      var distr = new NormalDistribution();
      distr.Params = new NormalDistribution.Parameters(1, 2);

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
      distr.FromSample(sample);
      var pars = distr.Params;

      // assert
      Assert.AreEqual(1.875D, pars.Mu);
      Assert.AreEqual(1.74553000547D, pars.Sigma, EPS);
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
      var res = distr.FromSample(sample);
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

    #endregion

    #region Bernoulli Distribution

    [TestMethod]
    public void BernoulliDistribution_Value()
    {
      // arrange
      var distr = new BernoulliDistribution();
      distr.Params = new BernoulliDistribution.Parameters(0.3D);

      // act
      var v0 = distr.Value(0);
      var v1 = distr.Value(1);

      // assert
      Assert.AreEqual(0.7D, v0);
      Assert.AreEqual(0.3D, v1);
    }

    [TestMethod]
    public void BernoulliDistribution_MaximumLikelihood_FromSample()
    {
      // arrange
      var distr = new BernoulliDistribution();
      var sample = new[] { 1.0D, 1.0D, 0.0D, 1.0D };

      // act
      distr.FromSample(sample);
      var pars = distr.Params;

      // assert
      Assert.AreEqual(0.75D, pars.P);
    }

    [TestMethod]
    public void BernoulliDistribution_MaximumLikelihood_FromClassifiedSample()
    {
      // arrange
      var distr = new BernoulliDistribution();
      var sample = new ClassifiedSample<double[]>
      {
        { new[] { 1.0D, 1.0D, 0.0D }, new Class("A", 1) },
        { new[] { 1.0D, 0.0D, 1.0D }, new Class("A", 1) },
        { new[] { 0.0D, 0.0D, 1.0D }, new Class("B", 2) },
        { new[] { 0.0D, 0.0D, 0.0D }, new Class("B", 2) },
      };

      // act
      var res = distr.FromSample(sample);
      var dA1 = res[new ClassFeatureKey(new Class("A", 1), 0)];
      var dA2 = res[new ClassFeatureKey(new Class("A", 1), 1)];
      var dA3 = res[new ClassFeatureKey(new Class("A", 1), 2)];
      var dB1 = res[new ClassFeatureKey(new Class("B", 2), 0)];
      var dB2 = res[new ClassFeatureKey(new Class("B", 2), 1)];
      var dB3 = res[new ClassFeatureKey(new Class("B", 2), 2)];

      // assert
      Assert.AreEqual(1.0D, dA1.P, EPS);
      Assert.AreEqual(0.5D, dA2.P, EPS);
      Assert.AreEqual(0.5D, dA3.P, EPS);
      Assert.AreEqual(0.0D, dB1.P, EPS);
      Assert.AreEqual(0.0D, dB2.P, EPS);
      Assert.AreEqual(0.5D, dB3.P, EPS);
    }

    #endregion

    #region Multinomial Part Distribution

    [TestMethod]
    public void MultinomialPartDistribution_Value()
    {
      // arrange
      var distr = new MultinomialPartDistribution();
      distr.Params = new MultinomialPartDistribution.Parameters(0.3D);

      // act
      var v0 = distr.Value(0);
      var v1 = distr.Value(1);
      var v2 = distr.Value(2);

      // assert
      Assert.AreEqual( 1.0D, v0);
      Assert.AreEqual( 0.3D, v1);
      Assert.AreEqual(0.09D, v2);
    }

    [TestMethod]
    public void MultinomialPartDistribution_MaximumLikelihood_FromSample()
    {
      // arrange
      var distr = new MultinomialPartDistribution { TotalCount = 10 };
      var sample = new[] { 2.0D, 1.0D, 0.0D, 3.0D };

      // act
      distr.FromSample(sample);
      var pars = distr.Params;

      // assert
      Assert.AreEqual(0.60D, pars.P, EPS);
    }

    [TestMethod]
    public void MultinomialPartDistribution_MaximumLikelihood_FromSample_UseSmoothing()
    {
      // arrange
      var distr = new MultinomialPartDistribution { N = 10, UseSmoothing=true, Alpha=2, TotalCount=80 };
      var sample = new[] { 0.0D, 0.0D, 0.0D, 0.0D };

      // act
      distr.FromSample(sample);
      var pars = distr.Params;

      // assert
      Assert.AreEqual(0.02D, pars.P, EPS);
    }

    [TestMethod]
    public void MultinomialPartDistribution_MaximumLikelihood_FromClassifiedSample()
    {
      // arrange
      var sample = new ClassifiedSample<double[]>
      {
        { new[] { 1.0D, 2.0D, 0.0D }, new Class("A", 1) },
        { new[] { 3.0D, 0.0D, 2.0D }, new Class("A", 1) },
        { new[] { 0.0D, 3.0D, 1.0D }, new Class("B", 2) },
        { new[] { 0.0D, 2.0D, 0.0D }, new Class("B", 2) },
        { new[] { 0.0D, 2.0D, 2.0D }, new Class("B", 2) },
      };
      var n = 3; // sample[i].Key.Length - the length of the word dictionary
      var distr = new MultinomialPartDistribution { N = n };

      // act
      var res = distr.FromSample(sample);
      var dA1 = res[new ClassFeatureKey(new Class("A", 1), 0)];
      var dA2 = res[new ClassFeatureKey(new Class("A", 1), 1)];
      var dA3 = res[new ClassFeatureKey(new Class("A", 1), 2)];
      var dB1 = res[new ClassFeatureKey(new Class("B", 2), 0)];
      var dB2 = res[new ClassFeatureKey(new Class("B", 2), 1)];
      var dB3 = res[new ClassFeatureKey(new Class("B", 2), 2)];

      // assert
      Assert.AreEqual( 0.5D, dA1.P, EPS);
      Assert.AreEqual(0.25D, dA2.P, EPS);
      Assert.AreEqual(0.25D, dA3.P, EPS);
      Assert.AreEqual( 0.0D, dB1.P, EPS);
      Assert.AreEqual( 0.7D, dB2.P, EPS);
      Assert.AreEqual( 0.3D, dB3.P, EPS);
    }

    [TestMethod]
    public void MultinomialPartDistribution_MaximumLikelihood_FromClassifiedSample_UseSmoothing()
    {
      // arrange
      var sample = new ClassifiedSample<double[]>
      {
        { new[] { 1.0D, 2.0D, 0.0D }, new Class("A", 1) },
        { new[] { 3.0D, 0.0D, 2.0D }, new Class("A", 1) },
        { new[] { 0.0D, 3.0D, 1.0D }, new Class("B", 2) },
        { new[] { 0.0D, 2.0D, 0.0D }, new Class("B", 2) },
        { new[] { 0.0D, 2.0D, 2.0D }, new Class("B", 2) },
      };
      var n = 3; // sample[i].Key.Length - the length of the word dictionary
      var distr = new MultinomialPartDistribution { N = n, UseSmoothing=true, Alpha=2 };

      // act
      var res = distr.FromSample(sample);
      var dA1 = res[new ClassFeatureKey(new Class("A", 1), 0)];
      var dA2 = res[new ClassFeatureKey(new Class("A", 1), 1)];
      var dA3 = res[new ClassFeatureKey(new Class("A", 1), 2)];
      var dB1 = res[new ClassFeatureKey(new Class("B", 2), 0)];
      var dB2 = res[new ClassFeatureKey(new Class("B", 2), 1)];
      var dB3 = res[new ClassFeatureKey(new Class("B", 2), 2)];

      // assert
      Assert.AreEqual(  0.5D, dA1.P, EPS);
      Assert.AreEqual( 0.25D, dA2.P, EPS);
      Assert.AreEqual( 0.25D, dA3.P, EPS);
      Assert.AreEqual(0.125D, dB1.P, EPS);
      Assert.AreEqual(  0.7D, dB2.P, EPS);
      Assert.AreEqual(  0.3D, dB3.P, EPS);
    }

    #endregion
  }
}
