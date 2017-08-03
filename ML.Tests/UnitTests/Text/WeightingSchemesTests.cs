using System;
using System.Collections.Generic;
using Microsoft.VisualStudio.TestTools.UnitTesting;
using ML.TextMethods.WeightingSchemes;

namespace ML.Tests.UnitTests.Text
{
  [TestClass]
  public class WeightingSchemesTests : TestBase
  {
    [ClassInitialize]
    public static void ClassInit(TestContext context)
    {
      BaseClassInit(context);
    }

    #region TF weighting schemes

    [TestMethod]
    public void BinaryTFWeightingScheme_GetFrequency()
    {
      // arrange
      var scheme = new BinaryTFWeightingScheme();
      var freqs = new double[] { 5, 2, 0, 1, 4, 0, 0, 1 };

      // act
      var f1 = scheme.GetFrequency(freqs, 0);
      var f2 = scheme.GetFrequency(freqs, 1);
      var f3 = scheme.GetFrequency(freqs, 2);
      var f4 = scheme.GetFrequency(freqs, 3);
      var f5 = scheme.GetFrequency(freqs, 4);
      var f6 = scheme.GetFrequency(freqs, 5);
      var f7 = scheme.GetFrequency(freqs, 6);
      var f8 = scheme.GetFrequency(freqs, 7);

      //assert
      Assert.AreEqual(1, f1);
      Assert.AreEqual(1, f2);
      Assert.AreEqual(0, f3);
      Assert.AreEqual(1, f4);
      Assert.AreEqual(1, f5);
      Assert.AreEqual(0, f6);
      Assert.AreEqual(0, f7);
      Assert.AreEqual(1, f8);
    }

    [TestMethod]
    public void RawCountTFWeightingScheme_GetFrequency()
    {
      // arrange
      var scheme = new RawCountTFWeightingScheme();
      var freqs = new double[] { 5, 2, 0, 1, 4, 0, 0, 1 };

      // act
      var f1 = scheme.GetFrequency(freqs, 0);
      var f2 = scheme.GetFrequency(freqs, 1);
      var f3 = scheme.GetFrequency(freqs, 2);
      var f4 = scheme.GetFrequency(freqs, 3);
      var f5 = scheme.GetFrequency(freqs, 4);
      var f6 = scheme.GetFrequency(freqs, 5);
      var f7 = scheme.GetFrequency(freqs, 6);
      var f8 = scheme.GetFrequency(freqs, 7);

      //assert
      Assert.AreEqual(5, f1);
      Assert.AreEqual(2, f2);
      Assert.AreEqual(0, f3);
      Assert.AreEqual(1, f4);
      Assert.AreEqual(4, f5);
      Assert.AreEqual(0, f6);
      Assert.AreEqual(0, f7);
      Assert.AreEqual(1, f8);
    }

    [TestMethod]
    public void TermFrequencyTFWeightingScheme_GetFrequency()
    {
      // arrange
      var scheme = new TermFrequencyTFWeightingScheme();
      var freqs = new double[] { 5, 2, 0, 1, 4, 0, 0, 1 };

      // act
      var f1 = scheme.GetFrequency(freqs, 0);
      var f2 = scheme.GetFrequency(freqs, 1);
      var f3 = scheme.GetFrequency(freqs, 2);
      var f4 = scheme.GetFrequency(freqs, 3);
      var f5 = scheme.GetFrequency(freqs, 4);
      var f6 = scheme.GetFrequency(freqs, 5);
      var f7 = scheme.GetFrequency(freqs, 6);
      var f8 = scheme.GetFrequency(freqs, 7);

      //assert
      Assert.AreEqual(5.0/13, f1, EPS);
      Assert.AreEqual(2.0/13, f2, EPS);
      Assert.AreEqual(     0, f3, EPS);
      Assert.AreEqual(1.0/13, f4, EPS);
      Assert.AreEqual(4.0/13, f5, EPS);
      Assert.AreEqual(     0, f6, EPS);
      Assert.AreEqual(     0, f7, EPS);
      Assert.AreEqual(1.0/13, f8, EPS);
    }

    [TestMethod]
    public void LogNormalizationTFWeightingScheme_GetFrequency()
    {
      // arrange
      var scheme = new LogNormalizationTFWeightingScheme();
      var freqs = new double[] { 5, 2, 0, 1, 4, 0, 0, 1 };

      // act
      var f1 = scheme.GetFrequency(freqs, 0);
      var f2 = scheme.GetFrequency(freqs, 1);
      var f3 = scheme.GetFrequency(freqs, 2);
      var f4 = scheme.GetFrequency(freqs, 3);
      var f5 = scheme.GetFrequency(freqs, 4);
      var f6 = scheme.GetFrequency(freqs, 5);
      var f7 = scheme.GetFrequency(freqs, 6);
      var f8 = scheme.GetFrequency(freqs, 7);

      //assert
      Assert.AreEqual(2.60943791243D, f1, EPS);
      Assert.AreEqual(1.69314718056D, f2, EPS);
      Assert.AreEqual(0, f3, EPS);
      Assert.AreEqual(1, f4, EPS);
      Assert.AreEqual(2.38629436112D, f5, EPS);
      Assert.AreEqual(0, f6, EPS);
      Assert.AreEqual(0, f7, EPS);
      Assert.AreEqual(1, f8, EPS);
    }

    [TestMethod]
    public void DoubleNormalizationTFWeightingScheme_GetFrequency()
    {
      // arrange
      var scheme = new DoubleNormalizationTFWeightingScheme(0.7);
      var freqs = new double[] { 5, 2, 0, 1, 4, 0, 0, 1 };

      // act
      var f1 = scheme.GetFrequency(freqs, 0);
      var f2 = scheme.GetFrequency(freqs, 1);
      var f3 = scheme.GetFrequency(freqs, 2);
      var f4 = scheme.GetFrequency(freqs, 3);
      var f5 = scheme.GetFrequency(freqs, 4);
      var f6 = scheme.GetFrequency(freqs, 5);
      var f7 = scheme.GetFrequency(freqs, 6);
      var f8 = scheme.GetFrequency(freqs, 7);

      //assert
      Assert.AreEqual(   1, f1, EPS);
      Assert.AreEqual(0.82, f2, EPS);
      Assert.AreEqual( 0.7, f3, EPS);
      Assert.AreEqual(0.76, f4, EPS);
      Assert.AreEqual(0.94, f5, EPS);
      Assert.AreEqual( 0.7, f6, EPS);
      Assert.AreEqual( 0.7, f7, EPS);
      Assert.AreEqual(0.76, f8, EPS);
    }

    #endregion


    #region IDF weighting schemes

    [TestMethod]
    public void UnaryIDFWeightingScheme_GetWeights()
    {
      // arrange
      var scheme = new UnaryIDFWeightingScheme();
      var v = 10;
      var freqs = new int[] { 3, 1, 1, 5, 2, 8 };

      // act
      var result = scheme.GetWeights(v, freqs);

      // assert
      Assert.AreEqual(6, result.Length);
      Assert.AreEqual(1, result[0]);
      Assert.AreEqual(1, result[1]);
      Assert.AreEqual(1, result[2]);
      Assert.AreEqual(1, result[3]);
      Assert.AreEqual(1, result[4]);
      Assert.AreEqual(1, result[5]);
    }

    [TestMethod]
    public void StandartIDFWeightingScheme_GetWeights()
    {
      // arrange
      var scheme = new StandartIDFWeightingScheme();
      var v = 10;
      var freqs = new int[] { 3, 1, 1, 5, 2, 8 };

      // act
      var result = scheme.GetWeights(v, freqs);

      // assert
      Assert.AreEqual(6, result.Length);
      Assert.AreEqual(1.20397280433D, result[0], EPS);
      Assert.AreEqual(2.30258509299D, result[1], EPS);
      Assert.AreEqual(2.30258509299D, result[2], EPS);
      Assert.AreEqual(0.69314718056D, result[3], EPS);
      Assert.AreEqual(1.60943791243D, result[4], EPS);
      Assert.AreEqual(0.22314355131D, result[5], EPS);
    }

    [TestMethod]
    public void MaxIDFWeightingScheme_GetWeights()
    {
      // arrange
      var scheme = new MaxIDFWeightingScheme();
      var v = 10;
      var freqs = new int[] { 3, 1, 1, 5, 2, 8 };

      // act
      var result = scheme.GetWeights(v, freqs);

      // assert
      Assert.AreEqual(6, result.Length);
      Assert.AreEqual( 0.69314718056D, result[0], EPS);
      Assert.AreEqual( 1.38629436112D, result[1], EPS);
      Assert.AreEqual( 1.38629436112D, result[2], EPS);
      Assert.AreEqual( 0.28768207245D, result[3], EPS);
      Assert.AreEqual( 0.98082925301D, result[4], EPS);
      Assert.AreEqual(-0.11778303565D, result[5], EPS);
    }

    [TestMethod]
    public void SmoothIDFWeightingScheme_GetWeights()
    {
      // arrange
      var scheme = new SmoothIDFWeightingScheme();
      var v = 10;
      var freqs = new int[] { 3, 1, 1, 5, 2, 8 };

      // act
      var result = scheme.GetWeights(v, freqs);

      // assert
      Assert.AreEqual(6, result.Length);
      Assert.AreEqual(1.46633706879D, result[0], EPS);
      Assert.AreEqual( 2.3978952728D, result[1], EPS);
      Assert.AreEqual( 2.3978952728D, result[2], EPS);
      Assert.AreEqual(1.09861228867D, result[3], EPS);
      Assert.AreEqual(1.79175946923D, result[4], EPS);
      Assert.AreEqual(0.81093021621D, result[5], EPS);
    }

    [TestMethod]
    public void ProbabilisticIDFWeightingScheme_GetWeights()
    {
      // arrange
      var scheme = new ProbabilisticIDFWeightingScheme();
      var v = 10;
      var freqs = new int[] { 3, 1, 1, 5, 2, 8 };

      // act
      var result = scheme.GetWeights(v, freqs);

      // assert
      Assert.AreEqual(6, result.Length);
      Assert.AreEqual( 0.84729786038D, result[0], EPS);
      Assert.AreEqual( 2.19722457734D, result[1], EPS);
      Assert.AreEqual( 2.19722457734D, result[2], EPS);
      Assert.AreEqual(              0, result[3], EPS);
      Assert.AreEqual( 1.38629436112D, result[4], EPS);
      Assert.AreEqual(-1.38629436112D, result[5], EPS);
    }

    #endregion
  }
}
