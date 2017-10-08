using System;
using System.Linq;
using System.Collections.Generic;
using Microsoft.VisualStudio.TestTools.UnitTesting;
using ML.TextMethods.Algorithms;
using ML.TextMethods.Preprocessing;
using ML.TextMethods.Tokenization;
using ML.TextMethods.Stopwords;
using ML.TextMethods.Normalization;
using ML.TextMethods.Stemming;
using ML.Core;
using ML.Core.Distributions;
using ML.Contracts;
using ML.Core.Kernels;
using ML.BayesianMethods.Algorithms;

namespace ML.Tests.UnitTests.Text
{
  [TestClass]
  public class AlgorithmsTests : TestBase
  {
    [ClassInitialize]
    public static void ClassInit(TestContext context)
    {
      BaseClassInit(context);
    }

    #region BinaryNaiveBayesianAlgorithm

    [TestMethod]
    public void BinaryNaiveBayesianAlgorithm_Train()
    {
      // arrange
      var sample = getSample();
      var CLS1 = sample.CachedClasses.ElementAt(0);
      var CLS2 = sample.CachedClasses.ElementAt(1);
      var prep = getDefaultPreprocessor();
      var alg = new BinaryNaiveBayesianAlgorithm() { Preprocessor = prep };

      // act
      alg.Train(sample);

      // assert
      Assert.AreEqual(sample, alg.TrainingSample);
      Assert.AreEqual(prep, alg.Preprocessor);
      Assert.AreEqual(2, alg.ClassHist.Length);
      Assert.AreEqual(4, alg.ClassHist[0]);
      Assert.AreEqual(2, alg.ClassHist[1]);
      Assert.AreEqual(6, alg.DataCount);
      Assert.AreEqual(8, alg.DataDim);

      Assert.AreEqual(8, alg.Vocabulary.Count);
      Assert.AreEqual("cat",      alg.Vocabulary[0]);
      Assert.AreEqual("like",     alg.Vocabulary[1]);
      Assert.AreEqual("icecream", alg.Vocabulary[2]);
      Assert.AreEqual("at",       alg.Vocabulary[3]);
      Assert.AreEqual("dog",      alg.Vocabulary[4]);
      Assert.AreEqual("meet",     alg.Vocabulary[5]);
      Assert.AreEqual("world",    alg.Vocabulary[6]);
      Assert.AreEqual("seven",    alg.Vocabulary[7]);

      Assert.AreEqual(2, alg.PriorProbs.Length);
      Assert.AreEqual(Math.Log(4/6.0D), alg.PriorProbs[0], EPS);
      Assert.AreEqual(Math.Log(2/6.0D), alg.PriorProbs[1], EPS);

      Assert.AreEqual(2, alg.Weights.Length);
      Assert.AreEqual(8, alg.Weights[0].Length);
      Assert.AreEqual(8, alg.Weights[1].Length);
      Assert.AreEqual(5.0/12, alg.Weights[0][0]);
      Assert.AreEqual(2.0/12, alg.Weights[0][1]);
      Assert.AreEqual(2.0/12, alg.Weights[0][2]);
      Assert.AreEqual(2.0/12, alg.Weights[0][3]);
      Assert.AreEqual(2.0/12, alg.Weights[0][4]);
      Assert.AreEqual(3.0/12, alg.Weights[0][5]);
      Assert.AreEqual(2.0/12, alg.Weights[0][6]);
      Assert.AreEqual(1.0/12, alg.Weights[0][7]);

      Assert.AreEqual(0.1, alg.Weights[1][0]);
      Assert.AreEqual(0.2, alg.Weights[1][1]);
      Assert.AreEqual(0.1, alg.Weights[1][2]);
      Assert.AreEqual(0.1, alg.Weights[1][3]);
      Assert.AreEqual(0.3, alg.Weights[1][4]);
      Assert.AreEqual(0.1, alg.Weights[1][5]);
      Assert.AreEqual(0.2, alg.Weights[1][6]);
      Assert.AreEqual(0.2, alg.Weights[1][7]);
    }

    [TestMethod]
    public void BinaryNaiveBayesianAlgorithm_ExtractFeatureVector()
    {
      // arrange
      var sample = getSample();
      var CLS1 = sample.CachedClasses.ElementAt(0);
      var CLS2 = sample.CachedClasses.ElementAt(1);
      var prep = getDefaultPreprocessor();
      var alg = new BinaryNaiveBayesianAlgorithm() { Preprocessor = prep };
      bool isEmpty;

      // act
      alg.Train(sample);
      var result1 = alg.ExtractFeatureVector(testDocs()[0], out isEmpty);
      var result2 = alg.ExtractFeatureVector(testDocs()[1], out isEmpty);

      // assert
      Assert.AreEqual(8, result1.Length);
      Assert.AreEqual(1, result1[0]);
      Assert.AreEqual(0, result1[1]);
      Assert.AreEqual(1, result1[2]);
      Assert.AreEqual(0, result1[3]);
      Assert.AreEqual(0, result1[4]);
      Assert.AreEqual(0, result1[5]);
      Assert.AreEqual(0, result1[6]);
      Assert.AreEqual(0, result1[7]);

      Assert.AreEqual(8, result2.Length);
      Assert.AreEqual(1, result2[0]);
      Assert.AreEqual(1, result2[1]);
      Assert.AreEqual(1, result2[2]);
      Assert.AreEqual(0, result2[3]);
      Assert.AreEqual(1, result2[4]);
      Assert.AreEqual(0, result2[5]);
      Assert.AreEqual(1, result2[6]);
      Assert.AreEqual(1, result2[7]);
    }

    [TestMethod]
    public void BinaryNaiveBayesianAlgorithm_PredictTokens()
    {
      // arrange
      var sample = getSample();
      var CLS1 = sample.CachedClasses.ElementAt(0);
      var CLS2 = sample.CachedClasses.ElementAt(1);
      var prep = getDefaultPreprocessor();
      var alg = new BinaryNaiveBayesianAlgorithm() { Preprocessor = prep };

      // act
      alg.Train(sample);
      var result1 = alg.PredictTokens(testDocs()[0], 2);
      var result2 = alg.PredictTokens(testDocs()[1], 2);

      // assert
      Assert.AreEqual(2, result1.Length);
      Assert.AreEqual(CLS1, result1[0].Class);
      Assert.AreEqual(-4.17667299131D, result1[0].Score, EPS);
      Assert.AreEqual(CLS2, result1[1].Class);
      Assert.AreEqual(-6.94060910385D, result1[1].Score, EPS);
      Assert.AreEqual(CLS1, result2[0].Class);
      Assert.AreEqual(-11.4028820014D, result2[0].Score, EPS);
      Assert.AreEqual(CLS2, result2[1].Class);
      Assert.AreEqual(-11.9467900476D, result2[1].Score, EPS);
    }

    [TestMethod]
    public void BinaryNaiveBayesianAlgorithm_Predict()
    {
      // arrange
      var sample = getSample();
      var CLS1 = sample.CachedClasses.ElementAt(0);
      var CLS2 = sample.CachedClasses.ElementAt(1);
      var prep = getDefaultPreprocessor();
      var alg = new BinaryNaiveBayesianAlgorithm() { Preprocessor = prep };

      // act
      alg.Train(sample);
      var result1 = alg.Predict(testDocs()[0]);
      var result2 = alg.Predict(testDocs()[1]);
      var result3 = alg.Predict(testDocs()[2]);

      // assert
      Assert.AreEqual(CLS1, result1);
      Assert.AreEqual(CLS1, result2);
      Assert.AreEqual(CLS2, result3);
    }

    #endregion

    #region MultinomialNaiveBayesianAlgorithm

    [TestMethod]
    public void MultinomialNaiveBayesianAlgorithm_Train()
    {
      // arrange
      var sample = getSample();
      var CLS1 = sample.CachedClasses.ElementAt(0);
      var CLS2 = sample.CachedClasses.ElementAt(1);
      var prep = getDefaultPreprocessor();
      var alg = new MultinomialNaiveBayesianAlgorithm() { Preprocessor = prep };

      // act
      alg.Train(sample);

      // assert
      Assert.AreEqual(sample, alg.TrainingSample);
      Assert.AreEqual(prep, alg.Preprocessor);
      Assert.AreEqual(2, alg.ClassHist.Length);
      Assert.AreEqual(4, alg.ClassHist[0]);
      Assert.AreEqual(2, alg.ClassHist[1]);
      Assert.AreEqual(6, alg.DataCount);
      Assert.AreEqual(8, alg.DataDim);

      Assert.AreEqual(8, alg.Vocabulary.Count);
      Assert.AreEqual("cat",      alg.Vocabulary[0]);
      Assert.AreEqual("like",     alg.Vocabulary[1]);
      Assert.AreEqual("icecream", alg.Vocabulary[2]);
      Assert.AreEqual("at",       alg.Vocabulary[3]);
      Assert.AreEqual("dog",      alg.Vocabulary[4]);
      Assert.AreEqual("meet",     alg.Vocabulary[5]);
      Assert.AreEqual("world",    alg.Vocabulary[6]);
      Assert.AreEqual("seven",    alg.Vocabulary[7]);

      Assert.AreEqual(2, alg.PriorProbs.Length);
      Assert.AreEqual(Math.Log(4/6.0D), alg.PriorProbs[0], EPS);
      Assert.AreEqual(Math.Log(2/6.0D), alg.PriorProbs[1], EPS);

      Assert.AreEqual(2, alg.Weights.Length);
      Assert.AreEqual(8, alg.Weights[0].Length);
      Assert.AreEqual(8, alg.Weights[1].Length);
      Assert.AreEqual(Math.Log(7.0/21), alg.Weights[0][0], EPS);
      Assert.AreEqual(Math.Log(2.0/21), alg.Weights[0][1], EPS);
      Assert.AreEqual(Math.Log(2.0/21), alg.Weights[0][2], EPS);
      Assert.AreEqual(Math.Log(2.0/21), alg.Weights[0][3], EPS);
      Assert.AreEqual(Math.Log(2.0/21), alg.Weights[0][4], EPS);
      Assert.AreEqual(Math.Log(3.0/21), alg.Weights[0][5], EPS);
      Assert.AreEqual(Math.Log(2.0/21), alg.Weights[0][6], EPS);
      Assert.AreEqual(Math.Log(1.0/21), alg.Weights[0][7], EPS);

      Assert.AreEqual(Math.Log(1.0/14), alg.Weights[1][0], EPS);
      Assert.AreEqual(Math.Log(2.0/14), alg.Weights[1][1], EPS);
      Assert.AreEqual(Math.Log(1.0/14), alg.Weights[1][2], EPS);
      Assert.AreEqual(Math.Log(1.0/14), alg.Weights[1][3], EPS);
      Assert.AreEqual(Math.Log(4.0/14), alg.Weights[1][4], EPS);
      Assert.AreEqual(Math.Log(1.0/14), alg.Weights[1][5], EPS);
      Assert.AreEqual(Math.Log(2.0/14), alg.Weights[1][6], EPS);
      Assert.AreEqual(Math.Log(2.0/14), alg.Weights[1][7], EPS);
    }

    [TestMethod]
    public void MultinomialNaiveBayesianAlgorithm_ExtractFeatureVector()
    {
      // arrange
      var sample = getSample();
      var CLS1 = sample.CachedClasses.ElementAt(0);
      var CLS2 = sample.CachedClasses.ElementAt(1);
      var prep = getDefaultPreprocessor();
      var alg = new MultinomialNaiveBayesianAlgorithm() { Preprocessor = prep };
      bool isEmpty;

      // act
      alg.Train(sample);
      var result1 = alg.ExtractFeatureVector(testDocs()[0], out isEmpty);
      var result2 = alg.ExtractFeatureVector(testDocs()[1], out isEmpty);

      // assert
      Assert.AreEqual(8, result1.Length);
      Assert.AreEqual(1, result1[0]);
      Assert.AreEqual(0, result1[1]);
      Assert.AreEqual(1, result1[2]);
      Assert.AreEqual(0, result1[3]);
      Assert.AreEqual(0, result1[4]);
      Assert.AreEqual(0, result1[5]);
      Assert.AreEqual(0, result1[6]);
      Assert.AreEqual(0, result1[7]);

      Assert.AreEqual(8, result2.Length);
      Assert.AreEqual(1, result2[0]);
      Assert.AreEqual(2, result2[1]);
      Assert.AreEqual(1, result2[2]);
      Assert.AreEqual(0, result2[3]);
      Assert.AreEqual(2, result2[4]);
      Assert.AreEqual(0, result2[5]);
      Assert.AreEqual(1, result2[6]);
      Assert.AreEqual(1, result2[7]);
    }

    [TestMethod]
    public void MultinomialNaiveBayesianAlgorithm_PredictTokens()
    {
      // arrange
      var sample = getSample();
      var CLS1 = sample.CachedClasses.ElementAt(0);
      var CLS2 = sample.CachedClasses.ElementAt(1);
      var prep = getDefaultPreprocessor();
      var alg = new MultinomialNaiveBayesianAlgorithm() { Preprocessor = prep };

      // act
      alg.Train(sample);
      var result1 = alg.PredictTokens(testDocs()[0], 2);
      var result2 = alg.PredictTokens(testDocs()[1], 2);

      // assert
      Assert.AreEqual(2, result1.Length);
      Assert.AreEqual(CLS1, result1[0].Class);
      Assert.AreEqual(-3.85545265394D, result1[0].Score, EPS);
      Assert.AreEqual(CLS2, result1[1].Class);
      Assert.AreEqual(-6.3767269479D, result1[1].Score, EPS);
      Assert.AreEqual(CLS2, result2[0].Class);
      Assert.AreEqual(-16.6658934811D, result2[0].Score, EPS);
      Assert.AreEqual(CLS1, result2[1].Class);
      Assert.AreEqual(-18.6568513775D, result2[1].Score, EPS);
    }

    [TestMethod]
    public void MultinomialNaiveBayesianAlgorithm_Predict()
    {
      // arrange
      var sample = getSample();
      var CLS1 = sample.CachedClasses.ElementAt(0);
      var CLS2 = sample.CachedClasses.ElementAt(1);
      var prep = getDefaultPreprocessor();
      var alg = new MultinomialNaiveBayesianAlgorithm() { Preprocessor = prep };

      // act
      alg.Train(sample);
      var result1 = alg.Predict(testDocs()[0]);
      var result2 = alg.Predict(testDocs()[1]);
      var result3 = alg.Predict(testDocs()[2]);

      // assert
      Assert.AreEqual(CLS1, result1);
      Assert.AreEqual(CLS2, result2);
      Assert.AreEqual(CLS2, result3);
    }

    #endregion

    #region ComplementNaiveBayesianAlgorithm

    [TestMethod]
    public void ComplementNaiveBayesianAlgorithm_Train()
    {
      // arrange
      var sample = getSample();
      var CLS1 = sample.CachedClasses.ElementAt(0);
      var CLS2 = sample.CachedClasses.ElementAt(1);
      var prep = getDefaultPreprocessor();
      var alg = new ComplementNaiveBayesianAlgorithm() { Preprocessor = prep };

      // act
      alg.Train(sample);

      // assert
      Assert.AreEqual(sample, alg.TrainingSample);
      Assert.AreEqual(prep, alg.Preprocessor);
      Assert.AreEqual(2, alg.ClassHist.Length);
      Assert.AreEqual(4, alg.ClassHist[0]);
      Assert.AreEqual(2, alg.ClassHist[1]);
      Assert.AreEqual(6, alg.DataCount);
      Assert.AreEqual(8, alg.DataDim);

      Assert.AreEqual(8, alg.Vocabulary.Count);
      Assert.AreEqual("cat",      alg.Vocabulary[0]);
      Assert.AreEqual("like",     alg.Vocabulary[1]);
      Assert.AreEqual("icecream", alg.Vocabulary[2]);
      Assert.AreEqual("at",       alg.Vocabulary[3]);
      Assert.AreEqual("dog",      alg.Vocabulary[4]);
      Assert.AreEqual("meet",     alg.Vocabulary[5]);
      Assert.AreEqual("world",    alg.Vocabulary[6]);
      Assert.AreEqual("seven",    alg.Vocabulary[7]);

      Assert.AreEqual(2, alg.PriorProbs.Length);
      Assert.AreEqual(Math.Log(4/6.0D), alg.PriorProbs[0], EPS);
      Assert.AreEqual(Math.Log(2/6.0D), alg.PriorProbs[1], EPS);

      Assert.AreEqual(2, alg.Weights.Length);
      Assert.AreEqual(8, alg.Weights[0].Length);
      Assert.AreEqual(8, alg.Weights[1].Length);
      Assert.AreEqual(-Math.Log(1.0/14), alg.Weights[0][0], EPS);
      Assert.AreEqual(-Math.Log(2.0/14), alg.Weights[0][1], EPS);
      Assert.AreEqual(-Math.Log(1.0/14), alg.Weights[0][2], EPS);
      Assert.AreEqual(-Math.Log(1.0/14), alg.Weights[0][3], EPS);
      Assert.AreEqual(-Math.Log(4.0/14), alg.Weights[0][4], EPS);
      Assert.AreEqual(-Math.Log(1.0/14), alg.Weights[0][5], EPS);
      Assert.AreEqual(-Math.Log(2.0/14), alg.Weights[0][6], EPS);
      Assert.AreEqual(-Math.Log(2.0/14), alg.Weights[0][7], EPS);

      Assert.AreEqual(-Math.Log(7.0/21), alg.Weights[1][0], EPS);
      Assert.AreEqual(-Math.Log(2.0/21), alg.Weights[1][1], EPS);
      Assert.AreEqual(-Math.Log(2.0/21), alg.Weights[1][2], EPS);
      Assert.AreEqual(-Math.Log(2.0/21), alg.Weights[1][3], EPS);
      Assert.AreEqual(-Math.Log(2.0/21), alg.Weights[1][4], EPS);
      Assert.AreEqual(-Math.Log(3.0/21), alg.Weights[1][5], EPS);
      Assert.AreEqual(-Math.Log(2.0/21), alg.Weights[1][6], EPS);
      Assert.AreEqual(-Math.Log(1.0/21), alg.Weights[1][7], EPS);
    }

    [TestMethod]
    public void ComplementNaiveBayesianAlgorithm_ExtractFeatureVector()
    {
      // arrange
      var sample = getSample();
      var CLS1 = sample.CachedClasses.ElementAt(0);
      var CLS2 = sample.CachedClasses.ElementAt(1);
      var prep = getDefaultPreprocessor();
      var alg = new ComplementNaiveBayesianAlgorithm() { Preprocessor = prep };
      bool isEmpty;

      // act
      alg.Train(sample);
      var result1 = alg.ExtractFeatureVector(testDocs()[0], out isEmpty);
      var result2 = alg.ExtractFeatureVector(testDocs()[1], out isEmpty);

      // assert
      Assert.AreEqual(8, result1.Length);
      Assert.AreEqual(1, result1[0]);
      Assert.AreEqual(0, result1[1]);
      Assert.AreEqual(1, result1[2]);
      Assert.AreEqual(0, result1[3]);
      Assert.AreEqual(0, result1[4]);
      Assert.AreEqual(0, result1[5]);
      Assert.AreEqual(0, result1[6]);
      Assert.AreEqual(0, result1[7]);

      Assert.AreEqual(8, result2.Length);
      Assert.AreEqual(1, result2[0]);
      Assert.AreEqual(2, result2[1]);
      Assert.AreEqual(1, result2[2]);
      Assert.AreEqual(0, result2[3]);
      Assert.AreEqual(2, result2[4]);
      Assert.AreEqual(0, result2[5]);
      Assert.AreEqual(1, result2[6]);
      Assert.AreEqual(1, result2[7]);
    }

    [TestMethod]
    public void ComplementNaiveBayesianAlgorithm_PredictTokens()
    {
      // arrange
      var sample = getSample();
      var CLS1 = sample.CachedClasses.ElementAt(0);
      var CLS2 = sample.CachedClasses.ElementAt(1);
      var prep = getDefaultPreprocessor();
      var alg = new ComplementNaiveBayesianAlgorithm() { Preprocessor = prep };

      // act
      alg.Train(sample);
      var result1 = alg.PredictTokens(testDocs()[0], 2);
      var result2 = alg.PredictTokens(testDocs()[1], 2);

      // assert
      Assert.AreEqual(2, result1.Length);
      Assert.AreEqual(CLS1, result1[0].Class);
      Assert.AreEqual(4.8726495511D, result1[0].Score, EPS);
      Assert.AreEqual(CLS2, result1[1].Class);
      Assert.AreEqual(2.3513752571D, result1[1].Score, EPS);
      Assert.AreEqual(CLS2, result2[0].Class);
      Assert.AreEqual(17.152773981D, result2[0].Score, EPS);
      Assert.AreEqual(CLS1, result2[1].Class);
      Assert.AreEqual(15.161816084D, result2[1].Score, EPS);
    }

    [TestMethod]
    public void ComplementNaiveBayesianAlgorithm_Predict()
    {
      // arrange
      var sample = getSample();
      var CLS1 = sample.CachedClasses.ElementAt(0);
      var CLS2 = sample.CachedClasses.ElementAt(1);
      var prep = getDefaultPreprocessor();
      var alg = new ComplementNaiveBayesianAlgorithm() { Preprocessor = prep };

      // act
      alg.Train(sample);
      var result1 = alg.Predict(testDocs()[0]);
      var result2 = alg.Predict(testDocs()[1]);
      var result3 = alg.Predict(testDocs()[2]);

      // assert
      Assert.AreEqual(CLS1, result1);
      Assert.AreEqual(CLS2, result2);
      Assert.AreEqual(CLS2, result3);
    }

    #endregion

    #region ComplementOVANaiveBayesianAlgorithm

    [TestMethod]
    public void ComplementOVANaiveBayesianAlgorithm_Train()
    {
      // arrange
      var sample = getSample();
      var CLS1 = sample.CachedClasses.ElementAt(0);
      var CLS2 = sample.CachedClasses.ElementAt(1);
      var prep = getDefaultPreprocessor();
      var alg = new ComplementOVANaiveBayesianAlgorithm() { Preprocessor = prep };

      // act
      alg.Train(sample);

      // assert
      Assert.AreEqual(sample, alg.TrainingSample);
      Assert.AreEqual(prep, alg.Preprocessor);
      Assert.AreEqual(2, alg.ClassHist.Length);
      Assert.AreEqual(4, alg.ClassHist[0]);
      Assert.AreEqual(2, alg.ClassHist[1]);
      Assert.AreEqual(6, alg.DataCount);
      Assert.AreEqual(8, alg.DataDim);

      Assert.AreEqual(8, alg.Vocabulary.Count);
      Assert.AreEqual("cat",      alg.Vocabulary[0]);
      Assert.AreEqual("like",     alg.Vocabulary[1]);
      Assert.AreEqual("icecream", alg.Vocabulary[2]);
      Assert.AreEqual("at",       alg.Vocabulary[3]);
      Assert.AreEqual("dog",      alg.Vocabulary[4]);
      Assert.AreEqual("meet",     alg.Vocabulary[5]);
      Assert.AreEqual("world",    alg.Vocabulary[6]);
      Assert.AreEqual("seven",    alg.Vocabulary[7]);

      Assert.AreEqual(2, alg.PriorProbs.Length);
      Assert.AreEqual(Math.Log(4/6.0D), alg.PriorProbs[0], EPS);
      Assert.AreEqual(Math.Log(2/6.0D), alg.PriorProbs[1], EPS);

      Assert.AreEqual(2, alg.Weights.Length);
      Assert.AreEqual(8, alg.Weights[0].Length);
      Assert.AreEqual(8, alg.Weights[1].Length);
      Assert.AreEqual(Math.Log(7.0/21)-Math.Log(1.0/14), alg.Weights[0][0]);
      Assert.AreEqual(Math.Log(2.0/21)-Math.Log(2.0/14), alg.Weights[0][1]);
      Assert.AreEqual(Math.Log(2.0/21)-Math.Log(1.0/14), alg.Weights[0][2]);
      Assert.AreEqual(Math.Log(2.0/21)-Math.Log(1.0/14), alg.Weights[0][3]);
      Assert.AreEqual(Math.Log(2.0/21)-Math.Log(4.0/14), alg.Weights[0][4]);
      Assert.AreEqual(Math.Log(3.0/21)-Math.Log(1.0/14), alg.Weights[0][5]);
      Assert.AreEqual(Math.Log(2.0/21)-Math.Log(2.0/14), alg.Weights[0][6]);
      Assert.AreEqual(Math.Log(1.0/21)-Math.Log(2.0/14), alg.Weights[0][7]);

      Assert.AreEqual(Math.Log(1.0/14)-Math.Log(7.0/21), alg.Weights[1][0]);
      Assert.AreEqual(Math.Log(2.0/14)-Math.Log(2.0/21), alg.Weights[1][1]);
      Assert.AreEqual(Math.Log(1.0/14)-Math.Log(2.0/21), alg.Weights[1][2]);
      Assert.AreEqual(Math.Log(1.0/14)-Math.Log(2.0/21), alg.Weights[1][3]);
      Assert.AreEqual(Math.Log(4.0/14)-Math.Log(2.0/21), alg.Weights[1][4]);
      Assert.AreEqual(Math.Log(1.0/14)-Math.Log(3.0/21), alg.Weights[1][5]);
      Assert.AreEqual(Math.Log(2.0/14)-Math.Log(2.0/21), alg.Weights[1][6]);
      Assert.AreEqual(Math.Log(2.0/14)-Math.Log(1.0/21), alg.Weights[1][7]);
    }

    [TestMethod]
    public void ComplementOVANaiveBayesianAlgorithm_ExtractFeatureVector()
    {
      // arrange
      var sample = getSample();
      var CLS1 = sample.CachedClasses.ElementAt(0);
      var CLS2 = sample.CachedClasses.ElementAt(1);
      var prep = getDefaultPreprocessor();
      var alg = new ComplementOVANaiveBayesianAlgorithm() { Preprocessor = prep };
      bool isEmpty;

      // act
      alg.Train(sample);
      var result1 = alg.ExtractFeatureVector(testDocs()[0], out isEmpty);
      var result2 = alg.ExtractFeatureVector(testDocs()[1], out isEmpty);

      // assert
      Assert.AreEqual(8, result1.Length);
      Assert.AreEqual(1, result1[0]);
      Assert.AreEqual(0, result1[1]);
      Assert.AreEqual(1, result1[2]);
      Assert.AreEqual(0, result1[3]);
      Assert.AreEqual(0, result1[4]);
      Assert.AreEqual(0, result1[5]);
      Assert.AreEqual(0, result1[6]);
      Assert.AreEqual(0, result1[7]);

      Assert.AreEqual(8, result2.Length);
      Assert.AreEqual(1, result2[0]);
      Assert.AreEqual(2, result2[1]);
      Assert.AreEqual(1, result2[2]);
      Assert.AreEqual(0, result2[3]);
      Assert.AreEqual(2, result2[4]);
      Assert.AreEqual(0, result2[5]);
      Assert.AreEqual(1, result2[6]);
      Assert.AreEqual(1, result2[7]);
    }

    [TestMethod]
    public void ComplementOVANaiveBayesianAlgorithm_PredictTokens()
    {
      // arrange
      var sample = getSample();
      var CLS1 = sample.CachedClasses.ElementAt(0);
      var CLS2 = sample.CachedClasses.ElementAt(1);
      var prep = getDefaultPreprocessor();
      var alg = new ComplementOVANaiveBayesianAlgorithm() { Preprocessor = prep };

      // act
      alg.Train(sample);
      var result1 = alg.PredictTokens(testDocs()[0], 2);
      var result2 = alg.PredictTokens(testDocs()[1], 2);

      // assert
      Assert.AreEqual(2, result1.Length);
      Assert.AreEqual(CLS1, result1[0].Class);
      Assert.AreEqual(1.42266200529D, result1[0].Score, EPS);
      Assert.AreEqual(CLS2, result1[1].Class);
      Assert.AreEqual(-2.9267394021D, result1[1].Score, EPS);
      Assert.AreEqual(CLS2, result2[0].Class);
      Assert.AreEqual(1.58549278826D, result2[0].Score, EPS);
      Assert.AreEqual(CLS1, result2[1].Class);
      Assert.AreEqual(-3.0895701850D, result2[1].Score, EPS);
    }

    [TestMethod]
    public void ComplementOVANaiveBayesianAlgorithm_Predict()
    {
      // arrange
      var sample = getSample();
      var CLS1 = sample.CachedClasses.ElementAt(0);
      var CLS2 = sample.CachedClasses.ElementAt(1);
      var prep = getDefaultPreprocessor();
      var alg = new ComplementOVANaiveBayesianAlgorithm() { Preprocessor = prep };

      // act
      alg.Train(sample);
      var result1 = alg.Predict(testDocs()[0]);
      var result2 = alg.Predict(testDocs()[1]);
      var result3 = alg.Predict(testDocs()[2]);

      // assert
      Assert.AreEqual(CLS1, result1);
      Assert.AreEqual(CLS2, result2);
      Assert.AreEqual(CLS2, result3);
    }

    #endregion

    #region TFIDFNaiveBayesianAlgorithm

    [TestMethod]
    public void TFIDFNaiveBayesianAlgorithm_Train()
    {
      // arrange
      var sample = getSample();
      var CLS1 = sample.CachedClasses.ElementAt(0);
      var CLS2 = sample.CachedClasses.ElementAt(1);
      var prep = getDefaultPreprocessor();
      var alg = new TFIDFNaiveBayesianAlgorithm()
      {
        TFWeightingScheme  = Registry.TFWeightingScheme.LogNormalization,
        IDFWeightingScheme = Registry.IDFWeightingScheme.Standart,
        Preprocessor = prep
       };

      // act
      alg.Train(sample);

      // assert
      Assert.AreEqual(sample, alg.TrainingSample);
      Assert.AreEqual(prep, alg.Preprocessor);
      Assert.AreEqual(2, alg.ClassHist.Length);
      Assert.AreEqual(4, alg.ClassHist[0]);
      Assert.AreEqual(2, alg.ClassHist[1]);
      Assert.AreEqual(6, alg.DataCount);
      Assert.AreEqual(8, alg.DataDim);

      Assert.AreEqual(8, alg.Vocabulary.Count);
      Assert.AreEqual("cat",      alg.Vocabulary[0]);
      Assert.AreEqual("like",     alg.Vocabulary[1]);
      Assert.AreEqual("icecream", alg.Vocabulary[2]);
      Assert.AreEqual("at",       alg.Vocabulary[3]);
      Assert.AreEqual("dog",      alg.Vocabulary[4]);
      Assert.AreEqual("meet",     alg.Vocabulary[5]);
      Assert.AreEqual("world",    alg.Vocabulary[6]);
      Assert.AreEqual("seven",    alg.Vocabulary[7]);

      Assert.AreEqual(2, alg.PriorProbs.Length);
      Assert.AreEqual(Math.Log(4/6.0D), alg.PriorProbs[0], EPS);
      Assert.AreEqual(Math.Log(2/6.0D), alg.PriorProbs[1], EPS);

      Assert.AreEqual(2, alg.Weights.Length);
      Assert.AreEqual(8, alg.Weights[0].Length);
      Assert.AreEqual(8, alg.Weights[1].Length);
      Assert.AreEqual(-1.5552175827D, alg.Weights[0][0], EPS);
      Assert.AreEqual(-2.2401396740D, alg.Weights[0][1], EPS);
      Assert.AreEqual(-1.9851330973D, alg.Weights[0][2], EPS);
      Assert.AreEqual(-1.9851330973D, alg.Weights[0][3], EPS);
      Assert.AreEqual(-2.4263657885D, alg.Weights[0][4], EPS);
      Assert.AreEqual(-1.7821199307D, alg.Weights[0][5], EPS);
      Assert.AreEqual(-2.2401396740D, alg.Weights[0][6], EPS);
      Assert.AreEqual(-3.1098813602D, alg.Weights[0][7], EPS);

      Assert.AreEqual(-2.7404236664, alg.Weights[1][0], EPS);
      Assert.AreEqual(-1.8706819802, alg.Weights[1][1], EPS);
      Assert.AreEqual(-2.7404236664, alg.Weights[1][2], EPS);
      Assert.AreEqual(-2.7404236664, alg.Weights[1][3], EPS);
      Assert.AreEqual(-1.4480231657, alg.Weights[1][4], EPS);
      Assert.AreEqual(-2.7404236664, alg.Weights[1][5], EPS);
      Assert.AreEqual(-1.8706819802, alg.Weights[1][6], EPS);
      Assert.AreEqual(-1.6156754035, alg.Weights[1][7], EPS);
    }

    [TestMethod]
    public void TFIDFNaiveBayesianAlgorithm_ExtractFeatureVector()
    {
      // arrange
      var sample = getSample();
      var CLS1 = sample.CachedClasses.ElementAt(0);
      var CLS2 = sample.CachedClasses.ElementAt(1);
      var prep = getDefaultPreprocessor();
      var alg = new TFIDFNaiveBayesianAlgorithm()
      {
        TFWeightingScheme  = Registry.TFWeightingScheme.LogNormalization,
        IDFWeightingScheme = Registry.IDFWeightingScheme.Standart,
        Preprocessor = prep
       };
       bool isEmpty;

      // act
      alg.Train(sample);
      var result1 = alg.ExtractFeatureVector(testDocs()[0], out isEmpty);
      var result2 = alg.ExtractFeatureVector(testDocs()[1], out isEmpty);

      // assert
      Assert.AreEqual(8, result1.Length);
      Assert.AreEqual(1, result1[0]);
      Assert.AreEqual(0, result1[1]);
      Assert.AreEqual(1, result1[2]);
      Assert.AreEqual(0, result1[3]);
      Assert.AreEqual(0, result1[4]);
      Assert.AreEqual(0, result1[5]);
      Assert.AreEqual(0, result1[6]);
      Assert.AreEqual(0, result1[7]);

      Assert.AreEqual(8, result2.Length);
      Assert.AreEqual(1, result2[0]);
      Assert.AreEqual(2, result2[1]);
      Assert.AreEqual(1, result2[2]);
      Assert.AreEqual(0, result2[3]);
      Assert.AreEqual(2, result2[4]);
      Assert.AreEqual(0, result2[5]);
      Assert.AreEqual(1, result2[6]);
      Assert.AreEqual(1, result2[7]);
    }

    [TestMethod]
    public void TFIDFNaiveBayesianAlgorithm_PredictTokens()
    {
      // arrange
      var sample = getSample();
      var CLS1 = sample.CachedClasses.ElementAt(0);
      var CLS2 = sample.CachedClasses.ElementAt(1);
      var prep = getDefaultPreprocessor();
      var alg  = new TFIDFNaiveBayesianAlgorithm()
      {
        TFWeightingScheme  = Registry.TFWeightingScheme.LogNormalization,
        IDFWeightingScheme = Registry.IDFWeightingScheme.Standart,
        Preprocessor = prep
       };

      // act
      alg.Train(sample);
      var result1 = alg.PredictTokens(testDocs()[0], 2);
      var result2 = alg.PredictTokens(testDocs()[1], 2);

      // assert
      Assert.AreEqual(2, result1.Length);
      Assert.AreEqual(CLS1, result1[0].Class);
      Assert.AreEqual(-3.9458157882D, result1[0].Score, EPS);
      Assert.AreEqual(CLS2, result1[1].Class);
      Assert.AreEqual(-6.5794596214D, result1[1].Score, EPS);
      Assert.AreEqual(CLS2, result2[0].Class);
      Assert.AreEqual(-16.7032272969D, result2[0].Score, EPS);
      Assert.AreEqual(CLS1, result2[1].Class);
      Assert.AreEqual(-18.6288477476D, result2[1].Score, EPS);
    }

    [TestMethod]
    public void TFIDFNaiveBayesianAlgorithm_Predict()
    {
      // arrange
      var sample = getSample();
      var CLS1 = sample.CachedClasses.ElementAt(0);
      var CLS2 = sample.CachedClasses.ElementAt(1);
      var prep = getDefaultPreprocessor();
      var alg = new TFIDFNaiveBayesianAlgorithm()
      {
        TFWeightingScheme  = Registry.TFWeightingScheme.LogNormalization,
        IDFWeightingScheme = Registry.IDFWeightingScheme.Standart,
        Preprocessor = prep
       };

      // act
      alg.Train(sample);
      var result1 = alg.Predict(testDocs()[0]);
      var result2 = alg.Predict(testDocs()[1]);
      var result3 = alg.Predict(testDocs()[2]);

      // assert
      Assert.AreEqual(CLS1, result1);
      Assert.AreEqual(CLS2, result2);
      Assert.AreEqual(CLS2, result3);
    }

    #endregion

    #region TWCNaiveBayesianAlgorithm

    [TestMethod]
    public void TWCNaiveBayesianAlgorithm_Train()
    {
      // arrange
      var sample = getSample();
      var CLS1 = sample.CachedClasses.ElementAt(0);
      var CLS2 = sample.CachedClasses.ElementAt(1);
      var prep = getDefaultPreprocessor();
      var alg = new TWCNaiveBayesianAlgorithm()
      {
        TFWeightingScheme  = Registry.TFWeightingScheme.LogNormalization,
        IDFWeightingScheme = Registry.IDFWeightingScheme.Standart,
        Preprocessor = prep
       };

      // act
      alg.Train(sample);

      // assert
      Assert.AreEqual(sample, alg.TrainingSample);
      Assert.AreEqual(prep, alg.Preprocessor);
      Assert.AreEqual(2, alg.ClassHist.Length);
      Assert.AreEqual(4, alg.ClassHist[0]);
      Assert.AreEqual(2, alg.ClassHist[1]);
      Assert.AreEqual(6, alg.DataCount);
      Assert.AreEqual(8, alg.DataDim);

      Assert.AreEqual(8, alg.Vocabulary.Count);
      Assert.AreEqual("cat",      alg.Vocabulary[0]);
      Assert.AreEqual("like",     alg.Vocabulary[1]);
      Assert.AreEqual("icecream", alg.Vocabulary[2]);
      Assert.AreEqual("at",       alg.Vocabulary[3]);
      Assert.AreEqual("dog",      alg.Vocabulary[4]);
      Assert.AreEqual("meet",     alg.Vocabulary[5]);
      Assert.AreEqual("world",    alg.Vocabulary[6]);
      Assert.AreEqual("seven",    alg.Vocabulary[7]);

      Assert.AreEqual(2, alg.PriorProbs.Length);
      Assert.AreEqual(Math.Log(4/6.0D), alg.PriorProbs[0], EPS);
      Assert.AreEqual(Math.Log(2/6.0D), alg.PriorProbs[1], EPS);

      Assert.AreEqual(2, alg.Weights.Length);
      Assert.AreEqual(8, alg.Weights[0].Length);
      Assert.AreEqual(8, alg.Weights[1].Length);
      Assert.AreEqual(0.1415017531D, alg.Weights[0][0], EPS);
      Assert.AreEqual(0.1169948657D, alg.Weights[0][1], EPS);
      Assert.AreEqual(0.1415017531D, alg.Weights[0][2], EPS);
      Assert.AreEqual(0.1415017531D, alg.Weights[0][3], EPS);
      Assert.AreEqual(0.0969081526D, alg.Weights[0][4], EPS);
      Assert.AreEqual(0.1415017531D, alg.Weights[0][5], EPS);
      Assert.AreEqual(0.1123506098D, alg.Weights[0][6], EPS);
      Assert.AreEqual(0.1077393596D, alg.Weights[0][7], EPS);

      Assert.AreEqual(0.0968271218, alg.Weights[1][0], EPS);
      Assert.AreEqual(0.1314521840, alg.Weights[1][1], EPS);
      Assert.AreEqual(0.1219882919, alg.Weights[1][2], EPS);
      Assert.AreEqual(0.1250806983, alg.Weights[1][3], EPS);
      Assert.AreEqual(0.1396735112, alg.Weights[1][4], EPS);
      Assert.AreEqual(0.1092522527, alg.Weights[1][5], EPS);
      Assert.AreEqual(0.1190328859, alg.Weights[1][6], EPS);
      Assert.AreEqual(0.1566930542, alg.Weights[1][7], EPS);
    }

    [TestMethod]
    public void TWCNaiveBayesianAlgorithm_ExtractFeatureVector()
    {
      // arrange
      var sample = getSample();
      var CLS1 = sample.CachedClasses.ElementAt(0);
      var CLS2 = sample.CachedClasses.ElementAt(1);
      var prep = getDefaultPreprocessor();
      var alg = new TWCNaiveBayesianAlgorithm()
      {
        TFWeightingScheme  = Registry.TFWeightingScheme.LogNormalization,
        IDFWeightingScheme = Registry.IDFWeightingScheme.Standart,
        Preprocessor = prep
      };
      bool isEmpty;

      // act
      alg.Train(sample);
      var result1 = alg.ExtractFeatureVector(testDocs()[0], out isEmpty);
      var result2 = alg.ExtractFeatureVector(testDocs()[1], out isEmpty);

      // assert
      Assert.AreEqual(8, result1.Length);
      Assert.AreEqual(1, result1[0]);
      Assert.AreEqual(0, result1[1]);
      Assert.AreEqual(1, result1[2]);
      Assert.AreEqual(0, result1[3]);
      Assert.AreEqual(0, result1[4]);
      Assert.AreEqual(0, result1[5]);
      Assert.AreEqual(0, result1[6]);
      Assert.AreEqual(0, result1[7]);

      Assert.AreEqual(8, result2.Length);
      Assert.AreEqual(1, result2[0]);
      Assert.AreEqual(2, result2[1]);
      Assert.AreEqual(1, result2[2]);
      Assert.AreEqual(0, result2[3]);
      Assert.AreEqual(2, result2[4]);
      Assert.AreEqual(0, result2[5]);
      Assert.AreEqual(1, result2[6]);
      Assert.AreEqual(1, result2[7]);
    }

    [TestMethod]
    public void TWCNaiveBayesianAlgorithm_PredictTokens()
    {
      // arrange
      var sample = getSample();
      var CLS1 = sample.CachedClasses.ElementAt(0);
      var CLS2 = sample.CachedClasses.ElementAt(1);
      var prep = getDefaultPreprocessor();
      var alg  = new TWCNaiveBayesianAlgorithm()
      {
        TFWeightingScheme  = Registry.TFWeightingScheme.LogNormalization,
        IDFWeightingScheme = Registry.IDFWeightingScheme.Standart,
        UsePriors = true,
        Preprocessor = prep
       };

      // act
      alg.Train(sample);
      var result1 = alg.PredictTokens(testDocs()[0], 2);
      var result2 = alg.PredictTokens(testDocs()[1], 2);

      // assert
      Assert.AreEqual(2, result1.Length);
      Assert.AreEqual(CLS1, result1[0].Class);
      Assert.AreEqual(-0.1224616D, result1[0].Score, EPS);
      Assert.AreEqual(CLS2, result1[1].Class);
      Assert.AreEqual(-0.8797969D, result1[1].Score, EPS);
      Assert.AreEqual(CLS1, result2[0].Class);
      Assert.AreEqual(0.5254344D, result2[0].Score, EPS);
      Assert.AreEqual(CLS2, result2[1].Class);
      Assert.AreEqual(-0.0618195D, result2[1].Score, EPS);
    }

    [TestMethod]
    public void TWCNaiveBayesianAlgorithm_PredictTokens_NoPriors()
    {
      // arrange
      var sample = getSample();
      var CLS1 = sample.CachedClasses.ElementAt(0);
      var CLS2 = sample.CachedClasses.ElementAt(1);
      var prep = getDefaultPreprocessor();
      var alg  = new TWCNaiveBayesianAlgorithm()
      {
        TFWeightingScheme  = Registry.TFWeightingScheme.LogNormalization,
        IDFWeightingScheme = Registry.IDFWeightingScheme.Standart,
        UsePriors = false, // !!!
        Preprocessor = prep
       };

      // act
      alg.Train(sample);
      var result1 = alg.PredictTokens(testDocs()[0], 2);
      var result2 = alg.PredictTokens(testDocs()[1], 2);

      // assert
      Assert.AreEqual(2, result1.Length);
      Assert.AreEqual(CLS1, result1[0].Class);
      Assert.AreEqual(0.2830035D, result1[0].Score, EPS);
      Assert.AreEqual(CLS2, result1[1].Class);
      Assert.AreEqual(0.2188154D, result1[1].Score, EPS);
      Assert.AreEqual(CLS2, result2[0].Class);
      Assert.AreEqual(1.0367927D, result2[0].Score, EPS);
      Assert.AreEqual(CLS1, result2[1].Class);
      Assert.AreEqual(0.9308995D, result2[1].Score, EPS);
    }

    [TestMethod]
    public void TWCNaiveBayesianAlgorithm_Predict()
    {
      // arrange
      var sample = getSample();
      var CLS1 = sample.CachedClasses.ElementAt(0);
      var CLS2 = sample.CachedClasses.ElementAt(1);
      var prep = getDefaultPreprocessor();
      var alg = new TWCNaiveBayesianAlgorithm()
      {
        TFWeightingScheme  = Registry.TFWeightingScheme.LogNormalization,
        IDFWeightingScheme = Registry.IDFWeightingScheme.Standart,
        Preprocessor = prep
       };

      // act
      alg.Train(sample);
      var result1 = alg.Predict(testDocs()[0]);
      var result2 = alg.Predict(testDocs()[1]);
      var result3 = alg.Predict(testDocs()[2]);

      // assert
      Assert.AreEqual(CLS1, result1);
      Assert.AreEqual(CLS2, result2);
      Assert.AreEqual(CLS2, result3);
    }

    [TestMethod]
    public void TWCNaiveBayesianAlgorithm_Predict_NoPriors()
    {
      // arrange
      var sample = getSample();
      var CLS1 = sample.CachedClasses.ElementAt(0);
      var CLS2 = sample.CachedClasses.ElementAt(1);
      var prep = getDefaultPreprocessor();
      var alg = new TWCNaiveBayesianAlgorithm()
      {
        TFWeightingScheme  = Registry.TFWeightingScheme.LogNormalization,
        IDFWeightingScheme = Registry.IDFWeightingScheme.Standart,
        UsePriors = false, // !!!
        Preprocessor = prep
       };

      // act
      alg.Train(sample);
      var result1 = alg.Predict(testDocs()[0]);
      var result2 = alg.Predict(testDocs()[1]);
      var result3 = alg.Predict(testDocs()[2]);

      // assert
      Assert.AreEqual(CLS1, result1);
      Assert.AreEqual(CLS2, result2);
      Assert.AreEqual(CLS2, result3);
    }

    #endregion

    #region GeneralTextAlgorithm

    [TestMethod]
    public void GeneralTextAlgorithm_Train()
    {
      // arrange
      var sample = getSample();
      var CLS1 = sample.CachedClasses.ElementAt(0);
      var CLS2 = sample.CachedClasses.ElementAt(1);
      var prep = getDefaultPreprocessor();

      var kernel = new TriangularKernel();
      var subAlg = new NaiveBayesianKernelAlgorithm(kernel, 2.0D);
      var alg    = new GeneralTextAlgorithm(subAlg) { Preprocessor = prep };

      // act
      alg.Train(sample);

      // assert
      Assert.AreEqual(subAlg, alg.SubAlgorithm);
      var ts = subAlg.TrainingSample.ToList();
      Assert.AreEqual(1, ts[0].Key[0]);
      Assert.AreEqual(1, ts[0].Key[1]);
      Assert.AreEqual(1, ts[0].Key[2]);
      Assert.AreEqual(0, ts[0].Key[7]);
      Assert.AreEqual(CLS1, ts[0].Value);
      Assert.AreEqual(0, ts[5].Key[3]);
      Assert.AreEqual(2, ts[5].Key[4]);
      Assert.AreEqual(0, ts[5].Key[5]);
      Assert.AreEqual(1, ts[5].Key[6]);
      Assert.AreEqual(CLS2, ts[5].Value);

      Assert.AreEqual(2, subAlg.PriorProbs.Length);
      Assert.AreEqual(Math.Log(4.0D/6), subAlg.PriorProbs[0], EPS);
      Assert.AreEqual(Math.Log(2.0D/6), subAlg.PriorProbs[1], EPS);

      Assert.AreEqual(8, subAlg.DataDim);
      Assert.AreEqual(6, subAlg.DataCount);
      Assert.AreEqual(2, subAlg.ClassHist.Length);
      Assert.AreEqual(4, subAlg.ClassHist[0]);
      Assert.AreEqual(2, subAlg.ClassHist[1]);
    }

    [TestMethod]
    public void GeneralTextAlgorithm_ExtractFeatureVector()
    {
      // arrange
      var sample = getSample();
      var CLS1 = sample.CachedClasses.ElementAt(0);
      var CLS2 = sample.CachedClasses.ElementAt(1);
      var prep = getDefaultPreprocessor();

      var kernel = new TriangularKernel();
      var subAlg = new NaiveBayesianKernelAlgorithm(kernel, 2.0D);
      var alg    = new GeneralTextAlgorithm(subAlg) { Preprocessor = prep };
      bool isEmpty;

      // act
      alg.Train(sample);
      var result1 = alg.ExtractFeatureVector(testDocs()[0], out isEmpty);
      var result2 = alg.ExtractFeatureVector(testDocs()[1], out isEmpty);

      // assert
      Assert.AreEqual(8, result1.Length);
      Assert.AreEqual(1, result1[0]);
      Assert.AreEqual(0, result1[1]);
      Assert.AreEqual(1, result1[2]);
      Assert.AreEqual(0, result1[3]);
      Assert.AreEqual(0, result1[4]);
      Assert.AreEqual(0, result1[5]);
      Assert.AreEqual(0, result1[6]);
      Assert.AreEqual(0, result1[7]);

      Assert.AreEqual(8, result2.Length);
      Assert.AreEqual(1, result2[0]);
      Assert.AreEqual(2, result2[1]);
      Assert.AreEqual(1, result2[2]);
      Assert.AreEqual(0, result2[3]);
      Assert.AreEqual(2, result2[4]);
      Assert.AreEqual(0, result2[5]);
      Assert.AreEqual(1, result2[6]);
      Assert.AreEqual(1, result2[7]);
    }

    [TestMethod]
    public void GeneralTextAlgorithm_PredictTokens()
    {
      // arrange
      var sample = getSample();
      var CLS1 = sample.CachedClasses.ElementAt(0);
      var CLS2 = sample.CachedClasses.ElementAt(1);
      var prep = getDefaultPreprocessor();

      var kernel = new TriangularKernel();
      var subAlg = new NaiveBayesianKernelAlgorithm(kernel, 0.5D) { UseKernelMinValue=true, KernelMinValue=EPS_ROUGH };
      var alg    = new GeneralTextAlgorithm(subAlg) { Preprocessor = prep };

      // act
      alg.Train(sample);
      var result1 = alg.PredictTokens(testDocs()[0], 2);
      var result2 = alg.PredictTokens(testDocs()[1], 2);

      // assert
      Assert.AreEqual(2, result1.Length);
      Assert.AreEqual(CLS1, result1[0].Class);
      Assert.AreEqual(Math.Log(27.0D/8), result1[0].Score, EPS);
      Assert.AreEqual(CLS2, result1[1].Class);
      Assert.AreEqual(Math.Log(EPS_ROUGH*EPS_ROUGH*EPS_ROUGH*8.0D/6), result1[1].Score, EPS);
      Assert.AreEqual(CLS2, result2[0].Class);
      Assert.AreEqual(Math.Log(EPS_ROUGH*EPS_ROUGH*EPS_ROUGH*4.0/3), result2[0].Score, EPS);
      Assert.AreEqual(CLS1, result2[1].Class);
      Assert.AreEqual(Math.Log(EPS_ROUGH*EPS_ROUGH*EPS_ROUGH/4), result2[1].Score, EPS);
    }

    [TestMethod]
    public void GeneralTextAlgorithm_Predict()
    {
      // arrange
      var sample = getSample();
      var CLS1 = sample.CachedClasses.ElementAt(0);
      var CLS2 = sample.CachedClasses.ElementAt(1);
      var prep = getDefaultPreprocessor();

      var kernel = new TriangularKernel();
      var subAlg = new NaiveBayesianKernelAlgorithm(kernel, 2.0D);
      var alg    = new GeneralTextAlgorithm(subAlg){ Preprocessor = prep };

      // act
      alg.Train(sample);
      var result1 = alg.Predict(testDocs()[0]);
      var result2 = alg.Predict(testDocs()[1]);
      var result3 = alg.Predict(testDocs()[2]);

      // assert
      Assert.AreEqual(CLS1, result1);
      Assert.AreEqual(CLS2, result2);
      Assert.AreEqual(CLS2, result3);
    }


    #endregion

    #region .pvt

    private ITextPreprocessor getDefaultPreprocessor()
    {
      return new TextPreprocessor(new EnglishSimpleTokenizer(),
                                  new EnglishStopwords(),
                                  new EnglishSimpleNormalizer(),
                                  new EnglishPorterStemmer());
    }

    private ClassifiedSample<string> getSample()
    {
      var CLS1 = new Class("CAT", 0);
      var CLS2 = new Class("DOG", 1);
      return new ClassifiedSample<string>
      {
        { "my cats like icecream",              CLS1 },
        { "their   cat ate  dogs. Meet my cat", CLS1 },
        { "my cat meet with    other cats",     CLS1 },
        { "Are there no cats in the world?",    CLS1 },
        { "He likes my dog! At seven",             CLS2 },
        { "dogs, dogs everywhere in the world!!!", CLS2 }
      };
    }

    private List<string> testDocs()
    {
      return new List<string>
      {
        "What class icecream cat belongs to?",
        "I  like cats, but I like dogs too. The   dogs are the best icecream eaters in the seven worlds!!!",
        "What can I say about seven dogs? They are the best"
      };
    }

    #endregion
  }
}
