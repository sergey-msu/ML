using System;
using System.Collections.Generic;
using Microsoft.VisualStudio.TestTools.UnitTesting;
using ML.TextMethods.FeatureExtractors;
using ML.TextMethods.Preprocessing;
using ML.TextMethods.Tokenization;
using ML.TextMethods.Stemming;
using ML.TextMethods.Normalization;
using ML.TextMethods.Stopwords;

namespace ML.Tests.UnitTests.Text
{
  [TestClass]
  public class FeatureExtractorTests : TestBase
  {
    [ClassInitialize]
    public static void ClassInit(TestContext context)
    {
      BaseClassInit(context);
    }

    [TestMethod]
    public void BinaryFeatureExtractor_ExtractFeatureVector()
    {
      // arrange
      var proc = new TextPreprocessor(
                   new EnglishSimpleTokenizer(),
                   new EnglishStopwords(),
                   new EnglishSimpleNormalizer(),
                   new EnglishPorterStemmer());
      var extr = new BinaryFeatureExtractor();
      extr.Preprocessor = proc;
      extr.Vocabulary = new List<string> { "cat", "dog", "fish" };

      // act
      bool e1;
      bool e2;
      bool e3;
      var f1 = extr.ExtractFeatureVector("cats eat cat not dogs and fishes", out e1);
      var f2 = extr.ExtractFeatureVector("cat is not a fish or two fishes",  out e2);
      var f3 = extr.ExtractFeatureVector("to be a man or not to be a man",   out e3);

      //assert
      Assert.IsFalse(e1);
      Assert.AreEqual(3, f1.Length);
      Assert.AreEqual(1, f1[0]);
      Assert.AreEqual(1, f1[1]);
      Assert.AreEqual(1, f1[2]);
      Assert.IsFalse(e2);
      Assert.AreEqual(3, f2.Length);
      Assert.AreEqual(1, f2[0]);
      Assert.AreEqual(0, f2[1]);
      Assert.AreEqual(1, f2[2]);
      Assert.IsTrue(e3);
      Assert.AreEqual(3, f3.Length);
      Assert.AreEqual(0, f3[0]);
      Assert.AreEqual(0, f3[1]);
      Assert.AreEqual(0, f3[2]);
    }

    [TestMethod]
    public void MultinomialFeatureExtractor_ExtractFeatureVector()
    {
      // arrange
      var proc = new TextPreprocessor(
                   new EnglishSimpleTokenizer(),
                   new EnglishStopwords(),
                   new EnglishSimpleNormalizer(),
                   new EnglishPorterStemmer());
      var extr = new MultinomialFeatureExtractor();
      extr.Preprocessor = proc;
      extr.Vocabulary = new List<string> { "cat", "dog", "fish" };

      // act
      bool e1;
      bool e2;
      bool e3;
      var f1 = extr.ExtractFeatureVector("cats eat cat not dogs and fishes", out e1);
      var f2 = extr.ExtractFeatureVector("cat is not a fish or two fishes",  out e2);
      var f3 = extr.ExtractFeatureVector("to be a man or not to be a man",   out e3);

      //assert
      Assert.IsFalse(e1);
      Assert.AreEqual(3, f1.Length);
      Assert.AreEqual(2, f1[0]);
      Assert.AreEqual(1, f1[1]);
      Assert.AreEqual(1, f1[2]);
      Assert.IsFalse(e2);
      Assert.AreEqual(3, f2.Length);
      Assert.AreEqual(1, f2[0]);
      Assert.AreEqual(0, f2[1]);
      Assert.AreEqual(2, f2[2]);
      Assert.IsTrue(e3);
      Assert.AreEqual(3, f3.Length);
      Assert.AreEqual(0, f3[0]);
      Assert.AreEqual(0, f3[1]);
      Assert.AreEqual(0, f3[2]);
    }

    [TestMethod]
    public void FourierFeatureExtractor_ExtractFeatureVector()
    {
      // arrange
      var proc = new TextPreprocessor(
                   new EnglishSimpleTokenizer(),
                   new EnglishStopwords(),
                   new EnglishSimpleNormalizer(),
                   new EnglishPorterStemmer());
      var extr = new FourierFeatureExtractor { T = 2 };
      extr.Preprocessor = proc;
      extr.Vocabulary = new List<string> { "cat", "dog", "fish" };

      // act
      bool e1;
      bool e2;
      bool e3;
      var f1 = extr.ExtractFeatureVector("cats eat cat not dogs and fishes", out e1);
      var f2 = extr.ExtractFeatureVector("cat is not a fish or two fishes",  out e2);
      var f3 = extr.ExtractFeatureVector("to be a man or not to be a man",   out e3);

      //assert
      Assert.IsFalse(e1);
      Assert.AreEqual(3, f1.Length);
      Assert.AreEqual(Math.Sqrt((1+Math.Cos(2))*(1+Math.Cos(2)) + Math.Sin(2)*Math.Sin(2)), f1[0], EPS);
      Assert.AreEqual(1, f1[1], EPS);
      Assert.AreEqual(1, f1[2], EPS);
      Assert.IsFalse(e2);
      Assert.AreEqual(3, f2.Length);
      Assert.AreEqual(1, f2[0], EPS);
      Assert.AreEqual(0, f2[1], EPS);
      Assert.AreEqual(Math.Sqrt((1+Math.Cos(2))*(1+Math.Cos(2)) + Math.Sin(2)*Math.Sin(2)), f2[2]);
      Assert.IsTrue(e3);
      Assert.AreEqual(3, f3.Length);
      Assert.AreEqual(0, f3[0], EPS);
      Assert.AreEqual(0, f3[1], EPS);
      Assert.AreEqual(0, f3[2], EPS);
    }

    [TestMethod]
    public void ExtendedFourierFeatureExtractor_ExtractFeatureVector()
    {
      // arrange
      var proc = new TextPreprocessor(
                   new EnglishSimpleTokenizer(),
                   new EnglishStopwords(),
                   new EnglishSimpleNormalizer(),
                   new EnglishPorterStemmer());
      var extr = new ExtendedFourierFeatureExtractor { T = 2 };
      extr.Preprocessor = proc;
      extr.Vocabulary = new List<string> { "cat", "dog", "fish" };

      // act
      bool e1;
      bool e2;
      bool e3;
      var f1 = extr.ExtractFeatureVector("cats eat cat not dogs and fishes", out e1);
      var f2 = extr.ExtractFeatureVector("cat is not a fish or two fishes",  out e2);
      var f3 = extr.ExtractFeatureVector("to be a man or not to be a man",   out e3);

      //assert
      Assert.IsFalse(e1);
      Assert.AreEqual(6, f1.Length);
      Assert.AreEqual(2, f1[0]);
      Assert.AreEqual(1, f1[1]);
      Assert.AreEqual(1, f1[2]);
      Assert.AreEqual(Math.Sqrt((1+Math.Cos(2))*(1+Math.Cos(2)) + Math.Sin(2)*Math.Sin(2)), f1[3], EPS);
      Assert.AreEqual(1, f1[4], EPS);
      Assert.AreEqual(1, f1[5], EPS);
      Assert.IsFalse(e2);
      Assert.AreEqual(6, f2.Length);
      Assert.AreEqual(1, f2[0]);
      Assert.AreEqual(0, f2[1]);
      Assert.AreEqual(2, f2[2]);
      Assert.AreEqual(1, f2[3], EPS);
      Assert.AreEqual(0, f2[4], EPS);
      Assert.AreEqual(Math.Sqrt((1+Math.Cos(2))*(1+Math.Cos(2)) + Math.Sin(2)*Math.Sin(2)), f2[5]);
      Assert.IsTrue(e3);
      Assert.AreEqual(6, f3.Length);
      Assert.AreEqual(0, f3[0]);
      Assert.AreEqual(0, f3[1]);
      Assert.AreEqual(0, f3[2]);
      Assert.AreEqual(0, f3[3], EPS);
      Assert.AreEqual(0, f3[4], EPS);
      Assert.AreEqual(0, f3[5], EPS);
    }


  }
}
