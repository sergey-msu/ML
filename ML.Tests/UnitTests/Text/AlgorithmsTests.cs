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
      var alg = new BinaryNaiveBayesianAlgorithm(prep);

      // act
      alg.Train(sample);

      // assert
      Assert.AreEqual(sample, alg.TrainingSample);
      Assert.AreEqual(prep, alg.Preprocessor);
      Assert.AreEqual(2, alg.ClassHist.Count);
      Assert.AreEqual(4, alg.ClassHist[CLS1]);
      Assert.AreEqual(2, alg.ClassHist[CLS2]);
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

      Assert.AreEqual(2, alg.PriorProbs.Count);
      Assert.AreEqual(4/6.0D, alg.PriorProbs[CLS1]);
      Assert.AreEqual(2/6.0D, alg.PriorProbs[CLS2]);

      Assert.AreEqual(16, alg.Frequencies.Count);
      Assert.AreEqual(   1, alg.Frequencies[new ClassFeatureKey(CLS1, 0)]);
      Assert.AreEqual(0.25, alg.Frequencies[new ClassFeatureKey(CLS1, 1)]);
      Assert.AreEqual(0.25, alg.Frequencies[new ClassFeatureKey(CLS1, 2)]);
      Assert.AreEqual(0.25, alg.Frequencies[new ClassFeatureKey(CLS1, 3)]);
      Assert.AreEqual(0.25, alg.Frequencies[new ClassFeatureKey(CLS1, 4)]);
      Assert.AreEqual( 0.5, alg.Frequencies[new ClassFeatureKey(CLS1, 5)]);
      Assert.AreEqual(0.25, alg.Frequencies[new ClassFeatureKey(CLS1, 6)]);
      Assert.AreEqual(1/12.0D, alg.Frequencies[new ClassFeatureKey(CLS1, 7)]);

      Assert.AreEqual(0.1, alg.Frequencies[new ClassFeatureKey(CLS2, 0)]);
      Assert.AreEqual(0.5, alg.Frequencies[new ClassFeatureKey(CLS2, 1)]);
      Assert.AreEqual(0.1, alg.Frequencies[new ClassFeatureKey(CLS2, 2)]);
      Assert.AreEqual(0.1, alg.Frequencies[new ClassFeatureKey(CLS2, 3)]);
      Assert.AreEqual(  1, alg.Frequencies[new ClassFeatureKey(CLS2, 4)]);
      Assert.AreEqual(0.1, alg.Frequencies[new ClassFeatureKey(CLS2, 5)]);
      Assert.AreEqual(0.5, alg.Frequencies[new ClassFeatureKey(CLS2, 6)]);
      Assert.AreEqual(0.5, alg.Frequencies[new ClassFeatureKey(CLS2, 7)]);
    }

    [TestMethod]
    public void BinaryNaiveBayesianAlgorithm_ExtractFeatureVector()
    {
      // arrange
      var sample = getSample();
      var CLS1 = sample.CachedClasses.ElementAt(0);
      var CLS2 = sample.CachedClasses.ElementAt(1);
      var prep = getDefaultPreprocessor();
      var alg = new BinaryNaiveBayesianAlgorithm(prep);

      // act
      alg.Train(sample);
      var result1 = alg.ExtractFeatureVector(testDocs()[0]);
      var result2 = alg.ExtractFeatureVector(testDocs()[1]);

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
      var alg = new BinaryNaiveBayesianAlgorithm(prep);

      // act
      alg.Train(sample);
      var result1 = alg.PredictTokens(testDocs()[0], 2);
      var result2 = alg.PredictTokens(testDocs()[1], 2);

      // assert
      Assert.AreEqual(2, result1.Length);
      Assert.AreEqual(CLS1, result1[0].Class);
      Assert.AreEqual(CLS1, result1[0].Score);



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
      var CLS1 = new Class("CAT", 1);
      var CLS2 = new Class("DOG", 2);
      return new ClassifiedSample<string>
      {
        { "my cats like icecream",              CLS1 },
        { "their   cat ate  dogs. Meet my cat", CLS1 },
        { "my cat meet with    other cats",     CLS1 },
        { "Are there no cats in the world?",    CLS1 },
        { "He likes my dog! At seven",  CLS2 },
        { "dogs, dogs everywhere in the world!!!", CLS2 },
      };
    }

    private List<string> testDocs()
    {
      return new List<string>
      {
        "What class icecream cat belongs to?",
        "I  like cats, but I like dogs too. The   dogs are the best icecream eaters in the seven worlds!!!"
      };
    }

    #endregion
  }
}
