using System;
using System.Collections.Generic;
using ML.Core.Kernels;
using ML.BayesianMethods.Algorithms;
using ML.TextMethods.Algorithms;
using ML.TextMethods.Preprocessing;
using ML.TextMethods.Tokenization;
using ML.TextMethods.Stopwords;
using ML.TextMethods.Normalization;
using ML.TextMethods.Stemming;
using ML.TextMethods.FeatureExtractors;

namespace ML.TextTests
{
  public static class Examples
  {
    #region All

    public static TextAlgorithmBase Create_GeneralTextAlgorithm()
    {
      var proc = new TextPreprocessor(new EnglishSimpleTokenizer(),
                                      new EnglishStopwords(),
                                      new EnglishSimpleNormalizer(),
                                      new EnglishPorterStemmer());
      var kernel = new TriangularKernel();
      var subAlg = new NaiveBayesianKernelAlgorithm(kernel, 0.5D) { UseKernelMinValue=true, KernelMinValue=0.000001D };
      var alg    = new GeneralTextAlgorithm(subAlg) { Preprocessor = proc };

      return alg;
    }

    public static TextAlgorithmBase Create_FourierGeneralTextAlgorithm(double t)
    {
      var proc = new TextPreprocessor(new EnglishSimpleTokenizer(),
                                      new EnglishStopwords(),
                                      new EnglishSimpleNormalizer(),
                                      new EnglishPorterStemmer());
      var kernel = new TriangularKernel();
      var subAlg = new NaiveBayesianKernelAlgorithm(kernel, 0.5D) { UseKernelMinValue=true, KernelMinValue=0.000001D };
      var alg    = new GeneralTextAlgorithm(subAlg) { Preprocessor = proc, FeatureExtractor=new ExtendedFourierFeatureExtractor { T=t } };

      return alg;
    }

    public static TextAlgorithmBase Create_MultinomialAlgorithm()
    {
      var proc = new TextPreprocessor(new EnglishSimpleTokenizer(),
                                      new EnglishStopwords(),
                                      new EnglishSimpleNormalizer(),
                                      new EnglishPorterStemmer());
      var alg = new MultinomialNaiveBayesianAlgorithm() { Preprocessor = proc };

      return alg;
    }

    public static TextAlgorithmBase Create_FourierMultinomialAlgorithm(double t)
    {
      var proc = new TextPreprocessor(new EnglishSimpleTokenizer(),
                                      new EnglishStopwords(),
                                      new EnglishSimpleNormalizer(),
                                      new EnglishPorterStemmer());
      var alg = new MultinomialNaiveBayesianAlgorithm() { Preprocessor = proc, FeatureExtractor=new FourierFeatureExtractor { T=t } };

      return alg;
    }

    public static TextAlgorithmBase Create_TFIDFAlgorithm()
    {
      var proc = new TextPreprocessor(new EnglishSimpleTokenizer(),
                                      new EnglishStopwords(),
                                      new EnglishSimpleNormalizer(),
                                      new EnglishPorterStemmer());
      var alg = new TFIDFNaiveBayesianAlgorithm() { Preprocessor = proc };

      return alg;
    }

    public static TextAlgorithmBase Create_FourierTFIDFAlgorithm(double t)
    {
      var proc = new TextPreprocessor(new EnglishSimpleTokenizer(),
                                      new EnglishStopwords(),
                                      new EnglishSimpleNormalizer(),
                                      new EnglishPorterStemmer());
      var alg = new TFIDFNaiveBayesianAlgorithm() { Preprocessor = proc, FeatureExtractor=new FourierFeatureExtractor { T=t } };

      return alg;
    }

    public static TextAlgorithmBase Create_BinaryAlgorithm()
    {
      var proc = new TextPreprocessor(new EnglishSimpleTokenizer(),
                                      new EnglishStopwords(),
                                      new EnglishSimpleNormalizer(),
                                      new EnglishPorterStemmer());
      var alg = new BinaryNaiveBayesianAlgorithm() { Preprocessor = proc };

      return alg;
    }

    public static TextAlgorithmBase Create_ComplementAlgorithm()
    {
      var proc = new TextPreprocessor(new EnglishSimpleTokenizer(),
                                      new EnglishStopwords(),
                                      new EnglishSimpleNormalizer(),
                                      new EnglishPorterStemmer());
      var alg = new ComplementNaiveBayesianAlgorithm() { Preprocessor = proc };

      return alg;
    }

    public static TextAlgorithmBase Create_ComplementOVAAlgorithm()
    {
      var proc = new TextPreprocessor(new EnglishSimpleTokenizer(),
                                      new EnglishStopwords(),
                                      new EnglishSimpleNormalizer(),
                                      new EnglishPorterStemmer());
      var alg = new ComplementOVANaiveBayesianAlgorithm() { Preprocessor = proc };

      return alg;
    }

    public static TextAlgorithmBase Create_TWCAlgorithm()
    {
      var proc = new TextPreprocessor(new EnglishSimpleTokenizer(),
                                      new EnglishStopwords(),
                                      new EnglishSimpleNormalizer(),
                                      new EnglishPorterStemmer());
      var alg = new TWCNaiveBayesianAlgorithm() { Preprocessor = proc };

      return alg;
    }

    #endregion

    #region Spam

    public static TextAlgorithmBase Create_SpamAlgorithm()
    {
      var proc = new TextPreprocessor(new EnglishSimpleTokenizer(),
                                      new EnglishStopwords(),
                                      new EnglishSimpleNormalizer(),
                                      new EnglishPorterStemmer());
      var alg = new ComplementNaiveBayesianAlgorithm() { Preprocessor = proc };

      return alg;
    }

    #endregion

    #region Reuters R8

    public static TextAlgorithmBase Create_ReutersR8()
    {
      var proc = new TextPreprocessor(new EnglishSimpleTokenizer(),
                                      new EnglishStopwords(),
                                      new EnglishSimpleNormalizer(),
                                      new EnglishPorterStemmer());
      var alg = new MultinomialNaiveBayesianAlgorithm() { Preprocessor = proc };

      return alg;
    }

    #endregion

    #region 20 Newsgroups

    public static TextAlgorithmBase Create_Newsgroups20Algorithm()
    {
      var proc = new TextPreprocessor(new EnglishSimpleTokenizer(),
                                      new EnglishStopwords(),
                                      new EnglishSimpleNormalizer(),
                                      new EnglishPorterStemmer());
      var alg = new TFIDFNaiveBayesianAlgorithm() { Preprocessor = proc };

      return alg;
    }

    #endregion
  }
}
