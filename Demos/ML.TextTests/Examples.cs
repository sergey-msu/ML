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
      var alg    = new GeneralTextAlgorithm(proc, subAlg);

      return alg;
    }

    public static TextAlgorithmBase Create_MultinomialAlgorithm()
    {
      var proc = new TextPreprocessor(new EnglishSimpleTokenizer(),
                                      new EnglishStopwords(),
                                      new EnglishSimpleNormalizer(),
                                      new EnglishPorterStemmer());
      var alg = new MultinomialNaiveBayesianAlgorithm(proc);

      return alg;
    }

    public static TextAlgorithmBase Create_TFIDFAlgorithm()
    {
      var proc = new TextPreprocessor(new EnglishSimpleTokenizer(),
                                      new EnglishStopwords(),
                                      new EnglishSimpleNormalizer(),
                                      new EnglishPorterStemmer());
      var alg = new TFIDFNaiveBayesianAlgorithm(proc);

      return alg;
    }

    public static TextAlgorithmBase Create_BinaryAlgorithm()
    {
      var proc = new TextPreprocessor(new EnglishSimpleTokenizer(),
                                      new EnglishStopwords(),
                                      new EnglishSimpleNormalizer(),
                                      new EnglishPorterStemmer());
      var alg = new BinaryNaiveBayesianAlgorithm(proc);

      return alg;
    }

    public static TextAlgorithmBase Create_ComplementAlgorithm()
    {
      var proc = new TextPreprocessor(new EnglishSimpleTokenizer(),
                                      new EnglishStopwords(),
                                      new EnglishSimpleNormalizer(),
                                      new EnglishPorterStemmer());
      var alg = new ComplementNaiveBayesianAlgorithm(proc);

      return alg;
    }

    public static TextAlgorithmBase Create_ComplementOVAAlgorithm()
    {
      var proc = new TextPreprocessor(new EnglishSimpleTokenizer(),
                                      new EnglishStopwords(),
                                      new EnglishSimpleNormalizer(),
                                      new EnglishPorterStemmer());
      var alg = new ComplementOVANaiveBayesianAlgorithm(proc);

      return alg;
    }

    public static TextAlgorithmBase Create_TWCAlgorithm()
    {
      var proc = new TextPreprocessor(new EnglishSimpleTokenizer(),
                                      new EnglishStopwords(),
                                      new EnglishSimpleNormalizer(),
                                      new EnglishPorterStemmer());
      var alg = new TWCNaiveBayesianAlgorithm(proc);

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
      var alg = new ComplementNaiveBayesianAlgorithm(proc);

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
      var alg = new MultinomialNaiveBayesianAlgorithm(proc);

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
      var alg = new TFIDFNaiveBayesianAlgorithm(proc);

      return alg;
    }

    #endregion
  }
}
