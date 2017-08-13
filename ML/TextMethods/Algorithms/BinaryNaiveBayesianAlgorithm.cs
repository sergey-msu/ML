using System;
using System.Collections.Generic;
using System.Linq;
using ML.Core;
using ML.Contracts;

namespace ML.TextMethods.Algorithms
{
  public class BinaryNaiveBayesianAlgorithm : NaiveBayesianAlgorithmBase
  {
    public BinaryNaiveBayesianAlgorithm(ITextPreprocessor preprocessor)
      : base(preprocessor)
    {
    }

    #region Properties

    public override string Name { get { return "BINNB"; } }

    #endregion

    public override ClassScore[] PredictTokens(string obj, int cnt)
    {
      bool isEmpty;
      var data    = ExtractFeatureVector(obj, out isEmpty);
      var classes = Classes;
      var priors  = PriorProbs;
      var dim     = DataDim;
      var scores  = new List<ClassScore>();

      foreach (var cls in classes)
      {
        var score   = priors[cls.Value];
        var weights = Weights[cls.Value];

        for (int i=0; i<dim; i++)
        {
          var p = weights[i];
          var x = data[i];
          score += x*Math.Log(p) + (1-x)*Math.Log(1-p);
        }

        scores.Add(new ClassScore(cls, score));
      }

      return scores.OrderByDescending(s => s.Score)
                   .Take(cnt)
                   .ToArray();
    }

    public override double[] ExtractFeatureVector(string doc, out bool isEmpty)
    {
      var dict   = Vocabulary;
      var dim    = DataDim;
      var result = new double[dim];
      var prep   = Preprocessor;
      var tokens = prep.Preprocess(doc);
      isEmpty = true;

      foreach (var token in tokens)
      {
        var idx = dict.IndexOf(token);
        if (idx<0) continue;
        result[idx] = 1;
        isEmpty = false;
      }

      return result;
    }


    protected override double[][] TrainWeights()
    {
      var cHist = ClassHist;
      var dim   = Vocabulary.Count;
      var alp   = Alpha;
      var classes = Classes;
      var freqs = new double[Classes.Length][];
      foreach (var cls in classes)
        freqs[cls.Value] = new double[dim];

      foreach (var doc in TrainingSample)
      {
        var text = doc.Key;
        bool isEmpty;
        var data = ExtractFeatureVector(text, out isEmpty);
        if (isEmpty) continue;

        var cls = doc.Value;
        var fs = freqs[cls.Value];
        for (int i=0; i<dim; i++)
          fs[i] += data[i];
      }

      foreach (var cls in classes)
      {
        var fs = freqs[cls.Value];
        var t = (double)cHist[cls.Value];
        if (UseSmoothing) t += alp*dim;

        for (int i=0; i<dim; i++)
        {
          var f = fs[i];
          if (UseSmoothing) f += alp;

          fs[i] = f/t;
        }
      }

      return freqs;
    }
  }
}
