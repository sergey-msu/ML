using System;
using System.Collections.Generic;
using System.Linq;
using ML.Core;
using ML.Contracts;
using ML.Core.Distributions;

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
      var classes = TrainingSample.CachedClasses;
      var priors  = PriorProbs;
      var dim     = DataDim;
      var weights = Weights;
      var scores  = new List<ClassScore>();

      foreach (var cls in classes)
      {
        var score = priors[cls];
        for (int i=0; i<dim; i++)
        {
          var p = weights[new ClassFeatureKey(cls, i)];
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


    protected override Dictionary<ClassFeatureKey, double> TrainWeights()
    {
      var cHist = ClassHist;
      var N = Vocabulary.Count;
      var a = Alpha;
      var freqs = new Dictionary<ClassFeatureKey, double>();

      foreach (var doc in TrainingSample)
      {
        var text = doc.Key;
        var cls  = doc.Value;
        bool isEmpty;
        var data = ExtractFeatureVector(text, out isEmpty);
        if (isEmpty) continue;

        for (int i=0; i<N; i++)
        {
          var key = new ClassFeatureKey(cls, i);
          var f = data[i];
          double freq = freqs.TryGetValue(key, out freq) ? freq+f : f;
          freqs[key] = freq;
        }
      }

      foreach (var key in freqs.Keys.ToList())
      {
        var freq = freqs[key];
        var total = (double)cHist[key.Class];
        if (UseSmoothing)
        {
          freq  += a;
          total += (a*N);
        }

        freqs[key] = freq/total;
      }

      return freqs;
    }


  }
}
