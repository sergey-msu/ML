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
    private Dictionary<ClassFeatureKey, double> m_Frequencies;

    public BinaryNaiveBayesianAlgorithm(ITextPreprocessor preprocessor)
      : base(preprocessor)
    {
    }

    #region Properties

    public override string Name   { get { return "BINNB"; } }

    public Dictionary<ClassFeatureKey, double> Frequencies { get { return m_Frequencies; } }

    #endregion

    public override ClassScore[] PredictTokens(string obj, int cnt)
    {
      var data = ExtractFeatureVector(obj);
      var classes = TrainingSample.CachedClasses;
      var priors = PriorProbs;
      var dim = DataDim;
      var scores = new List<ClassScore>();

      foreach (var cls in classes)
      {
        var score = Math.Log(priors[cls]);
        for (int i=0; i<dim; i++)
        {
          var p = m_Frequencies[new ClassFeatureKey(cls, i)];
          var x = data[i];
          score += x*Math.Log(p) + (1-x)*Math.Log(1-p);
        }

        scores.Add(new ClassScore(cls, score));
      }

      return scores.OrderByDescending(s => s.Score)
                   .Take(cnt)
                   .ToArray();
    }

    public override double[] ExtractFeatureVector(string doc)
    {
      var dict   = Vocabulary;
      var dim    = dict.Count;
      var result = new double[dim];
      var prep   = Preprocessor;
      var tokens = prep.Preprocess(doc);

      foreach (var token in tokens)
      {
        var idx = dict.IndexOf(token);
        if (idx<0) continue;
        result[idx] = 1;
      }

      return result;
    }


    protected override void TrainImpl()
    {
      m_Frequencies = new Dictionary<ClassFeatureKey, double>();
      var cHist = ClassHist;
      var N = Vocabulary.Count;
      var a = Alpha;

      foreach (var doc in TrainingSample)
      {
        var text = doc.Key;
        var cls  = doc.Value;
        var data = ExtractFeatureVector(text);
        var dim = data.Length;

        for (int i=0; i<dim; i++)
        {
          var key = new ClassFeatureKey(cls, i);
          var f = data[i];
          double freq;
          if (!m_Frequencies.TryGetValue(key, out freq)) m_Frequencies[key] = f;
          else m_Frequencies[key] = freq+f;
        }
      }

      foreach (var key in m_Frequencies.Keys.ToList())
      {
        var freq = m_Frequencies[key];
        var total = (double)cHist[key.Class];
        if (UseSmoothing)
        {
          freq += a;
          total += (a*N);
        }

        m_Frequencies[key] = freq/total;
      }
    }


  }
}
