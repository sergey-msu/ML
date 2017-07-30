using System;
using System.Collections.Generic;
using System.Linq;
using ML.Core;
using ML.Contracts;
using ML.Core.Distributions;

namespace ML.TextMethods.Algorithms
{
  public class TFIDFNaiveBayesianAlgorithm : NaiveBayesianAlgorithmBase
  {
    private Dictionary<ClassFeatureKey, double> m_Frequencies;
    private List<double>        m_IDFWeights;
    private ITFWeightingScheme  m_TFWeightingScheme;
    private IIDFWeightingScheme m_IDFWeightingScheme;


    public TFIDFNaiveBayesianAlgorithm(ITextPreprocessor preprocessor)
      : base(preprocessor)
    {
      TFWeightingScheme  = Registry.TFWeightingScheme.RawCount;
      IDFWeightingScheme = Registry.IDFWeightingScheme.Standart;
    }

    #region Properties

    public override string Name { get { return "TFIDFNB"; } }

    public Dictionary<ClassFeatureKey, double> Frequencies { get { return m_Frequencies; } }

    public ITFWeightingScheme TFWeightingScheme
    {
      get { return m_TFWeightingScheme; }
      set
      {
        if (value==null)
          throw new MLException("TF weighting scheme cannot be null");

        m_TFWeightingScheme=value;
      }
    }

    public IIDFWeightingScheme IDFWeightingScheme
    {
      get { return m_IDFWeightingScheme; }
      set
      {
        if (value==null)
          throw new MLException("IDF weighting scheme cannot be null");

        m_IDFWeightingScheme=value;
      }
    }


    #endregion

    public override ClassScore[] PredictTokens(string obj, int cnt)
    {
      var data    = ExtractFeatureVector(obj);
      var classes = TrainingSample.CachedClasses;
      var priors  = PriorProbs;
      var dim     = DataDim;
      var scores  = new List<ClassScore>();

      foreach (var cls in classes)
      {
        var score = Math.Log(priors[cls]);
        for (int i=0; i<dim; i++)
        {
          var x = data[i];
          if (x==0) continue;
          var p = m_Frequencies[new ClassFeatureKey(cls, i)];
          score += x*Math.Log(p);
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
      var freqs  = new int[dim];
      var prep   = Preprocessor;
      var tokens = prep.Preprocess(doc);

      foreach (var token in tokens)
      {
        var idx = dict.IndexOf(token);
        if (idx<0) continue;
        freqs[idx] += 1;
      }

      m_TFWeightingScheme.Reset();
      for (int i=0; i<dim; i++)
      {
        var freq = freqs[i];
        result[i] = m_TFWeightingScheme.GetFrequency(freqs, i) * m_IDFWeights[i];
      }

      return result;
    }


    protected override void TrainImpl()
    {
      var cnt = DataCount;
      var N = Vocabulary.Count;
      var a = Alpha;
      var cTotal    = new Dictionary<Class, int>();
      var idfFreqs  = new List<int>(N);
      m_Frequencies = new Dictionary<ClassFeatureKey, double>();

      foreach (var doc in TrainingSample)
      {
        var text = doc.Key;
        var cls  = doc.Value;
        var data = ExtractFeatureVector(text);

        if (!cTotal.ContainsKey(cls)) cTotal[cls] = 0;

        for (int i=0; i<N; i++)
        {
          var key = new ClassFeatureKey(cls, i);
          var f = data[i];

          double freq;
          if (!m_Frequencies.TryGetValue(key, out freq)) m_Frequencies[key] = f;
          else m_Frequencies[key] = freq+f;

          if (f>0) idfFreqs[i] += 1;
          cTotal[cls] += (int)f;
        }
      }

      foreach (var key in m_Frequencies.Keys.ToList())
      {
        var freq = m_Frequencies[key];
        var total = (double)cTotal[key.Class];
        if (UseSmoothing)
        {
          freq  += a;
          total += (a*N);
        }

        m_Frequencies[key] = freq/total;
      }

      m_IDFWeights = IDFWeightingScheme.GetWeights(cnt, idfFreqs);
    }

  }
}
