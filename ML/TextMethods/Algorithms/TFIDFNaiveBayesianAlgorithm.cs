using System;
using System.Collections.Generic;
using System.Linq;
using ML.Core;
using ML.Contracts;
using ML.Core.Distributions;

namespace ML.TextMethods.Algorithms
{
  public class TFIDFNaiveBayesianAlgorithm : LinearNaiveBayesianAlgorithmBase
  {
    #region Inner

    protected struct FreqData
    {
      public FreqData(double value, Class cls)
      {
        Value = value;
        Class = cls;
      }

      public readonly double Value;
      public readonly Class  Class;
    }

    #endregion

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

    protected override Dictionary<ClassFeatureKey, double> TrainWeights()
    {
      var dim       = DataDim;
      var cnt       = DataCount;
      var alpha     = Alpha;
      var weights   = new Dictionary<ClassFeatureKey, double>();
      var clsTotals = new Dictionary<Class, double>();
      var idfFreqs  = new int[dim];
      var freqDatas = new FreqData[dim, cnt];

      // TF transform
      int idx = -1;
      foreach (var doc in TrainingSample)
      {
        idx++;
        var text = doc.Key;
        var cls  = doc.Value;
        var data = ExtractFrequencies(text);

        m_TFWeightingScheme.Reset();

        for (int i=0; i<dim; i++)
        {
          var f = data[i];
          var tf = m_TFWeightingScheme.GetFrequency(data, i);
          freqDatas[i, idx] = new FreqData(tf, cls);

          if (f>0) idfFreqs[i] += 1;
        }
      }

      // IDF transform
      var idfWeights = IDFWeightingScheme.GetWeights(dim, idfFreqs);
      for (var i=0; i<dim; i++)
      {
        var idf = idfWeights[i];
        for (var j=0; j<cnt; j++)
        {
          var fData  = freqDatas[i, j];
          var fValue = fData.Value*idf;
          var cls    = fData.Class;
          var key = new ClassFeatureKey(cls, i);

          double w = weights.TryGetValue(key, out w) ? (w+fValue) : fValue;
          weights[key] = w;

          double t = clsTotals.TryGetValue(cls, out t) ? (t+fValue) : fValue;
          clsTotals[cls] = t;
        }
      }

      // calculate weights
      foreach (var key in weights.Keys.ToList())
      {
        var w = weights[key];
        var t = clsTotals[key.Class];
        if (UseSmoothing)
        {
          w += alpha;
          t += (alpha*dim);
        }

        weights[key] = Math.Log(w/t);
      }

      return weights;
    }

  }
}
