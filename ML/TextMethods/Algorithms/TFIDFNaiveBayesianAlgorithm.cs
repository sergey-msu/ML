using System;
using System.Collections.Generic;
using ML.Core;
using ML.Contracts;
using ML.Core.Serialization;

namespace ML.TextMethods.Algorithms
{
  public class TFIDFNaiveBayesianAlgorithm : NaiveBayesianAlgorithmBase
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

    protected override double[][] TrainWeights()
    {
      var dim = DataDim;
      var cnt = DataCount;
      var alp = Alpha;
      var classes = Classes;
      var cTotals  = new double[classes.Length];
      var idfFreqs = new int[dim];
      var weights  = new double[classes.Length][];
      foreach (var cls in classes)
        weights[cls.Value] = new double[dim];
      var freqDatas = new FreqData[dim][];
      for (int i=0; i<dim; i++)
        freqDatas[i] = new FreqData[cnt];

      // TF transform
      int idx = -1;
      foreach (var doc in TrainingSample)
      {
        var text = doc.Key;
        var cls  = doc.Value;
        bool isEmpty;
        var data = ExtractFeatureVector(text, out isEmpty);
        if (isEmpty) continue;

        idx++;

        m_TFWeightingScheme.Reset();

        for (int i=0; i<dim; i++)
        {
          var f = data[i];
          var tf = m_TFWeightingScheme.GetFrequency(data, i);
          freqDatas[i][idx] = new FreqData(tf, cls);

          if (f>0) idfFreqs[i] += 1;
        }
      }

      cnt = idx+1;

      // IDF transform
      var idfWeights = IDFWeightingScheme.GetWeights(dim, idfFreqs);
      for (var i=0; i<dim; i++)
      {
        var idf = idfWeights[i];
        var freqData = freqDatas[i];

        for (var j=0; j<cnt; j++)
        {
          var fData  = freqData[j];
          var fValue = fData.Value*idf;
          var cls    = fData.Class;

          weights[cls.Value][i] += fValue;
          cTotals[cls.Value] += fValue;
        }
      }

      // calculate weights
      foreach (var cls in classes)
      {
        var ws = weights[cls.Value];
        var t  = cTotals[cls.Value];
        if (UseSmoothing) t += alp*dim;

        for (int i=0; i<dim; i++)
        {
          var w = ws[i];
          if (UseSmoothing) w += alp;

          ws[i] = Math.Log(w/t);
        }
      }

      return weights;
    }


    #region Serialization

    public override void Serialize(MLSerializer ser)
    {
      base.Serialize(ser);

      ser.Write("TF_SCHEME", m_TFWeightingScheme);
      ser.Write("IDF_SCHEME", m_IDFWeightingScheme);
    }

    public override void Deserialize(MLSerializer ser)
    {
      base.Deserialize(ser);

      m_TFWeightingScheme  = ser.ReadObject<ITFWeightingScheme>("TF_SCHEME");
      m_IDFWeightingScheme = ser.ReadObject<IIDFWeightingScheme>("IDF_SCHEME");
    }

    #endregion
  }
}
