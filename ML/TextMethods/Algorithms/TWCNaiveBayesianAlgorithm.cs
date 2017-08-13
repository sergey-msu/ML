using System;
using System.Collections.Generic;
using ML.Core;
using ML.Contracts;

namespace ML.TextMethods.Algorithms
{
  /// <summary>
  /// http://machinelearning.wustl.edu/mlpapers/paper_files/icml2003_RennieSTK03.pdf
  /// </summary>
  public class TWCNaiveBayesianAlgorithm : TFIDFNaiveBayesianAlgorithm
  {
    public TWCNaiveBayesianAlgorithm(ITextPreprocessor preprocessor)
      : base(preprocessor)
    {
      UsePriors = false; // not to use prior probabilities is the default TWC behaviour
    }

    #region Properties

    public override string Name { get { return "TWCNB"; } }

    #endregion

    protected override double[][] TrainWeights()
    {
      var dim = DataDim;
      var cnt = DataCount;
      var alp = Alpha;
      var classes = Classes;
      var idfFreqs = new int[dim];
      var cTotals  = new double[classes.Length];
      var weights  = new double[Classes.Length][];
      foreach (var cls in Classes)
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

        TFWeightingScheme.Reset();

        for (int i=0; i<dim; i++)
        {
          var f = data[i];
          var tf = TFWeightingScheme.GetFrequency(data, i);
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
          var fData = freqData[j];
          freqData[j] = new FreqData(fData.Value*idf, fData.Class);
        }
      }

      // length norm
      for (var j=0; j<cnt; j++)
      {
        var norm = 0.0D;
        for (var i=0; i<dim; i++)
        {
          var fData = freqDatas[i][j];
          norm += (fData.Value * fData.Value);
        }
        norm = Math.Sqrt(norm);

        for (var i=0; i<dim; i++)
        {
          var fData = freqDatas[i][j];
          freqDatas[i][j] = new FreqData(fData.Value/norm, fData.Class);
        }
      }

      // complements
      for (var j=0; j<cnt; j++)
      {
        foreach (var cCls in Classes)
        {
          var ws = weights[cCls.Value];

          for (var i=0; i<dim; i++)
          {
            var fData = freqDatas[i][j];
            var f     = fData.Value;
            var cls   = fData.Class;

            if (cCls.Equals(cls)) break;

            ws[i] += f;
            cTotals[cCls.Value] += f;
          }
        }
      }

      // take logarithm
      foreach (var cls in Classes)
      {
        var ws = weights[cls.Value];
        var t  = cTotals[cls.Value];
        if (UseSmoothing) t += alp*dim;

        for (int i=0; i<dim; i++)
        {
          var w = ws[i];
          if (UseSmoothing) w += alp;

          ws[i] = -Math.Log(w/t);
        }
      }

      // weight normalization

      foreach (var cls in Classes)
      {
        cTotals[cls.Value] = 0;
        var ws = weights[cls.Value];

        for (var i=0; i<dim; i++)
          cTotals[cls.Value] += Math.Abs(ws[i]);
      }

      foreach (var cls in Classes)
      {
        var t = cTotals[cls.Value];
        var ws = weights[cls.Value];

        for (var i=0; i<dim; i++)
          ws[i] = ws[i]/t;
      }

      return weights;
    }

  }
}
