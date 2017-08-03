using System;
using System.Collections.Generic;
using System.Linq;
using ML.Core;
using ML.Contracts;
using ML.Core.Distributions;

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
    }

    #region Properties

    public override string Name   { get { return "TWCNB"; } }

    #endregion

    protected override Dictionary<ClassFeatureKey, double> TrainWeights()
    {
      var dim       = DataDim;
      var cnt       = DataCount;
      var classes   = TrainingSample.Classes;
      var alpha     = Alpha;
      var weights   = new Dictionary<ClassFeatureKey, double>();
      var cClsTotals = new Dictionary<Class, double>();
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

        TFWeightingScheme.Reset();

        for (int i=0; i<dim; i++)
        {
          var f = data[i];
          var tf = TFWeightingScheme.GetFrequency(data, i);
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
          freqDatas[i, j] = new FreqData(fData.Value*idf, fData.Class);
        }
      }

      // length norm
      for (var j=0; j<cnt; j++)
      {
        var norm = 0.0D;
        for (var i=0; i<dim; i++)
        {
          var fData = freqDatas[i, j];
          norm += (fData.Value * fData.Value);
        }
        norm = Math.Sqrt(norm);

        for (var i=0; i<dim; i++)
        {
          var fData = freqDatas[i, j];
          freqDatas[i, j] = new FreqData(fData.Value/norm, fData.Class);
        }
      }

      // complements
      for (var j=0; j<cnt; j++)
      {
        foreach (var cCls in classes)
        {
          for (var i=0; i<dim; i++)
          {
            var fData = freqDatas[i, j];
            var f     = fData.Value;
            var cls   = fData.Class;

            if (cCls.Equals(cls)) break;

            var key = new ClassFeatureKey(cCls, i);

            double w = weights.TryGetValue(key, out w) ? w += f : w = f;
            weights[key] = w;

            double t = cClsTotals.TryGetValue(cCls, out t) ? t += f : f;
            cClsTotals[cCls] = t;
          }
        }
      }

      // take logarithm
      foreach (var key in weights.Keys.ToList())
      {
        var w = weights[key];
        var t = cClsTotals[key.Class];
        if (UseSmoothing)
        {
          w += alpha;
          t += (alpha*dim);
        }

        weights[key] = -Math.Log(w/t);
      }

      // weight normalization

      foreach (var cls in classes)
      {
        cClsTotals[cls] = 0;

        for (var i=0; i<dim; i++)
          cClsTotals[cls] += Math.Abs(weights[new ClassFeatureKey(cls, i)]);
      }

      foreach (var cls in classes)
      {
        var t = cClsTotals[cls];
        for (var i=0; i<dim; i++)
        {
          var key = new ClassFeatureKey(cls, i);
          var w = weights[key]/t;
          weights[key] = w;
        }
      }

      return weights;
    }

  }
}
