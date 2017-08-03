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
  public class ComplementNaiveBayesianAlgorithm : LinearNaiveBayesianAlgorithmBase
  {
    public ComplementNaiveBayesianAlgorithm(ITextPreprocessor preprocessor)
      : base(preprocessor)
    {
    }

    #region Properties

    public override string Name   { get { return "COMPLNB"; } }

    #endregion

    protected override Dictionary<ClassFeatureKey, double> TrainWeights()
    {
      var dim      = DataDim;
      var a        = Alpha;
      var ccTotals = new Dictionary<Class, int>();
      var weights  = new Dictionary<ClassFeatureKey, double>();
      var classes  = TrainingSample.Classes;

      foreach (var doc in TrainingSample)
      {
        var text = doc.Key;
        var cls  = doc.Value;
        var data = ExtractFeatureVector(text);

        foreach (var cCls in classes)
        {
          if (cCls.Equals(cls)) continue;
          if (!ccTotals.ContainsKey(cCls)) ccTotals[cCls] = 0;

          for (int i=0; i<dim; i++)
          {
            var key = new ClassFeatureKey(cCls, i);

            var f = data[i];
            double w = weights.TryGetValue(key, out w) ? (w+f) : f;
            weights[key] = w;

            ccTotals[cCls] += (int)f;
          }
        }
      }

      foreach (var key in weights.Keys.ToList())
      {
        var w = weights[key];
        var t = (double)ccTotals[key.Class];
        if (UseSmoothing)
        {
          w += a;
          t += (a*dim);
        }

        weights[key] = -Math.Log(w/t);
      }

      return weights;
    }

  }
}
