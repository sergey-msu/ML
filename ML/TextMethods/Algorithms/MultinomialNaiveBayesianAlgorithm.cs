using System;
using System.Collections.Generic;
using System.Linq;
using ML.Core;
using ML.Contracts;
using ML.Core.Distributions;

namespace ML.TextMethods.Algorithms
{
  public class MultinomialNaiveBayesianAlgorithm : NaiveBayesianAlgorithmBase
  {
    public MultinomialNaiveBayesianAlgorithm(ITextPreprocessor preprocessor)
      : base(preprocessor)
    {
    }

    #region Properties

    public override string Name   { get { return "MNOMNB"; } }

    #endregion

    protected override Dictionary<ClassFeatureKey, double> TrainWeights()
    {
      var dim    = DataDim;
      var a      = Alpha;
      var cTotal = new Dictionary<Class, int>();
      var weights  = new Dictionary<ClassFeatureKey, double>();

      foreach (var doc in TrainingSample)
      {
        var text = doc.Key;
        var cls  = doc.Value;
        var data = ExtractFeatureVector(text);

        if (!cTotal.ContainsKey(cls)) cTotal[cls] = 0;

        for (int i=0; i<dim; i++)
        {
          var key = new ClassFeatureKey(cls, i);

          var f = data[i];
          double w = weights.TryGetValue(key, out w) ? (w+f) : f;
          weights[key] = w;

          cTotal[cls] += (int)f;
        }
      }

      foreach (var key in weights.Keys.ToList())
      {
        var w = weights[key];
        var t = (double)cTotal[key.Class];
        if (UseSmoothing)
        {
          w += a;
          t += (a*dim);
        }

        weights[key] = Math.Log(w/t);
      }

      return weights;
    }

  }
}
