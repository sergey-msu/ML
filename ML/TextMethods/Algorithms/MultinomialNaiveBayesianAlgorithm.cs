using System;
using System.Collections.Generic;

namespace ML.TextMethods.Algorithms
{
  public class MultinomialNaiveBayesianAlgorithm : NaiveBayesianAlgorithmBase
  {
    public MultinomialNaiveBayesianAlgorithm()
    {
    }

    #region Properties

    public override string Name   { get { return "MNOMNB"; } }

    #endregion

    protected override double[][] TrainWeights()
    {
      var dim     = DataDim;
      var alp     = Alpha;
      var classes = Classes;
      var cTotals = new int[classes.Length];
      var weights = new double[classes.Length][];
      foreach (var cls in classes)
        weights[cls.Value] = new double[dim];

      foreach (var doc in TrainingSample)
      {
        var text = doc.Key;
        var cls  = doc.Value;
        bool isEmpty;
        var data = ExtractFeatureVector(text, out isEmpty);
        if (isEmpty) continue;

        var ws = weights[cls.Value];

        for (int i=0; i<dim; i++)
        {
          var f = (int)data[i];
          ws[i] += f;
          cTotals[cls.Value] += f;
        }
      }

      foreach (var cls in classes)
      {
        var ws = weights[cls.Value];
        var t = (double)cTotals[cls.Value];
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

  }
}
