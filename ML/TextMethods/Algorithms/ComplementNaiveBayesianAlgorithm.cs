using System;
using ML.Contracts;

namespace ML.TextMethods.Algorithms
{
  /// <summary>
  /// http://machinelearning.wustl.edu/mlpapers/paper_files/icml2003_RennieSTK03.pdf
  /// </summary>
  public class ComplementNaiveBayesianAlgorithm : NaiveBayesianAlgorithmBase
  {
    public ComplementNaiveBayesianAlgorithm(ITextPreprocessor preprocessor)
      : base(preprocessor)
    {
    }

    #region Properties

    public override string Name { get { return "COMPLNB"; } }

    #endregion

    protected override double[][] TrainWeights()
    {
      var dim      = DataDim;
      var alp      = Alpha;
      var classes  = Classes;
      var ccTotals = new int[classes.Length];
      var weights  = new double[classes.Length][];
      foreach (var cls in classes)
        weights[cls.Value] = new double[dim];

      foreach (var doc in TrainingSample)
      {
        var text = doc.Key;
        var cls  = doc.Value;
        bool isEmpty;
        var data = ExtractFeatureVector(text, out isEmpty);
        if (isEmpty) continue;

        foreach (var cCls in classes)
        {
          if (cCls.Equals(cls)) continue;

          var ws = weights[cCls.Value];

          for (int i=0; i<dim; i++)
          {
            var f = (int)data[i];
            ws[i] += f;
            ccTotals[cCls.Value] += f;
          }
        }
      }

      foreach (var cls in classes)
      {
        var ws = weights[cls.Value];
        var t  = (double)ccTotals[cls.Value];
        if (UseSmoothing) t += alp*dim;

        for (int i=0; i<dim; i++)
        {
          var w = ws[i];
          if (UseSmoothing) w += alp;

          ws[i] = -Math.Log(w/t);
        }
      }

      return weights;
    }

  }
}
