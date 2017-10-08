using System;
using ML.Contracts;

namespace ML.TextMethods.Algorithms
{
  /// <summary>
  /// http://machinelearning.wustl.edu/mlpapers/paper_files/icml2003_RennieSTK03.pdf
  /// </summary>
  public class ComplementOVANaiveBayesianAlgorithm : NaiveBayesianAlgorithmBase
  {
    public ComplementOVANaiveBayesianAlgorithm()
    {
    }

    #region Properties

    public override string Name   { get { return "COMPOVALNB"; } }

    #endregion

    protected override double[][] TrainWeights()
    {
      var dim      = DataDim;
      var alp      = Alpha;
      var classes  = Classes;
      var ccTotals = new int[classes.Length];
      var cTotals  = new int[classes.Length];
      var freqs    = new double[classes.Length][];
      var cFreqs   = new double[classes.Length][];
      foreach (var cls in classes)
      {
        freqs[cls.Value]  = new double[dim];
        cFreqs[cls.Value] = new double[dim];
      }

      foreach (var doc in TrainingSample)
      {
        var text = doc.Key;
        var cls  = doc.Value;
        bool isEmpty;
        var data = ExtractFeatureVector(text, out isEmpty);
        if (isEmpty) continue;

        foreach (var cCls in classes)
        {
          var sameClass = cCls.Equals(cls);
          var fr = sameClass ? freqs : cFreqs;
          var ct = sameClass ? cTotals : ccTotals;
          var fs = fr[cCls.Value];

          for (int i=0; i<dim; i++)
          {
            var f = (int)data[i];
            fs[i] += f;
            ct[cCls.Value] += f;
          }
        }
      }

      foreach (var cls in classes)
      {
        var fs  = freqs[cls.Value];
        var cfs = cFreqs[cls.Value];
        var ct  = (double)cTotals[cls.Value];
        var cct = (double)ccTotals[cls.Value];
        if (UseSmoothing)
        {
          ct  += alp*dim;
          cct += alp*dim;
        }

        for (int i=0; i<dim; i++)
        {
          var f  = fs[i];
          var cf = cfs[i];
          if (UseSmoothing)
          {
            f  += alp;
            cf += alp;
          }

          cfs[i] = Math.Log(f/ct) - Math.Log(cf/cct);
        }
      }

      return cFreqs;
    }

  }
}
