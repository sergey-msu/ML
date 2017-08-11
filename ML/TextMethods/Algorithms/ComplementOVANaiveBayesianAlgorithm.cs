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
  public class ComplementOVANaiveBayesianAlgorithm : NaiveBayesianAlgorithmBase
  {
    public ComplementOVANaiveBayesianAlgorithm(ITextPreprocessor preprocessor)
      : base(preprocessor)
    {
    }

    #region Properties

    public override string Name   { get { return "COMPOVALNB"; } }

    #endregion

    protected override Dictionary<ClassFeatureKey, double> TrainWeights()
    {
      var dim      = DataDim;
      var a        = Alpha;
      var ccTotals = new Dictionary<Class, int>();
      var cTotals  = new Dictionary<Class, int>();
      var freqs    = new Dictionary<ClassFeatureKey, double>();
      var cFreqs   = new Dictionary<ClassFeatureKey, double>();
      var classes  = TrainingSample.Classes;

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
          if (!ct.ContainsKey(cCls)) ct[cCls] = 0;

          for (int i=0; i<dim; i++)
          {
            var key = new ClassFeatureKey(cCls, i);

            var f = data[i];
            double freq = fr.TryGetValue(key, out freq) ? (freq+f) : f;
            fr[key] = freq;

            ct[cCls] += (int)f;
          }
        }
      }

      foreach (var key in cFreqs.Keys.ToList())
      {
        var freq    = freqs[key];
        var cFreq   = cFreqs[key];
        var cTotal  = (double)cTotals[key.Class];
        var ccTotal = (double)ccTotals[key.Class];

        if (UseSmoothing)
        {
          freq    += a;
          cTotal  += (a*dim);
          cFreq   += a;
          ccTotal += (a*dim);
        }

        cFreqs[key] = Math.Log(freq/cTotal) - Math.Log(cFreq/ccTotal);
      }

      return cFreqs;
    }

  }
}
