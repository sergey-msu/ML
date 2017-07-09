using System;
using System.Collections.Generic;
using System.Linq;
using ML.Core;
using ML.Contracts;
using ML.Utils;

namespace ML.BayesianMethods.Algorithms
{
  /// <summary>
  /// Naive Bayesian non-parametric classification algorithm.
  /// Deals with a probability distributions on classes (not to be confused with Bayesian learning, where probability distributions are considered on algorithm parameters)
  /// in a special case of independent (as random variables) features.
  /// If class multiplicative penalties are absent, the algorithm is non-parametric Parzen window implementation of Maximum posterior probability (MAP) classification
  /// </summary>
  public class NaiveBayesianAlgorithm : BayesianNonparametricAlgorithmBase
  {
    public NaiveBayesianAlgorithm(IKernel kernel,
                                  double h = 1,
                                  Dictionary<Class, double> classLosses=null)
      : base(kernel, h, classLosses)
    {
    }

    public override string ID { get { return "NBAYES"; } }

    public override string Name { get { return "Naive Bayesian non-parametric classification"; } }


    /// <summary>
    /// Classify point
    /// </summary>
    public override Class Predict(double[] obj)
    {
      var dim     = TrainingSample.GetDimension();
      var classes = TrainingSample.CachedClasses;

      var lHist = new Dictionary<Class, int>();
      var pHist = new Dictionary<Class, double>();
      var yHist = new Dictionary<Class, double>();
      foreach (var cls in classes)
      {
        lHist[cls] = 0;
        pHist[cls] = 0.0D;
        yHist[cls] = 0.0D;
      }

      for (int i=0; i<dim; i++)
      {
        foreach (var pData in TrainingSample)
        {
          var data = pData.Key;
          var cls  = pData.Value;

          if (i==0) lHist[cls] += 1;

          var r = (obj[i] - data[i])/H;
          pHist[cls] += Kernel.Value(r);
        }

        foreach (var cls in classes)
        {
          yHist[cls] += Math.Log(pHist[cls] / (H * lHist[cls]));
          pHist[cls] = 0.0D;
        }
      }

      foreach (var cls in classes)
      {
        var ly = (ClassLosses == null) ? 1.0D : ClassLosses[cls];
        yHist[cls] += Math.Log(lHist[cls]*ly / TrainingSample.Count);
      }

      var max = double.MinValue;
      var result = Class.Unknown;
      foreach (var cls in classes)
      {
        var prob = yHist[cls];
        if (prob > max)
        {
          max = prob;
          result = cls;
        }
      }

      return result;
    }

    /// <summary>
    /// Estimated closeness of given point to given classes
    /// </summary>
    public override double CalculateClassScore(double[] obj, Class cls)
    {
      var dim = TrainingSample.GetDimension();
      var my = 0;
      var p = 0.0D;
      var y = 0.0D;

      for (int i=0; i<dim; i++)
      {
        foreach (var pData in TrainingSample.Where(d => d.Value.Equals(cls)))
        {
          var data = pData.Key;

          if (i==0) my += 1;

          var r = (obj[i] - pData.Key[i])/H;
          p += Kernel.Value(r);
        }

        y += Math.Log(p / (H * my));
        p = 0.0D;
      }

      double penalty;
      if (ClassLosses != null && ClassLosses.TryGetValue(cls, out penalty))
        y += Math.Log(my*penalty / TrainingSample.Count);

      return y;
    }
  }
}
