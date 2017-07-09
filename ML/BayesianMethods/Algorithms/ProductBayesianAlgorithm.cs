using System;
using System.Collections.Generic;
using System.Linq;
using ML.Core;
using ML.Contracts;
using ML.Utils;

namespace ML.BayesianMethods.Algorithms
{
  /// <summary>
  /// Bayesian non-parametric classification algorithm.
  /// Deals with a probability distributions on classes (not to be confused with Bayesian learning, where probability distributions are considered on algorithm parameters).
  /// If class multiplicative penalties are absent, the algorithm is non-parametric Parzen window implementation of maximum posterior probability (MAP) classification.
  /// Key idea is to model multidimensional distributions as products of Parzen approximation of all features
  /// </summary>
  public class ProductBayesianAlgorithm : BayesianNonparametricAlgorithmBase
  {
    private readonly double[] m_Hs;

    public ProductBayesianAlgorithm(IKernel kernel,
                                    double h = 1,
                                    Dictionary<Class, double> classLosses=null,
                                    double[] hs = null)
      : base(kernel, h, classLosses)
    {
      m_Hs = hs;
    }


    public override string ID { get { return "BAYES"; } }

    public override string Name { get { return "Bayesian non-parametric classification"; } }

    /// <summary>
    /// Window widths
    /// </summary>
    public double[] Hs { get { return m_Hs; } }


    /// <summary>
    /// Classify point
    /// </summary>
    public override Class Predict(double[] obj)
    {
      var pHist = new Dictionary<Class, double>();
      var dim = TrainingSample.GetDimension();

      foreach (var pData in TrainingSample)
      {
        var data = pData.Key;
        var cls  = pData.Value;

        var p = 1.0D;
        for (int i=0; i<dim; i++)
        {
          var h = (m_Hs != null ) ? m_Hs[i] : H;
          var r = (obj[i] - data[i])/h;
          p *= (Kernel.Value(r)/h);
        }

        if (!pHist.ContainsKey(cls)) pHist[cls] = p;
        else pHist[cls] += p;
      }

      if (ClassLosses != null)
      {
        foreach (var score in pHist)
        {
          double penalty;
          if(ClassLosses.TryGetValue(score.Key, out penalty))
            pHist[score.Key] = penalty*score.Value;
        }
      }

      var result = Class.Unknown;
      var max = double.MinValue;
      foreach (var score in pHist)
      {
        if (score.Value > max)
        {
          max = score.Value;
          result = score.Key;
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
      var score = 0.0D;

      foreach (var pData in TrainingSample.Where(d => d.Value.Equals(cls)))
      {
        var data = pData.Key;

        var p = 1.0D;
        for (int i=0; i<dim; i++)
        {
          var h = (m_Hs != null ) ? m_Hs[i] : H;
          var r = (obj[i] - data[i])/h;
          p *= (Kernel.Value(r)/h);
        }

        score += p;
      }

      double penalty;
      if (ClassLosses != null && ClassLosses.TryGetValue(cls, out penalty))
        score *= (penalty / TrainingSample.Count);

      return score;
    }
  }
}
