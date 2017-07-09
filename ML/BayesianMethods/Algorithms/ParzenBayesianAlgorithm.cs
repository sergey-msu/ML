using System;
using System.Collections.Generic;
using System.Linq;
using ML.Core;
using ML.Contracts;

namespace ML.BayesianMethods.Algorithms
{
  /// <summary>
  /// Bayesian non-parametric classification algorithm.
  /// Deals with a probability distributions on classes (not to be confused with Bayesian learning, where probability distributions are considered on algorithm parameters).
  /// If class multiplicative penalties are absent, the algorithm is non-parametric Parzen window implementation of maximum posterior probability (MAP) classification
  /// </summary>
  public class ParzenBayesianAlgorithm : BayesianNonparametricAlgorithmBase, IMetricAlgorithm<double[]>
  {
    private readonly IMetric<double[]> m_Metric;

    public ParzenBayesianAlgorithm(IMetric<double[]> metric,
                                   IKernel kernel,
                                   double h = 1,
                                   Dictionary<Class, double> classLosses=null)
      : base(kernel, h, classLosses)
    {
      if (metric == null)
        throw new MLException("BayesianAlgorithm.ctor(metric=null)");

      m_Metric = metric;
    }

    public override string ID { get { return "BAYES"; } }

    public override string Name { get { return "Bayesian non-parametric classification"; } }

    /// <summary>
    /// Space metric
    /// </summary>
    public IMetric<double[]> Metric { get { return m_Metric; } }


    /// <summary>
    /// Classify point
    /// </summary>
    public override Class Predict(double[] obj)
    {
      var hist = new Dictionary<Class, double>();

      foreach (var pData in TrainingSample)
      {
        var r = Metric.Dist(obj, pData.Key) / H;
        var k = Kernel.Value(r);
        var cls = pData.Value;

        if (!hist.ContainsKey(cls)) hist[cls] = k;
        else hist[cls] += k;
      }

      if (ClassLosses != null)
      {
        foreach (var score in hist)
        {
          double penalty;
          if(ClassLosses.TryGetValue(score.Key, out penalty))
            hist[score.Key] = penalty*score.Value;
        }
      }

      var result = Class.Unknown;
      var max = double.MinValue;
      foreach (var score in hist)
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
    /// Estimates closeness of given point to given classes
    /// </summary>
    public override double CalculateClassScore(double[] obj, Class cls)
    {
      var score = 0.0D;
      foreach (var pData in TrainingSample.Where(d => d.Value.Equals(cls)))
      {
        var r = Metric.Dist(pData.Key, obj) / H;
        score += Kernel.Value(r);
      }

      double penalty;
      if (ClassLosses != null && ClassLosses.TryGetValue(cls, out penalty))
        score *= penalty;

      return score;
    }
  }
}
