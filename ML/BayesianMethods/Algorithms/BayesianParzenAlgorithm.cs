using System;
using System.Collections.Generic;
using System.Linq;
using ML.Core;
using ML.Contracts;

namespace ML.BayesianMethods.Algorithms
{
  /// <summary>
  /// Bayesian non-parametric classification algorithm with general multidimensional kernel.
  ///
  /// a(x) = argmax[ ly*P(y)*p(x|y) ]
  /// where p(x|y) = 1/(m*V)*SUMM( K(r(x-xi)/h)/h,, i=1..cnt),
  /// r  - metric function
  /// ly - penalty for error on object of class y
  /// V  - norm constant, supposed to be independent of class, so it can be omitted in argmax
  ///
  /// Deals with a probability distributions on classes (not to be confused with Bayesian learning, where probability distributions are considered on algorithm parameters).
  /// If class multiplicative penalties are absent, the algorithm is non-parametric Parzen window implementation of maximum posterior probability (MAP) classification
  /// </summary>
  public class BayesianParzenAlgorithm : BayesianKernelAlgorithmBase, IMetricAlgorithm<double[]>
  {
    private readonly IMetric<double[]> m_Metric;

    public BayesianParzenAlgorithm(IMetric<double[]> metric,
                                   IKernel kernel,
                                   double h = 1,
                                   Dictionary<Class, double> classLosses=null)
      : base(kernel, h, classLosses)
    {
      if (metric == null)
        throw new MLException("BayesianParzenAlgorithm.ctor(metric=null)");

      m_Metric = metric;
    }

    public override string ID { get { return "BAYES"; } }
    public override string Name { get { return "Bayesian Parzen non-parametric classification with general multidimensional kernel"; } }

    /// <summary>
    /// Space metric
    /// </summary>
    public IMetric<double[]> Metric { get { return m_Metric; } }


    /// <summary>
    /// Classify point
    /// </summary>
    public override Class Predict(double[] obj)
    {
      var classes = DataClasses;
      var pHist = new Dictionary<Class, double>();

      foreach (var pData in TrainingSample)
      {
        var r = Metric.Dist(obj, pData.Key) / H;
        var k = Kernel.Value(r);
        var cls = pData.Value;

        if (!pHist.ContainsKey(cls)) pHist[cls] = k;
        else pHist[cls] += k;
      }

      var result = Class.Unknown;
      var max = double.MinValue;
      foreach (var cls in classes)
      {
        var p = pHist[cls];
        double penalty;
        if (ClassLosses != null && ClassLosses.TryGetValue(cls, out penalty))
          p = penalty*p;

        if (p > max)
        {
          max = p;
          result = cls;
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
