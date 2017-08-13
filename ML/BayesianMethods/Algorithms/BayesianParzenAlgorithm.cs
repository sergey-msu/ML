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
                                   double[] classLosses=null)
      : base(kernel, h, classLosses)
    {
      if (metric == null)
        throw new MLException("BayesianParzenAlgorithm.ctor(metric=null)");

      m_Metric = metric;
    }

    public override string Name { get { return "BAYES"; } }

    /// <summary>
    /// Space metric
    /// </summary>
    public IMetric<double[]> Metric { get { return m_Metric; } }


    /// <summary>
    /// Classify point
    /// </summary>
    public override ClassScore[] PredictTokens(double[] obj, int cnt)
    {
      var classes = Classes;
      var priors  = PriorProbs;
      var pHist = new Dictionary<Class, double>();

      foreach (var pData in TrainingSample)
      {
        var r = Metric.Dist(obj, pData.Key) / H;
        var k = Kernel.Value(r);
        var cls = pData.Value;

        if (!pHist.ContainsKey(cls)) pHist[cls] = k;
        else pHist[cls] += k;
      }

      var scores = new List<ClassScore>();
      foreach (var cls in classes)
      {
        var p = Math.Log(pHist[cls]) + priors[cls.Value];
        scores.Add(new ClassScore(cls, p));
      }

      return scores.OrderByDescending(s => s.Score)
                   .Take(cnt)
                   .ToArray();
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

      score = Math.Log(score) + PriorProbs[cls.Value];

      return score;
    }
  }
}
