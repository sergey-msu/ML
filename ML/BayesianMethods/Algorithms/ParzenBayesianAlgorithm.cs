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
  public class ParzenBayesianAlgorithm : ClassificationAlgorithmBase<double[]>, IKernelAlgorithm<double[]>, IMetricAlgorithm<double[]>
  {
    private readonly IMetric<double[]> m_Metric;
    private readonly IKernel m_Kernel;
    private readonly Dictionary<Class, double> m_ClassLosses;
    private double m_H;

    public ParzenBayesianAlgorithm(IMetric<double[]> metric,
                                   IKernel kernel,
                                   double h = 1,
                                   Dictionary<Class, double> classLosses=null)
    {
      if (metric == null)
        throw new MLException("BayesianAlgorithm.ctor(metric=null)");
      if (kernel == null)
        throw new MLException("BayesianAlgorithm.ctor(kernel=null)");

      m_Metric = metric;
      m_Kernel = kernel;
      m_ClassLosses = classLosses;
    }

    public override string ID { get { return "BAYES"; } }

    public override string Name { get { return "Bayesian non-parametric classification"; } }

    /// <summary>
    /// Space metric
    /// </summary>
    public IMetric<double[]> Metric { get { return m_Metric; } }

    /// <summary>
    /// Kernel function
    /// </summary>
    public IKernel Kernel { get { return m_Kernel; } }

    /// <summary>
    /// Additional multiplicative penalty to wrong object classification.
    /// If null, all class penalties dafault to 1 (no special effect on classification - pure MAP classification)
    /// </summary>
    public Dictionary<Class, double> ClassLosses { get { return m_ClassLosses; } }

    /// <summary>
    /// Window width
    /// </summary>
    public double H
    {
     get { return m_H; }
     set
     {
       if (value <= double.Epsilon)
         throw new MLException("BayesianAlgorithm.H(value<=0)");

       m_H = value;
     }
    }


    /// <summary>
    /// Classify point
    /// </summary>
    public override Class Predict(double[] obj)
    {
      var hist = new Dictionary<Class, double>();

      foreach (var pData in TrainingSample)
      {
        var r = Metric.Dist(obj, pData.Key) / m_H;
        var k = Kernel.Value(r);
        var cls = pData.Value;

        if (!hist.ContainsKey(cls)) hist[cls] = k;
        else hist[cls] += k;
      }

      if (m_ClassLosses != null)
      {
        foreach (var score in hist)
        {
          double penalty;
          if(m_ClassLosses.TryGetValue(score.Key, out penalty))
            hist[score.Key] = penalty*score.Value;
        }
      }

      Class result = null;
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
    public double CalculateClassScore(double[] obj, Class cls)
    {
      var score = 0.0D;
      foreach (var pData in TrainingSample.Where(d => d.Value.Equals(cls)))
      {
        var r = Metric.Dist(pData.Key, obj) / m_H;
        score += Kernel.Value(r);
      }

      double penalty;
      if (m_ClassLosses != null && m_ClassLosses.TryGetValue(cls, out penalty))
        score *= penalty;

      return score;
    }

    protected override void DoTrain()
    {
      // Nonparametric Bayesian methods are not trainable by default
    }
  }
}
