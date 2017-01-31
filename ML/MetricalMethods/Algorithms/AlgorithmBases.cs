using System;
using System.Collections.Generic;
using System.Linq;
using ML.Contracts;
using ML.Core;

namespace ML.MetricalMethods.Algorithms
{
  /// <summary>
  /// Base class for metric algorithm supplied with some spacial metric
  /// </summary>
  public abstract class MetricAlgorithmBase : AlgorithmBase, IMetricAlgorithm
  {
    public readonly IMetric m_Metric;

    protected MetricAlgorithmBase(ClassifiedSample classifiedSample,
                                  IMetric metric)
      : base(classifiedSample)
    {
      if (metric == null)
        throw new MLException("MetricAlgorithmBase.ctor(metric=null)");

      m_Metric = metric;
    }

    /// <summary>
    /// Space metric
    /// </summary>
    public IMetric Metric { get { return m_Metric; } }

    /// <summary>
    /// Classify point
    /// </summary>
    public override Class Classify(Point x)
    {
      Class result = null;
      var maxEst = double.MinValue;

      foreach (var cls in Classes.Values)
      {
        var est = EstimateClose(x, cls);
        if (est > maxEst)
        {
          maxEst = est;
          result = cls;
        }
      }

      return result;
    }

    /// <summary>
    /// Estimated closeness of given point to given classes
    /// </summary>
    public abstract double EstimateClose(Point point, Class cls);

    /// <summary>
    /// Calculates margins
    /// </summary>
    public Dictionary<int, double> CalculateMargins()
    {
      var result = new SortedDictionary<int, double>();
      int idx = -1;

      foreach (var pData in TrainingSample)
      {
        idx++;
        double maxi = double.MinValue;
        double si = 0;

        foreach (var cls in Classes.Values)
        {
          var closeness = EstimateClose(pData.Key, cls);
          if (cls == pData.Value) si = closeness;
          else
          {
            if (maxi < closeness) maxi = closeness;
          }
        }

        result.Add(idx, si - maxi);
      }

      return result.OrderBy(r => r.Value).ToDictionary(r => r.Key, r => r.Value);
    }
  }

  /// <summary>
  /// Base class for metric algorithm that relies on order of training sample with respect to fixed test point
  /// </summary>
  public abstract class OrderedMetricAlgorithmBase : MetricAlgorithmBase
  {
    protected OrderedMetricAlgorithmBase(ClassifiedSample classifiedSample, IMetric metric)
      : base(classifiedSample, metric)
    {
    }

    /// <summary>
    /// Estimated closeness of given point to given classes
    /// </summary>
    public override double EstimateClose(Point x, Class cls)
    {
      var closeness = 0.0D;
      var sLength = TrainingSample.Count;

      var orderedSample = Metric.Sort(x, TrainingSample.Points);

      for (int i = 0; i < sLength; i++)
      {
        var sPoint = orderedSample.ElementAt(i);
        var sClass = TrainingSample[sPoint.Key];
        if (!sClass.Equals(cls)) continue;

        closeness += CalculateWeight(i, x, orderedSample);
      }

      return closeness;
    }

    /// <summary>
    /// Calculate 'weight' - a contribution of training point (i-th from ordered training sample)
    /// to closeness of test point to its class
    /// </summary>
    /// <param name="i">Point number in ordered training sample</param>
    /// <param name="x">Test point</param>
    /// <param name="orderedSample">Ordered training sample</param>
    protected abstract double CalculateWeight(int i, Point x, Dictionary<Point, double> orderedSample);
  }

  /// <summary>
  /// Base class for metric algorithm supplied with some kernel function
  /// </summary>
  public abstract class KernelAlgorithmBase : OrderedMetricAlgorithmBase
  {
    private readonly IKernel m_Kernel;

    public KernelAlgorithmBase(ClassifiedSample classifiedSample,
                               IMetric metric,
                               IKernel kernel)
      : base(classifiedSample, metric)
    {
      if (kernel == null)
        throw new MLException("KernelAlgorithmBase.ctor(kernel=null)");

      m_Kernel = kernel;
    }

    public IKernel Kernel { get { return m_Kernel; } }
  }
}
