using System;
using System.Collections.Generic;
using System.Linq;
using ML.Contracts;

namespace ML.Core.Algorithms
{
  /// <summary>
  /// Base class for metric algorithm supplied with some spacial metric
  /// </summary>
  public abstract class MetricAlgorithmBase<TParam> : AlgorithmBase<TParam>, IMetricAlgorithm
  {
    public readonly IMetric m_Metric;

    protected MetricAlgorithmBase(ClassifiedSample classifiedSample,
                                  IMetric metric,
                                  TParam pars)
      : base(classifiedSample, pars)
    {
      if (metric == null)
        throw new ArgumentException("MetricAlgorithmBase.ctor(metric=null)");

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
      var maxEst = float.MinValue;

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
    public abstract float EstimateClose(Point point, Class cls);
  }

  /// <summary>
  /// Base class for metric algorithm that relies on order of training sample with respect to fixed test point
  /// </summary>
  public abstract class OrderedMetricAlgorithmBase<TParam> : MetricAlgorithmBase<TParam>
  {
    protected OrderedMetricAlgorithmBase(ClassifiedSample classifiedSample, IMetric metric, TParam pars)
      : base(classifiedSample, metric, pars)
    {
    }

    /// <summary>
    /// Estimated closeness of given point to given classes
    /// </summary>
    public override float EstimateClose(Point x, Class cls)
    {
      var closeness = 0.0F;
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
    protected abstract float CalculateWeight(int i, Point x, Dictionary<Point, float> orderedSample);
  }

  /// <summary>
  /// Base class for metric algorithm supplied with some kernel function
  /// </summary>
  public abstract class KernelAlgorithmBase<TParam> : OrderedMetricAlgorithmBase<TParam>
  {
    private readonly IKernel m_Kernel;

    public KernelAlgorithmBase(ClassifiedSample classifiedSample,
                               IMetric metric,
                               IKernel kernel,
                               TParam pars)
      : base(classifiedSample, metric, pars)
    {
      if (kernel == null)
        throw new ArgumentException("KernelAlgorithmBase.ctor(kernel=null)");

      m_Kernel = kernel;
    }

    public IKernel Kernel { get { return m_Kernel; } }
  }
}
