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
  public abstract class MetricAlgorithmBase<TObj> : AlgorithmBase<TObj>, IMetricAlgorithm<TObj>
  {
    public readonly IMetric m_Metric;

    protected MetricAlgorithmBase(ClassifiedSample<TObj> classifiedSample, IMetric metric)
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
    public override Class Classify(TObj obj)
    {
      Class result = null;
      var maxEst = double.MinValue;

      foreach (var cls in Classes.Values)
      {
        var est = EstimateProximity(obj, cls);
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
    public abstract double EstimateProximity(TObj obj, Class cls);

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
          var closeness = EstimateProximity(pData.Key, cls);
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

    public double EstimateClose(object obj, Class cls)
    {
      return EstimateClose((double[])obj, cls);
    }
  }

  /// <summary>
  /// Base class for metric algorithm that relies on order of training sample with respect to fixed test point
  /// </summary>
  public abstract class OrderedMetricAlgorithmBase : MetricAlgorithmBase<double[]>
  {
    protected OrderedMetricAlgorithmBase(ClassifiedSample<double[]> classifiedSample, IMetric metric)
      : base(classifiedSample, metric)
    {
    }

    /// <summary>
    /// Estimated closeness of given point to given classes
    /// </summary>
    public override double EstimateProximity(double[] obj, Class cls)
    {
      var closeness = 0.0D;
      var sLength = TrainingSample.Count;

      var orderedSample = Metric.Sort(obj, TrainingSample.Objects);

      for (int i = 0; i < sLength; i++)
      {
        var sPoint = orderedSample.ElementAt(i);
        var sClass = TrainingSample[sPoint.Key];
        if (!sClass.Equals(cls)) continue;

        closeness += CalculateWeight(i, obj, orderedSample);
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
    protected abstract double CalculateWeight(int i, double[] x, Dictionary<double[], double> orderedSample);
  }

  /// <summary>
  /// Base class for metric algorithm supplied with some kernel function
  /// </summary>
  public abstract class KernelAlgorithmBase : OrderedMetricAlgorithmBase
  {
    private readonly IFunction m_Kernel;

    public KernelAlgorithmBase(ClassifiedSample<double[]> classifiedSample,
                               IMetric metric,
                               IFunction kernel)
      : base(classifiedSample, metric)
    {
      if (kernel == null)
        throw new MLException("KernelAlgorithmBase.ctor(kernel=null)");

      m_Kernel = kernel;
    }

    public IFunction Kernel { get { return m_Kernel; } }
  }
}
