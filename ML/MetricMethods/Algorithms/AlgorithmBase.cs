using System;
using System.Collections.Generic;
using System.Linq;
using ML.Contracts;
using ML.Core;

namespace ML.MetricMethods.Algorithms
{
  /// <summary>
  /// Base class for metric algorithm supplied with some spacial metric
  /// </summary>
  public abstract class MetricAlgorithmBase<TObj> : ClassificationAlgorithmBase<TObj>, IMetricAlgorithm<TObj>
  {
    private readonly IMetric<TObj> m_Metric;

    protected MetricAlgorithmBase(IMetric<TObj> metric)
      : base()
    {
      if (metric == null)
        throw new MLException("MetricAlgorithmBase.ctor(metric=null)");

      m_Metric = metric;
    }

    /// <summary>
    /// Space metric
    /// </summary>
    public IMetric<TObj> Metric { get { return m_Metric; } }

    /// <summary>
    /// Estimate value of Γ(obj, cls) - object proximity to some class
    /// </summary>
    public abstract double CalculateClassScore(TObj obj, Class cls);

    /// <summary>
    /// Classify point
    /// </summary>
    public override ClassScore[] PredictTokens(TObj obj, int cnt)
    {
      var scores = new List<ClassScore>();

      foreach (var cls in TrainingSample.Classes)
      {
        var score = CalculateClassScore(obj, cls);
        scores.Add(new ClassScore(cls, score));
      }

      return scores.OrderByDescending(s => s.Score)
                   .Take(cnt)
                   .ToArray();
    }
  }

  /// <summary>
  /// Base class for metric algorithm that relies on order of training sample with respect to fixed test point
  /// </summary>
  public abstract class OrderedMetricAlgorithmBase<TObj> : MetricAlgorithmBase<TObj>
  {
    protected OrderedMetricAlgorithmBase(IMetric<TObj> metric)
      : base(metric)
    {
    }

    /// <summary>
    /// Estimated closeness of given point to given classes
    /// </summary>
    public override double CalculateClassScore(TObj obj, Class cls)
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
    protected abstract double CalculateWeight(int i, TObj x, Dictionary<TObj, double> orderedSample);
  }
}
