using System;
using System.Collections.Generic;
using System.Linq;
using ML.Core.Contracts;

namespace ML.Core.Algorithms
{
  /// <summary>
  /// Base class for algorithm relied on traing samople order with respect to fixed test point
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

      for (int i=0; i<sLength; i++)
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
}
