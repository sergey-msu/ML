using System;
using System.Collections.Generic;
using System.Linq;
using ML.Core.Contracts;

namespace ML.Core.Algorithms
{
  public abstract class OrderedMetricAlgorithmBase<TParam> : MetricAlgorithmBase<TParam>
  {
    protected OrderedMetricAlgorithmBase(ClassifiedSample classifiedSample, IMetric metric, TParam pars)
      : base(classifiedSample, metric, pars)
    {
    }

    public override float EstimateClose(Point x, Class cls)
    {
      var closeness = 0.0F;
      var sLength = TrainingSample.Count;

      var orderedSample = Metric.Order(x, TrainingSample.Points);

      for (int i=0; i<sLength; i++)
      {
        var sPoint = orderedSample.ElementAt(i);
        var sClass = TrainingSample[sPoint.Key];
        if (!sClass.Equals(cls)) continue;

        closeness += CalculateWeight(i, x, orderedSample);
      }

      return closeness;
    }

    protected abstract float CalculateWeight(int i, Point x, Dictionary<Point, float> orderedSample);
  }
}
