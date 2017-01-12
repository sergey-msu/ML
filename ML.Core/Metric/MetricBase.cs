using System;
using System.Linq;
using System.Collections.Generic;
using ML.Core.Contracts;

namespace ML.Core.Metric
{
  /// <summary>
  /// Represents Euclidead metrics
  /// </summary>
  public abstract class MetricBase : IMetric
  {
    /// <summary>
    /// Distance between two points
    /// </summary>
    public abstract float Dist(Point p1, Point p2);

    /// <summary>
    /// Metric name
    /// </summary>
    public abstract string Name { get; }

    /// <summary>
    /// Sorts some point set with respect to the distance to some fixed point
    /// </summary>
    /// <param name="x">Origin point</param>
    /// <param name="sample">Sample point set</param>
    public Dictionary<Point, float> Sort(Point x, IEnumerable<Point> sample)
    {
      if (sample == null) return null;
      return sample.Select(p => new { Point=p, Dist=Dist(x, p) })
                   .OrderBy(p => p.Dist)
                   .ToDictionary(p => p.Point, p => p.Dist);
    }

  }
}
