using System;
using System.Linq;
using System.Collections.Generic;
using ML.Contracts;

namespace ML.Core.Metric
{
  /// <summary>
  /// Represents base class for metric
  /// </summary>
  public abstract class MetricBase<TObj> : IMetric<TObj>
  {
    /// <summary>
    /// Metric mnemonic ID
    /// </summary>
    public abstract string Name { get; }

    /// <summary>
    /// Distance between two points
    /// </summary>
    public abstract double Dist(TObj p1, TObj p2);

    /// <summary>
    /// Sorts some point set with respect to the distance to some fixed point
    /// </summary>
    /// <param name="x">Origin point</param>
    /// <param name="sample">Sample point set</param>
    public Dictionary<TObj, double> Sort(TObj x, IEnumerable<TObj> sample)
    {
      if (sample == null) return null;
      return sample.Select(p => new { Point=p, Dist=Dist(x, p) })
                   .OrderBy(p => p.Dist)
                   .ToDictionary(p => p.Point, p => p.Dist);
    }

  }
}
