using System.Collections.Generic;
using ML.Core;

namespace ML.Contracts
{
  /// <summary>
  /// Metric contract
  /// </summary>
  public interface IMetric<TObj> : INamed
  {
    /// <summary>
    /// Distance between two points
    /// </summary>
    double Dist(TObj p1, TObj p2);

    /// <summary>
    /// Sorts some point set with respect to the distance to some fixed point
    /// </summary>
    /// <param name="x">Origin point</param>
    /// <param name="sample">Sample point set</param>
    Dictionary<TObj, double> Sort(TObj x, IEnumerable<TObj> sample);
  }
}
