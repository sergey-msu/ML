using System.Collections.Generic;
using ML.Core;

namespace ML.Contracts
{
  /// <summary>
  /// Metric contract
  /// </summary>
  public interface IMetric : IMnemonicNamed
  {
    /// <summary>
    /// Distance between two points
    /// </summary>
    float Dist(Point p1, Point p2);

    /// <summary>
    /// Sorts some point set with respect to the distance to some fixed point
    /// </summary>
    /// <param name="x">Origin point</param>
    /// <param name="sample">Sample point set</param>
    Dictionary<Point, float> Sort(Point x, IEnumerable<Point> sample);
  }
}
