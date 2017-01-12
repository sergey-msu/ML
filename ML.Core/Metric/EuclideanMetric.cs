using System;
using System.Linq;
using System.Collections.Generic;
using ML.Core.Contracts;

namespace ML.Core.Metric
{
  /// <summary>
  /// Represents Euclidead metrics
  /// </summary>
  public sealed class EuclideanMetric : MetricBase
  {
    /// <summary>
    /// Distance between two points
    /// </summary>
    public override float Dist(Point p1, Point p2)
    {
      Point.CheckDimensions(p1, p2);
      return (float)Math.Sqrt(Dist2(p1, p2));
    }

    /// <summary>
    /// Metric name
    /// </summary>
    public override string Name { get { return "Euclidean"; } }

    /// <summary>
    /// Squared distance between two points
    /// </summary>
    public float Dist2(Point p1, Point p2)
    {
      Point.CheckDimensions(p1, p2);

      var dim = p1.Dimension;
      float sum2 = 0.0F;

      for (int i=0; i<dim; i++)
        sum2 += (p1[i]-p2[i])*(p1[i]-p2[i]);

      return sum2;
    }
  }
}
