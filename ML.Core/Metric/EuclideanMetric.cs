using System;
using System.Linq;
using System.Collections.Generic;
using ML.Core.Contracts;

namespace ML.Core.Metric
{
  public sealed class EuclideanMetric : IMetric
  {
    public float Dist(Point p1, Point p2)
    {
      Point.CheckDimensions(p1, p2);
      return (float)Math.Sqrt(Dist2(p1, p2));
    }

    public string Name { get { return "Euclidean"; } }

    public Dictionary<Point, float> Order(Point x, IEnumerable<Point> sample)
    {
      if (sample == null) return null;
      return sample.Select(p => new { Point=p, Dist=Dist(x, p) })
                   .OrderBy(p => p.Dist)
                   .ToDictionary(p => p.Point, p => p.Dist);
    }

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
