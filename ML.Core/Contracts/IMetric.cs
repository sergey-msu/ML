using System.Collections.Generic;

namespace ML.Core.Contracts
{
  public interface IMetric
  {
    string Name { get; }

    float Dist(Point p1, Point p2);

    Dictionary<Point, float> Order(Point x, IEnumerable<Point> sample);
  }
}
