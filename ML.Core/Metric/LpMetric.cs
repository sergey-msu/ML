using System;

namespace ML.Core.Metric
{
  /// <summary>
  /// Represents L_p metrics
  /// </summary>
  public sealed class LpMetric : MetricBase
  {
    public LpMetric(float p)
    {
      if (p<1)
        throw new ArgumentException("p should be greaeter or equals that 1 to represent a valid metric");

      P = p;
    }

    /// <summary>
    /// A value of P exponent
    /// </summary>
    public readonly float P;

    /// <summary>
    /// Distance between two points
    /// </summary>
    public override float Dist(Point p1, Point p2)
    {
      Point.CheckDimensions(p1, p2);

      var dim = p1.Dimension;
      double sum = 0.0F;

      for (int i=0; i<dim; i++)
        sum += Math.Pow(Math.Abs(p1[i]-p2[i]), P);

      return (float)Math.Pow(sum, 1.0F / P);
    }

    /// <summary>
    /// Metric name
    /// </summary>
    public override string Name { get { return "L p"; } }
  }
}
