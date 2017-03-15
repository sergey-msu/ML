using ML.Core.Mathematics;
using System;

namespace ML.Core.Metric
{
  /// <summary>
  /// Represents L_p metrics
  /// </summary>
  public sealed class LpMetric : MetricBase
  {
    public override string ID { get { return "LP"; } }
    public override string Name { get { return "L p"; } }

    public LpMetric(double p)
    {
      if (p<1)
        throw new MLException("p should be greaeter or equals that 1 to represent a valid metric");

      P = p;
    }

    /// <summary>
    /// A value of P exponent
    /// </summary>
    public readonly double P;

    /// <summary>
    /// Distance between two points
    /// </summary>
    public override double Dist(double[] p1, double[] p2)
    {
      MathUtils.CheckDimensions(p1, p2);

      var dim = p1.Length;
      double sum = 0.0F;

      for (int i=0; i<dim; i++)
        sum += Math.Pow(Math.Abs(p1[i]-p2[i]), P);

      return Math.Pow(sum, 1.0F / P);
    }
  }
}
