using ML.Core.Mathematics;
using System;

namespace ML.Core.Metric
{
  /// <summary>
  /// Represents L_infty metrics
  /// </summary>
  public sealed class LInftyMetric : MetricBase
  {
    public override string ID { get { return "LINFTY"; } }
    public override string Name { get { return "L infinity"; } }

    /// <summary>
    /// Distance between two points
    /// </summary>
    public override double Dist(double[] p1, double[] p2)
    {
      MathUtils.CheckDimensions(p1, p2);

      var dim = p1.Length;
      var max = double.MinValue;

      for (int i=0; i<dim; i++)
      {
        var abs = Math.Abs(p1[i]-p2[i]);
        if (abs > max) max = abs;
      }

      return max;
    }
  }
}
