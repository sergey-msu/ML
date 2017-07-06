using System;
using ML.Contracts;

namespace ML.Core.Kernels
{
  /// <summary>
  /// Rectangular kernel r -> 1, [-1, 1]
  /// </summary>
  public sealed class RectangularKernel : IKernel
  {
    public const double COEFF = 0.5D;

    public string ID { get { return "RECT"; } }
    public string Name { get { return "Rectangular"; } }

    public double Value(double r)
    {
      return (r > -1 && r < 1) ? COEFF : 0;
    }

  }
}
