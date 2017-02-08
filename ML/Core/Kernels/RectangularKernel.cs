using System;
using ML.Contracts;

namespace ML.Core.Kernels
{
  /// <summary>
  /// Rectangular kernel r -> 1, [-1, 1]
  /// </summary>
  public sealed class RectangularKernel : IFunction
  {
    public string ID { get { return "RECT"; } }
    public string Name { get { return "Rectangular"; } }

    public double Calculate(double r)
    {
      return (r > -1 && r < 1) ? 1 : 0;
    }
  }
}
