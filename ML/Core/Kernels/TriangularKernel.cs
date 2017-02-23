using System;
using ML.Contracts;

namespace ML.Core.Kernels
{
  /// <summary>
  /// Triangular kernel r -> 1-|r|, [-1, 1]
  /// </summary>
  public sealed class TriangularKernel : IFunction
  {
    public string ID { get { return "TRN"; } }
    public string Name { get { return "Triangular"; } }

    public double Value(double r)
    {
      return (r > -1 && r < 1) ? (1 - Math.Abs(r)) : 0;
    }

    public double Derivative(double r)
    {
      return (r > -1 && r < 1) ? (r > 0 ? -1 : 1) : 0;
    }

  }
}
