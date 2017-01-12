using System;
using ML.Core.Contracts;

namespace ML.Core.Kernels
{
  /// <summary>
  /// Triangular kernel r -> 1-|r|, [-1, 1]
  /// </summary>
  public sealed class TriangularKernel : IKernel
  {
    public string Name { get { return "Triangular"; } }

    public float Calculate(float r)
    {
      return (r >= -1 && r <= 1) ? (1 - Math.Abs(r)) : 0;
    }
  }
}
