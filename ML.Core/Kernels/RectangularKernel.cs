using System;
using ML.Core.Contracts;

namespace ML.Core.Kernels
{
  public sealed class RectangularKernel : IKernel
  {
    public string Name { get { return "Rectangular"; } }

    public float Calculate(float r)
    {
      return (r >= -1 && r <= 1) ? 1 : 0;
    }
  }
}
