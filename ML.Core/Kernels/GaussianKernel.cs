using System;
using ML.Core.Contracts;

namespace ML.Core.Kernels
{
  /// <summary>
  /// Gaussian kernel r -> exp(-r^2)
  /// </summary>
  public sealed class GaussianKernel : IKernel
  {
    public string Name { get { return "Gaussian"; } }

    public float Calculate(float r)
    {
      return (float)Math.Exp(-2*r*r);
    }
  }
}
