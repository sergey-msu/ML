using System;
using ML.Contracts;

namespace ML.Core.Kernels
{
  /// <summary>
  /// Gaussian kernel r -> exp(-r^2)
  /// </summary>
  public sealed class GaussianKernel : IKernel
  {
    public string ID { get { return "GAUSS"; } }
    public string Name { get { return "Gaussian"; } }

    public float Calculate(float r)
    {
      return (float)Math.Exp(-r*r);
    }
  }
}
