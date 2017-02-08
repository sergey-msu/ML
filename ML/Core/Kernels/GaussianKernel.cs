using System;
using ML.Contracts;

namespace ML.Core.Kernels
{
  /// <summary>
  /// Gaussian kernel r -> exp(-r^2)
  /// </summary>
  public sealed class GaussianKernel : IFunction
  {
    public string ID { get { return "GAUSS"; } }
    public string Name { get { return "Gaussian"; } }

    public double Calculate(double r)
    {
      return (double)Math.Exp(-r*r);
    }
  }
}
