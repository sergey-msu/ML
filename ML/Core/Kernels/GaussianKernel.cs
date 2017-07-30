using System;
using ML.Contracts;

namespace ML.Core.Kernels
{
  /// <summary>
  /// Gaussian kernel r -> 1/sqrt(2*pi)*exp(-r^2/2)
  /// </summary>
  public sealed class GaussianKernel : IKernel
  {
    public const double COEFF = 0.398942280401D;

    public string Name { get { return "GAUSS"; } }


    public double Value(double r)
    {
      return COEFF*Math.Exp(-r*r/2);
    }
  }
}
