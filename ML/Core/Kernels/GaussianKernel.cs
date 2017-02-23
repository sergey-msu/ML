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

    public double Value(double r)
    {
      return Math.Exp(-r*r);
    }

    public double Derivative(double r)
    {
      return -2.0D * r * Math.Exp(-r*r);
    }

  }
}
