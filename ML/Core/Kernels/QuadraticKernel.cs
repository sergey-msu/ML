using ML.Contracts;

namespace ML.Core.Kernels
{
  /// <summary>
  /// Quardatic (Optimal Epanechnikov) kernel r -> 3/4*(1-r^2), [-1, 1]
  /// </summary>
  public sealed class QuadraticKernel : IKernel
  {
    public const double COEFF = 0.75D;

    public string Name { get { return "QDR"; } }


    public double Value(double r)
    {
      return (r > -1 && r < 1) ? COEFF*(1 - r*r) : 0;
    }
  }
}
