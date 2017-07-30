using ML.Contracts;

namespace ML.Core.Kernels
{
  /// <summary>
  /// Quartic kernel r -> 15/16*(1-r^2)^2, [-1, 1]
  /// </summary>
  public sealed class QuarticKernel : IKernel
  {
    public const double COEFF = 0.9375D;

    public string Name { get { return "QRT"; } }


    public double Value(double r)
    {
      return (r > -1 && r < 1) ? (1 - r*r)*(1 - r*r) : 0;
    }

    public double Derivative(double r)
    {
      return (r > -1 && r < 1) ? COEFF*(r*r-1)*r*4 : 0;
    }

  }
}
