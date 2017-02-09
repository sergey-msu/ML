using ML.Contracts;

namespace ML.Core.Kernels
{
  /// <summary>
  /// Quartic kernel r -> (1-r^2)^2, [-1, 1]
  /// </summary>
  public sealed class QuarticKernel : IFunction
  {
    public string ID { get { return "QRT"; } }
    public string Name { get { return "Quartic"; } }

    public double Value(double r)
    {
      return (r > -1 && r < 1) ? (1 - r*r)*(1 - r*r) : 0;
    }

    public double Derivative(double r)
    {
      return (r > -1 && r < 1) ? (r*r-1)*r*4 : 0;
    }

  }
}
