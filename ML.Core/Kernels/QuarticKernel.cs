using ML.Core.Contracts;

namespace ML.Core.Kernels
{
  /// <summary>
  /// Quartic kernel r -> (1-r^2)^2, [-1, 1]
  /// </summary>
  public sealed class QuarticKernel : IKernel
  {
    public string Name { get { return "Quartic"; } }

    public float Calculate(float r)
    {
      return (r >= -1 && r <= 1) ? (1 - r*r)*(1 - r*r) : 0;
    }
  }
}
