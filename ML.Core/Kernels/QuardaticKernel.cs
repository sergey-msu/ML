using ML.Core.Contracts;

namespace ML.Core.Kernels
{
  /// <summary>
  /// Quardatic kernel r -> r^2, [-1, 1]
  /// </summary>
  public sealed class QuardaticKernel : IKernel
  {
    public string Name { get { return "Quardatic"; } }

    public float Calculate(float r)
    {
      return (r >= -1 && r <= 1) ? (1 - r*r) : 0;
    }
  }
}
