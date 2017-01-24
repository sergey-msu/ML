using ML.Contracts;

namespace ML.Core.Kernels
{
  /// <summary>
  /// Quardatic kernel r -> r^2, [-1, 1]
  /// </summary>
  public sealed class QuadraticKernel : IKernel
  {
    public string ID { get { return "QDR"; } }
    public string Name { get { return "Quadratic"; } }

    public float Calculate(float r)
    {
      return (r >= -1 && r <= 1) ? (1 - r*r) : 0;
    }
  }
}
