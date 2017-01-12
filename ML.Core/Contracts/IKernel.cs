namespace ML.Core.Contracts
{
  /// <summary>
  /// Simple kernel contract
  /// </summary>
  public interface IKernel
  {
    /// <summary>
    /// Kernel name
    /// </summary>
    string Name { get; }

    /// <summary>
    /// Do calculate kernel value
    /// </summary>
    float Calculate(float r);
  }
}
