namespace ML.Contracts
{
  /// <summary>
  /// Simple kernel contract
  /// </summary>
  public interface IKernel : IMnemonicNamed
  {
    /// <summary>
    /// Do calculate kernel value
    /// </summary>
    double Calculate(double r);
  }
}
