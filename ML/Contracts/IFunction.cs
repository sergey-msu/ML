namespace ML.Contracts
{
  /// <summary>
  /// Simple function contract (i.e. activation function for neural networks, kernels etc.)
  /// </summary>
  public interface IFunction : IMnemonicNamed
  {
    /// <summary>
    /// Do calculate function value
    /// </summary>
    double Value(double r);

    /// <summary>
    /// Do calculate function's derivative value
    /// </summary>
    double Derivative(double r);
  }
}
