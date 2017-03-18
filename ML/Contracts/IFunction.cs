namespace ML.Contracts
{
  /// <summary>
  /// Simple function contract (i.e. activation function for neural networks, kernels etc.)
  /// </summary>
  public interface IFunction : IMnemonicNamed
  {
    /// <summary>
    /// Calculates function value
    /// </summary>
    double Value(double r);

    /// <summary>
    /// Calculates function's derivative value
    /// </summary>
    double Derivative(double r);
  }

  public interface IActivationFunction : IFunction
  {
    /// <summary>
    /// Calculates function's derivative value by function value (i.e. y'(y) = y - for y=exp(x))
    /// </summary>
    double DerivativeFromValue(double y);
  }
}
