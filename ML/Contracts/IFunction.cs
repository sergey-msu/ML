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

  /// <summary>
  /// Represents 1D kernel: even, positive, L1-normed to 1 real-valued function
  /// </summary>
  public interface IKernel : IMnemonicNamed
  {
    /// <summary>
    /// Calculates kernel value
    /// </summary>
    double Value(double r);
  }

  /// <summary>
  /// Contract for neural networks activation function
  /// </summary>
  public interface IActivationFunction : IFunction
  {
    /// <summary>
    /// Calculates function's derivative value by function value (i.e. y'(y) = y - for y=exp(x))
    /// </summary>
    double DerivativeFromValue(double y);
  }

  /// <summary>
  /// Contract for loss function
  /// </summary>
  public interface ILossFunction
  {
    /// <summary>
    /// Calculates loss value for a given actual and expected input
    /// </summary>
    double Value(double[] actual, double[] expected);

    /// <summary>
    /// Calculates idx-th derivative for a given actual and expected input
    /// </summary>
    double Derivative(int idx, double[] actual, double[] expected);
  }
}
