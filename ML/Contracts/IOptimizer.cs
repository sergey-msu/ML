namespace ML.Contracts
{
  /// <summary>
  /// Contract for a Convnet gradient descent optimizer
  /// </summary>
  public interface IOptimizer
  {
    /// <summary>
    /// Last weight vector step value (squared)
    /// </summary>
    double Step2 { get; }

    /// <summary>
    /// Optimize current updates and apply it to source weight vector
    /// </summary>
    void Push(double[][] weights, double[][] gradient, double learningRate);
  }
}
