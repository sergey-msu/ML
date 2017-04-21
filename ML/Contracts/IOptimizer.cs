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
    /// Set source weight vector
    /// </summary>
    void Init(double[][] weights);

    /// <summary>
    /// Optimize current updates and apply it to source weight vector
    /// </summary>
    void Push(double[][] gradient, double learningRate);
  }
}
