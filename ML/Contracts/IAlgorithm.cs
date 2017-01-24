using ML.Core;
using System.Collections.Generic;

namespace ML.Contracts
{
  /// <summary>
  /// Classification algorithm contract
  /// </summary>
  public interface IAlgorithm : IMnemonicNamed
  {
    /// <summary>
    /// Initial classified training sample
    /// </summary>
    ClassifiedSample TrainingSample { get; }

    /// <summary>
    /// Known classes
    /// </summary>
    Dictionary<string, Class> Classes { get; }

    /// <summary>
    /// Do classify some point
    /// </summary>
    Class Classify(Point point);
  }

  /// <summary>
  /// Contract for general metric classification algorithm
  /// </summary>
  public interface IMetricAlgorithm : IAlgorithm
  {
    /// <summary>
    /// Space metric
    /// </summary>
    IMetric Metric { get; }

    /// <summary>
    /// Estimate point closeness to some class
    /// </summary>
    float EstimateClose(Point point, Class cls);
  }
}
