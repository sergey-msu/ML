using ML.Core;
using System.Collections.Generic;

namespace ML.Contracts
{
  /// <summary>
  /// Contract of classification algorithm with training sample
  /// </summary>
  public interface ISupervisedAlgorithm : IMnemonicNamed
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
  public interface IMetricAlgorithm : ISupervisedAlgorithm
  {
    /// <summary>
    /// Space metric
    /// </summary>
    IMetric Metric { get; }

    /// <summary>
    /// Estimate point closeness to some class
    /// </summary>
    double EstimateClose(Point point, Class cls);

    /// <summary>
    /// Calculates margins
    /// </summary>
    Dictionary<int, double> CalculateMargins();
  }
}
