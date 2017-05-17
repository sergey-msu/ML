using ML.Core;
using System.Collections.Generic;

namespace ML.Contracts
{
  /// <summary>
  /// Contract of classification algorithm with training sample
  /// </summary>
  public interface ISupervisedAlgorithm<TObj> : IMnemonicNamed
  {
    /// <summary>
    /// Initial classified training sample
    /// </summary>
    ClassifiedSample<TObj> TrainingSample { get; }

    /// <summary>
    /// Known classes
    /// </summary>
    Dictionary<string, Class> Classes { get; }

    /// <summary>
    /// Do classify some object
    /// </summary>
    Class Classify(TObj obj);
  }

  /// <summary>
  /// Contract for general metric classification algorithm
  /// </summary>
  public interface IMetricAlgorithm<TObj> : ISupervisedAlgorithm<TObj>
  {
    /// <summary>
    /// Space metric
    /// </summary>
    IMetric Metric { get; }

    /// <summary>
    /// Estimate point closeness to some class
    /// </summary>
    double EstimateProximity(TObj obj, Class cls);

    /// <summary>
    /// Calculates margins
    /// </summary>
    Dictionary<int, double> CalculateMargins();
  }
}
