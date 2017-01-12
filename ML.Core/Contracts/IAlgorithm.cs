using System.Collections.Generic;

namespace ML.Core.Contracts
{
  /// <summary>
  /// Classification algorithm contract
  /// </summary>
  public interface IAlgorithm
  {
    /// <summary>
    /// Algorithm mnemonic ID
    /// </summary>
    string ID { get; }

    /// <summary>
    /// Algorithm name
    /// </summary>
    string Name { get; }

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

    /// <summary>
    /// Estimate point closeness to some class
    /// </summary>
    float EstimateClose(Point point, Class cls);
  }
}
