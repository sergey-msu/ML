using System;

namespace ML.Contracts
{
  /// <summary>
  /// Contract for learning rate scheduler
  /// </summary>
  public interface ILearningRateScheduler
  {
    /// <summary>
    /// Returns rate to be used in current epoch
    /// </summary>
    double GetRate(int epoch);
  }
}
