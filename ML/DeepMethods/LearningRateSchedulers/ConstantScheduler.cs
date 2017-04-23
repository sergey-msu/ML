using System;
using ML.Contracts;
using ML.Core;

namespace ML.DeepMethods.LearningRateSchedulers
{
  /// <summary>
  /// Trivial learning rate scheduler.
  /// Always returns initial learning rate
  /// </summary>
  public class ConstantScheduler : ILearningRateScheduler
  {
    private readonly double m_InitLearningRate;

    public ConstantScheduler(double initLearningRate)
    {
      if (initLearningRate<=0)
        throw new MLException("Initial learning rate value must be positive");

      m_InitLearningRate = initLearningRate;
    }

    public double GetRate(int epoch)
    {
      return m_InitLearningRate;
    }
  }
}
