using System;
using ML.Contracts;
using ML.Core;

namespace ML.DeepMethods.LearningRateSchedulers
{
  /// <summary>
  /// Time-based learning rate scheduler.
  /// Decreases initial learning rate by factor 1/(1+decay*epoch) - polynomial decay of learning rate
  /// </summary>
  public class TimeBasedScheduler : ILearningRateScheduler
  {
    private readonly double m_InitLearningRate;
    private readonly double m_Decay;

    public TimeBasedScheduler(double initLearningRate, double decay)
    {
      if (initLearningRate<=0)
        throw new MLException("Initial learning rate value must be positive");
      if (decay<0)
        throw new MLException("Decay value must be positive");

      m_InitLearningRate = initLearningRate;
      m_Decay = decay;
    }

    public double GetRate(int epoch)
    {
      return m_InitLearningRate / (1 + m_Decay*epoch);
    }
  }
}
