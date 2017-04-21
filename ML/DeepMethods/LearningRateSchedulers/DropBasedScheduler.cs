using System;
using ML.Contracts;
using ML.Core;

namespace ML.DeepMethods.LearningRateSchedulers
{
  /// <summary>
  /// Drop-based learning rate scheduler.
  /// Decreases initial learning rate by some factor every n-th epoch - exponential decay of learning rate
  /// </summary>
  public class DropBasedScheduler : ILearningRateScheduler
  {
    private double m_CurrentLearningRate;
    private readonly double m_EpochStep;
    private readonly double m_DropRate;

    public DropBasedScheduler(double initLearningRate, double epochStep, double dropRate)
    {
      if (initLearningRate<=0)
        throw new MLException("Initial earning rate value must be positive");
      if (epochStep<0)
        throw new MLException("Epoch step value must be positive");
      if (dropRate<=0 || dropRate>=1)
        throw new MLException("Drop rate value must be in (0,1)");

      m_CurrentLearningRate = initLearningRate;
      m_EpochStep = epochStep;
      m_DropRate = dropRate;
    }

    public double GetRate(int epoch)
    {
      if (epoch % m_EpochStep == 0)
        m_CurrentLearningRate *= m_DropRate;

      return m_CurrentLearningRate;
    }
  }
}
