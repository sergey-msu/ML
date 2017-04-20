using System.Collections.Generic;
using ML.DeepMethods.LossFunctions;
using ML.DeepMethods.LearningRateSchedulers;
using ML.DeepMethods.Optimizers;

namespace ML.DeepMethods.Registry
{
  public static class Loss
  {
    private static readonly Dictionary<double, LpLoss> m_Lp = new Dictionary<double, LpLoss>();

    public static readonly EuclideanLoss           Euclidean           = new EuclideanLoss();
    public static readonly CrossEntropyLoss        CrossEntropy        = new CrossEntropyLoss();
    public static readonly CrossEntropySoftMaxLoss CrossEntropySoftMax = new CrossEntropySoftMaxLoss();

    public static LpLoss Lp(double p)
    {
      LpLoss result;
      if (!m_Lp.TryGetValue(p, out result))
      {
        result = new LpLoss(p);
        m_Lp[p] = result;
      }

      return result;
    }
  }

  public static class Optimizer
  {
    public static readonly NopeOptimizer Nope = new NopeOptimizer();
  }

  public static class LearningRateScheduler
  {
     public static NopeScheduler Nope(double initLearningRate)
     {
       return new NopeScheduler(initLearningRate);
     }

     public static TimeBasedScheduler TimeBased(double initLearningRate, double decay)
     {
       return new TimeBasedScheduler(initLearningRate, decay);
     }

     public static DropBasedScheduler DropBased(double initLearningRate, double epochStep, double dropRate)
     {
       return new DropBasedScheduler(initLearningRate, epochStep, dropRate);
     }
  }
}


