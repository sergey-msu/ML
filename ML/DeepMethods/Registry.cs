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
    private static readonly Dictionary<double, MomentumOptimizer> m_Momentum = new Dictionary<double, MomentumOptimizer>();
    private static readonly Dictionary<double, AdagradOptimizer> m_Adagrad = new Dictionary<double, AdagradOptimizer>();
    private static readonly Dictionary<double, RMSPropOptimizer> m_RMSProp = new Dictionary<double, RMSPropOptimizer>();

    public static readonly SGDOptimizer SGD = new SGDOptimizer();


    public static MomentumOptimizer Momentum(double mu)
    {
      MomentumOptimizer result;
      if (!m_Momentum.TryGetValue(mu, out result))
      {
        result = new MomentumOptimizer(mu);
        m_Momentum[mu] = result;
      }

      return result;
    }

    public static AdagradOptimizer Adagrad(double eps)
    {
      AdagradOptimizer result;
      if (!m_Adagrad.TryGetValue(eps, out result))
      {
        result = new AdagradOptimizer(eps);
        m_Adagrad[eps] = result;
      }

      return result;
    }

    public static RMSPropOptimizer RMSProp(double epsilon, double gamma)
    {
      return new RMSPropOptimizer(epsilon, gamma);
    }
  }

  public static class LearningRateScheduler
  {
     public static ConstantScheduler Constant(double initLearningRate)
     {
       return new ConstantScheduler(initLearningRate);
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


