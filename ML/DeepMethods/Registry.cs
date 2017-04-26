using System;
using System.Collections.Generic;
using ML.Core;
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
      return Core.Utils.GetThroughMap(p, m_Lp);
    }
  }

  public static class Optimizer
  {
    private static readonly Dictionary<double, MomentumOptimizer> m_Momentum = new Dictionary<double, MomentumOptimizer>();
    private static readonly Dictionary<double, AdagradOptimizer>  m_Adagrad  = new Dictionary<double, AdagradOptimizer>();
    private static readonly Dictionary<double, AdadeltaOptimizer> m_Adadelta = new Dictionary<double, AdadeltaOptimizer>();
    private static readonly Dictionary<double, RMSPropOptimizer>  m_RMSProp  = new Dictionary<double, RMSPropOptimizer>();
    private static readonly Dictionary<double, AdamOptimizer>     m_Adam     = new Dictionary<double, AdamOptimizer>();

    public static readonly SGDOptimizer SGD = new SGDOptimizer();


    public static MomentumOptimizer Momentum(double mu)
    {
      return Core.Utils.GetThroughMap(mu, m_Momentum);
    }

    public static AdagradOptimizer Adagrad(double epsilon)
    {
      return Core.Utils.GetThroughMap(epsilon, m_Adagrad);
    }

    public static AdadeltaOptimizer Adadelta(double gamma, double epsilon, bool useLearningRate)
    {
      return new AdadeltaOptimizer(gamma, epsilon, useLearningRate);
    }

    public static RMSPropOptimizer RMSProp(double gamma, double epsilon)
    {
      return new RMSPropOptimizer(gamma, epsilon);
    }

    public static AdamOptimizer Adam(double beta1, double beta2, double epsilon)
    {
      return new AdamOptimizer(beta1, beta2, epsilon);
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


