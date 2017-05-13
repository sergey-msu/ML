using System;
using System.Collections.Generic;
using ML.Core;
using ML.DeepMethods.LossFunctions;
using ML.DeepMethods.LearningRateSchedulers;
using ML.DeepMethods.Optimizers;
using ML.DeepMethods.Regularization;

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

  public static class Regularizator
  {
    private static readonly Dictionary<double, L2Regularizator> m_L2 = new Dictionary<double, L2Regularizator>();

    public static L2Regularizator L2(double coeff)
    {
      return Core.Utils.GetThroughMap(coeff, m_L2);
    }
  }

  public static class Optimizer
  {
    public static readonly SGDOptimizer      SGD      = new SGDOptimizer();
    public static readonly AdadeltaOptimizer Adadelta = new AdadeltaOptimizer();
    public static readonly AdagradOptimizer  Adagrad  = new AdagradOptimizer();
    public static readonly AdamaxOptimizer   Adamax   = new AdamaxOptimizer();
    public static readonly AdamOptimizer     Adam     = new AdamOptimizer();
    public static readonly MomentumOptimizer Momentum = new MomentumOptimizer();
    public static readonly RMSPropOptimizer  RMSProp  = new RMSPropOptimizer();
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


