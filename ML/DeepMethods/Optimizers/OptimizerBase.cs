using System;
using ML.Contracts;
using ML.Core;

namespace ML.DeepMethods.Optimizers
{
  /// <summary>
  /// Base class for all optimizers
  /// </summary>
  public abstract class OptimizerBase : IOptimizer
  {
    protected double[][] m_Weights;
    protected double m_Step2;

    protected OptimizerBase()
    {
    }


    /// <summary>
    /// Last weight vector step value (squared)
    /// </summary>
    public double Step2 { get { return m_Step2; } }


    /// <summary>
    /// Set source weight vector
    /// </summary>
    public virtual void Init(double[][] weights)
    {
      if (weights==null)
        throw new MLException("Weights can not be null");

      m_Weights = weights;
    }

    /// <summary>
    /// Push current gradient vector to optimizer
    /// </summary>
    public abstract void Push(double[][] gradient, double learningRate);
  }
}
