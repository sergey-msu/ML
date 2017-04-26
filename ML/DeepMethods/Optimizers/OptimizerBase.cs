using System;
using ML.Contracts;
using ML.Core;

namespace ML.DeepMethods.Optimizers
{
  /// <summary>
  /// Base class for all optimizers.
  /// (see http://caffe.berkeleyvision.org/tutorial/solver.html for standart optimizers description)
  /// </summary>
  public abstract class OptimizerBase : IOptimizer
  {
    protected double m_Step2;

    protected OptimizerBase()
    {
    }


    /// <summary>
    /// Last weight vector step value (squared)
    /// </summary>
    public double Step2 { get { return m_Step2; } }


    /// <summary>
    /// Push current gradient vector to optimizer
    /// </summary>
    public void Push(double[][] weights, double[][] gradient, double learningRate)
    {
      if (weights==null)
        throw new MLException("Weights is null");
      if (gradient==null)
        throw new MLException("Gradient in null");

      DoPush(weights, gradient, learningRate);
    }


    protected abstract void DoPush(double[][] weights, double[][] gradient, double learningRate);
  }
}
