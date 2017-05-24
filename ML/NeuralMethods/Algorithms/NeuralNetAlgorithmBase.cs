using System;
using System.Collections.Generic;
using ML.Contracts;
using ML.Core;
using ML.NeuralMethods.Models;

namespace ML.NeuralMethods.Algorithms
{
  /// <summary>
  /// Feedforward Neural Network machine learning algorithm
  /// </summary>
  public abstract class NeuralNetAlgorithmBase : MultiRegressionAlgorithmBase<double[]>
  {
    private NeuralNetwork m_Net;

    protected NeuralNetAlgorithmBase(NeuralNetwork net)
    {
      if (net==null)
        throw new MLException("Network can not be null");

      m_Net = net;
    }

    /// <summary>
    /// The result of the algorithm
    /// </summary>
    public NeuralNetwork Net { get { return m_Net; } }

    /// <summary>
    /// Maps object to corresponding class
    /// </summary>
    public override double[] Predict(double[] x)
    {
      return m_Net.Calculate(x);
    }


    public override IEnumerable<ErrorInfo<double[], double[]>> GetErrors(MultiRegressionSample<double[]> testSample, double threshold, bool parallel)
    {
      var isTraining = m_Net.IsTraining;
      m_Net.IsTraining = false;
      try
      {
        return base.GetErrors(testSample, threshold, parallel);
      }
      finally
      {
        m_Net.IsTraining = isTraining;
      }
    }

    /// <summary>
    /// Teaches algorithm, produces Result output
    /// </summary>
    protected sealed override void DoTrain()
    {
      m_Net.IsTraining = true;
      try
      {
        TrainImpl();
      }
      finally
      {
        m_Net.IsTraining = false;
      }
    }


    protected abstract void TrainImpl();
  }
}
