using System;
using System.Collections.Generic;
using System.Linq;
using ML.Core;
using ML.Core.Mathematics;
using ML.DeepMethods.Models;
using ML.Contracts;

namespace ML.DeepMethods.Algorithms
{
  /// <summary>
  /// Feedforward Convolutional Neural Network machine learning algorithm
  /// </summary>
  public abstract class ConvNetAlgorithmBase : MultiRegressionAlgorithmBase<double[][,]>
  {
    private ConvNet m_Net;

    protected ConvNetAlgorithmBase(ConvNet net)
    {
      if (net==null)
        throw new MLException("Network can not be null");

      m_Net = net;
    }

    /// <summary>
    /// The result of the algorithm
    /// </summary>
    public ConvNet Net { get { return m_Net; } }


    /// <summary>
    /// Maps object to corresponding class
    /// </summary>
    public override double[] Predict(double[][,] x)
    {
      var result = m_Net.Calculate(x);
      var len = result.Length;

      var flatResult = new double[len];
      for (int i=0; i<len; i++)
        flatResult[i] = result[i][0,0];

      return flatResult;
    }

    public override IEnumerable<ErrorInfo<double[][,], double[]>> GetErrors(MultiRegressionSample<double[][,]> testSample, double threshold, bool parallel)
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
        TrainImpl(TrainingSample);
      }
      finally
      {
        m_Net.IsTraining = false;
      }
    }


    protected abstract void TrainImpl(MultiRegressionSample<double[][,]> sample);

    public abstract void Build();
  }
}
